import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from RSAM import SAM2UNet
from kitti_dataloader import Seq02NpZDataset
from preprocess.parser import Parser, SemanticKitti
from utils.utils import *

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to store the checkpoint")
########## MODEL CONFIG
parser.add_argument("--stem", required=True, type=str)
parser.add_argument("--freeze_weight", type=str2bool, required=True,
                    help="Whether to freeze the weights (True/False)")
parser.add_argument("--adapter",       type=str2bool, required=True,
                    help="Whether to enable the adapter (True/False)")
parser.add_argument("--msca",          type=str2bool, required=True,
                    help="Whether to use MSCA (True/False)")
##########
parser.add_argument("--epochs", type=int, default=20,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(),
                    help="number of GPUs for DDP")
parser.add_argument("--unify_dim", type=int, default=256)
args = parser.parse_args()
print("Parsed Arguments:")
for arg, value in vars(args).items():
    print(f"{arg:20}: {value}")



def soft_iou_loss(logits, mask, num_classes, smooth=1e-6, ignore_index=0):
    probs = F.softmax(logits, dim=1)
    with torch.no_grad():
        gt = torch.eye(num_classes, device=mask.device)[mask]
        gt = gt.permute(0,3,1,2).float()
    valid = (mask != ignore_index).unsqueeze(1).float()
    probs = probs * valid
    gt    = gt    * valid
    dims = (0,2,3)
    intersection = probs.mul(gt).sum(dim=dims)
    total        = probs.add(gt).sum(dim=dims)
    union        = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou = torch.cat([iou[:ignore_index], iou[ignore_index+1:]], dim=0)
    return 1.0 - iou.mean()


def structure_loss(logits, mask, num_classes=20,
                   weight_ce=0.3, weight_iou=0.7):
    ce = F.cross_entropy(logits, mask,
                         ignore_index=0,
                         reduction='mean')
    iou = soft_iou_loss(logits, mask, num_classes,
                        ignore_index=0)
    return weight_ce * ce + weight_iou * iou

# -----------------------------------------------------------------------------
# 1) Dice loss (soft, averaged over classes 1…C-1)
# -----------------------------------------------------------------------------
def dice_loss(logits, mask, num_classes, ignore_index=0, smooth=1e-6):
    probs = F.softmax(logits, dim=1)                              # [B,C,H,W]
    # one-hot GT
    gt = F.one_hot(mask.clamp(min=0), num_classes)                # [B,H,W,C]
    gt = gt.permute(0,3,1,2).float()                              # [B,C,H,W]
    # mask out ignore_index
    valid = (mask != ignore_index).unsqueeze(1).float()           # [B,1,H,W]
    probs = probs * valid
    gt    = gt    * valid

    # per-class intersection & union
    dims = (0,2,3)
    inter = (probs * gt).sum(dims)                                # [C]
    union = probs.sum(dims) + gt.sum(dims)                        # [C]
    dice  = (2*inter + smooth) / (union + smooth)                 # [C]

    # drop the void class
    dice = dice[1:]
    return 1.0 - dice.mean()


# -----------------------------------------------------------------------------
# 2) Lovász-Softmax (see: https://arxiv.org/abs/1705.08790)
# -----------------------------------------------------------------------------
def lovasz_softmax(logits, mask, ignore_index=0):
    C = logits.size(1)
    # flatten predictions & labels
    logits_flat = logits.permute(0,2,3,1).reshape(-1, C)   # [N,C]
    labels_flat = mask.view(-1)                            # [N]
    valid = labels_flat != ignore_index
    logits_flat = logits_flat[valid]
    labels_flat = labels_flat[valid]
    if labels_flat.numel() == 0:
        return torch.tensor(0., device=logits.device)

    prob = F.softmax(logits_flat, dim=1)                   # [M,C]
    gt = F.one_hot(labels_flat, C).float()                 # [M,C]

    losses = []
    for c in range(1, C):
        fg = gt[:, c]                                      # [M]
        if fg.sum() == 0:
            continue
        errors = (prob[:, c] - fg).abs()                   # [M]
        errs_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        # compute Jaccard curve
        gtsum = fg.sum()
        intersection = gtsum - fg_sorted.cumsum(0)
        union = gtsum + (1 - fg_sorted).cumsum(0)
        jacc = 1. - intersection / union                   # [M]

        # now compute the *gradient* of the Jaccard with respect to sorting
        if jacc.numel() > 1:
            grad = jacc.clone()
            grad[1:] = jacc[1:] - jacc[:-1]
        else:
            grad = jacc

        losses.append((errs_sorted * grad).sum())
    if not losses:
        return torch.tensor(0., device=logits.device)
    return sum(losses) / len(losses)


# -----------------------------------------------------------------------------
# 3) Boundary loss (penalize mistakes near class edges)
# -----------------------------------------------------------------------------
def boundary_loss(logits, mask, num_classes, ignore_index=0):
    # compute a simple boundary map from GT via Sobel filters
    with torch.no_grad():
        gt = F.one_hot(mask.clamp(min=0), num_classes)            # [B,H,W,C]
        gt = gt.permute(0,3,1,2).float()                           # [B,C,H,W]
        # only consider real classes
        gt = gt[:,1:,:,:]
        # Sobel kernels
        kx = torch.tensor(
            [[1.0, 0.0, -1.0],
             [2.0, 0.0, -2.0],
             [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=gt.device
        ).view(1, 1, 3, 3)

        ky = kx.transpose(2, 3)  # still float, same device
        edges = []
        for c in range(gt.size(1)):
            g = gt[:,c:c+1]
            ex = F.conv2d(g, kx, padding=1)
            ey = F.conv2d(g, ky, padding=1)
            edges.append(torch.sqrt(ex**2 + ey**2))
        boundary_map = torch.clamp(torch.cat(edges,1), 0, 1)       # [B,C-1,H,W]
        # collapse to a single-channel weight
        boundary_map = boundary_map.max(dim=1, keepdim=True)[0]   # [B,1,H,W]

    probs = F.softmax(logits, dim=1)                              # [B,C,H,W]
    one_hot = F.one_hot(mask.clamp(min=0), num_classes)           # [B,H,W,C]
    one_hot = one_hot.permute(0,3,1,2).float()                    # [B,C,H,W]
    valid = (mask != ignore_index).unsqueeze(1).float()           # [B,1,H,W]

    diff = (probs - one_hot).abs() * valid                        # [B,C,H,W]
    # weight only boundary pixels
    b_loss = (diff * boundary_map).sum() / (boundary_map.sum() + 1e-6)
    return b_loss


# -----------------------------------------------------------------------------
# 4) Combined loss
# -----------------------------------------------------------------------------
def combined_loss(logits, mask, num_classes=20,
                  w_ce=0.25, w_dice=0.25, w_lovasz=0.4, w_boundary=0.1):
    """
    A mix of four losses:
      - standard cross-entropy
      - soft Dice
      - Lovasz-Softmax
      - boundary-sensitive
    """
    freqs = torch.tensor([
    0.03150183342534689,  # 0: unlabeled (void)
    0.04260782867450238,  # 1: car
    0.00016609538710765,  # 2: bicycle
    0.00039838616015114,  # 3: motorcycle
    0.00216493982413381,  # 4: truck
    0.00180705529788636,  # 5: other-vehicle
    0.00033758327431050,  # 6: person
    0.00012711105887399,  # 7: bicyclist
    0.00003746106399997,  # 8: motorcyclist
    0.19879647126983287,  # 9: road
    0.01471716954988821,  # 10: parking
    0.14392298360372000,  # 11: sidewalk
    0.00390485530374720,  # 12: other-ground
    0.13268619447774860,  # 13: building
    0.07235922294562230,  # 14: fence
    0.26681502148037506,  # 15: vegetation
    0.00603501201262603,  # 16: trunk
    0.07814222006271769,  # 17: terrain
    0.00285549819386317,  # 18: pole
    0.00061559580861899,  # 19: traffic-sign
    ], dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    eps    = 1e-6
    median = freqs[1:].median()    
    w_c    = median / (freqs + eps)      # up‐weight rare classes
    w_c[0] = 0.0                         # ignore void
    # 1) CE (ignore void)
    ce = F.cross_entropy(logits, mask,
                         ignore_index=0, reduction='mean', weight=w_c)
    # 2) Dice
    d  = dice_loss(logits, mask, num_classes, ignore_index=0)
    # 3) Lovasz
    l  = lovasz_softmax(logits, mask, ignore_index=0)
    # 4) Boundary
    b  = boundary_loss(logits, mask, num_classes, ignore_index=0)

    return w_ce*ce + w_dice*d + w_lovasz*l + w_boundary*b



def evaluate_metrics(model, dataloader, num_classes=20,
                     ignore_index=0, device='cuda'):
    # build confusion matrix
    conf_matrix = torch.zeros((num_classes, num_classes),
                              dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                _, _, _, _, _, _, _, _, _) in dataloader:
            x = in_vol.to(device, non_blocking=True)
            target = proj_labels.to(device, non_blocking=True)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            valid = target != ignore_index
            t = target[valid].view(-1)
            p = preds[valid].view(-1)
            idx = num_classes * t + p
            cm = torch.bincount(idx, minlength=num_classes**2)
            conf_matrix += cm.reshape(num_classes, num_classes)
    # total and correct
    total = conf_matrix.sum().item()
    correct = conf_matrix.trace().item()
    pixel_acc = correct / total if total > 0 else 0.0
    # per-class accuracy
    class_acc = []
    counts = conf_matrix.sum(dim=1)
    for cls in range(num_classes):
        if cls == ignore_index: continue
        gt_count = counts[cls].item()
        if gt_count > 0:
            class_acc.append(conf_matrix[cls, cls].item() / gt_count)
    mean_acc = sum(class_acc) / len(class_acc) if class_acc else 0.0
    # per-class IoU and frequency
    ious, freqs = [], []
    for cls in range(num_classes):
        if cls == ignore_index: continue
        tp = conf_matrix[cls, cls].item()
        fp = conf_matrix[:, cls].sum().item() - tp
        fn = conf_matrix[cls, :].sum().item() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou = tp / denom
            ious.append(iou)
            freqs.append(conf_matrix[cls, :].sum().item() / total)
    miou = sum(ious) / len(ious) if ious else 0.0
    fw_iou = sum(f * i for f, i in zip(freqs, ious)) if ious else 0.0
    return {
        'pixel_acc': pixel_acc,
        'mean_acc': mean_acc,
        'miou': miou,
        'fw_iou': fw_iou,
    }


def seed_torch(seed=1904):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def initiate_parser():
    ARCH = load_yaml("config/arch/LENet.yaml")
    DATA = load_yaml("config/labels/semantic-kitti.yaml")
    train_data =SemanticKitti(root="dataset",
                              sequences=DATA["split"]["train"],
                              labels=DATA["labels"],
                              color_map=DATA["color_map"],
                              learning_map=DATA["learning_map"],
                              learning_map_inv=DATA["learning_map_inv"],
                              sensor=ARCH["dataset"]["sensor"],
                              max_points=ARCH["dataset"]["max_points"],
                              transform=True,
                              gt=True,
                              drop_few_static_frames=False
                              )
    trainloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   # shuffle=False,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   drop_last=True)
    valid_dataset = SemanticKitti(root="dataset",
                                  sequences=DATA["split"]["valid"],
                                  labels=DATA["labels"],
                                  color_map=DATA["color_map"],
                                  learning_map=DATA["learning_map"],
                                  learning_map_inv=DATA["learning_map_inv"],
                                  sensor=ARCH["dataset"]["sensor"],
                                  max_points=ARCH["dataset"]["max_points"],
                                  gt=True)

    validloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   drop_last=True)
    return trainloader, validloader
def main(args):
    # prepare dataset & loaders (same data for train/val)
    # dataset = Seq02NpZDataset("semk_cache02/seq_02")
    # dataset_val = Seq02NpZDataset("semk_cache02/seq_02_val")
    # train_loader = DataLoader(dataset, batch_size=args.batch_size,
    #                           shuffle=True, num_workers=8)
    # val_loader = DataLoader(dataset_val, batch_size=args.batch_size,
    #                         shuffle=False, num_workers=4)
    train_loader, val_loader = initiate_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SAM2UNet("sam2.1_hiera_t.yaml","sam2.1_hiera_tiny.pt", args.stem, args.freeze_weight, args.adapter, args.msca).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                _, _, _, _, _, _, _, _, _) in enumerate(train_loader, 1):
            x = in_vol.to(device, non_blocking=True)
            target = proj_labels.to(device, non_blocking=True).long()
            optimizer.zero_grad()
            out_main, out_aux= model(x)
            loss = combined_loss(out_main, target)  + sum(combined_loss(aux_pred, target) for aux_pred in out_aux)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}]  "
                      f"Batch Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}")

        # every 5 epochs: validate
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = evaluate_metrics(model, val_loader,
                                       num_classes=20,
                                       ignore_index=0,
                                       device=device)
            print("*** Validation metrics after epoch {} ***".format(epoch))
            print(f"  Pixel Acc: {metrics['pixel_acc']:.4f}")
            print(f"  Mean  Acc: {metrics['mean_acc']:.4f}")
            print(f"  mIoU     : {metrics['miou']:.4f}")
            print(f"  fwIoU    : {metrics['fw_iou']:.4f}")
            ckpt = os.path.join(args.save_path, f'SAM2-UNet-{epoch}.pth')
            torch.save(model.state_dict(), ckpt)
            print(f"[Saved Snapshot:] {ckpt}")

if __name__ == "__main__":
    seed_torch(1904)
    main(args)
