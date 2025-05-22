#!/usr/bin/env python
# train_ddp_accum.py – Distributed training with optional gradient accumulation
# -----------------------------------------------------------------------------
#  This is a lightly‑refactored version of the user‑supplied train_ddp.py.  The
#  only functional change is **gradient accumulation** so that you can train
#  with an effectively larger batch size even when memory is limited.  Enable
#  it via the new --accum_steps CLI flag (default: 1 → no accumulation).
# -----------------------------------------------------------------------------

import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
import torch.multiprocessing as mp

# project‑specific imports — unchanged from the original file
from RSAM import SAM2UNet
from preprocess.parser import Parser, SemanticKitti
from utils.utils import *
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from schedulefree import AdamWScheduleFree

# -----------------------------------------------------------------------------
#  LOSS FUNCTIONS (identical to the original script)
# -----------------------------------------------------------------------------

def soft_iou_loss(logits, mask, num_classes, smooth=1e-6, ignore_index=0):
    probs = F.softmax(logits, dim=1)
    with torch.no_grad():
        gt = torch.eye(num_classes, device=mask.device)[mask]
        gt = gt.permute(0, 3, 1, 2).float()
    valid = (mask != ignore_index).unsqueeze(1).float()
    probs = probs * valid
    gt = gt * valid
    dims = (0, 2, 3)
    intersection = probs.mul(gt).sum(dim=dims)
    total = probs.add(gt).sum(dim=dims)
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou = torch.cat([iou[:ignore_index], iou[ignore_index + 1:]], dim=0)
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
# Dice, Lovasz, Boundary, and Combined losses (unchanged)
# -----------------------------------------------------------------------------

def dice_loss(logits, mask, num_classes, ignore_index=0, smooth=1e-6):
    probs = F.softmax(logits, dim=1)
    gt = F.one_hot(mask.clamp(min=0), num_classes)
    gt = gt.permute(0, 3, 1, 2).float()
    valid = (mask != ignore_index).unsqueeze(1).float()
    probs = probs * valid
    gt = gt * valid
    dims = (0, 2, 3)
    inter = (probs * gt).sum(dims)
    union = probs.sum(dims) + gt.sum(dims)
    dice = (2 * inter + smooth) / (union + smooth)
    dice = dice[1:]
    return 1.0 - dice.mean()


def lovasz_softmax(logits, mask, ignore_index=0):
    C = logits.size(1)
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = mask.view(-1)
    valid = labels_flat != ignore_index
    logits_flat = logits_flat[valid]
    labels_flat = labels_flat[valid]
    if labels_flat.numel() == 0:
        return torch.tensor(0., device=logits.device)

    prob = F.softmax(logits_flat, dim=1)
    gt = F.one_hot(labels_flat, C).float()

    losses = []
    for c in range(1, C):
        fg = gt[:, c]
        if fg.sum() == 0:
            continue
        errors = (prob[:, c] - fg).abs()
        errs_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        gtsum = fg.sum()
        intersection = gtsum - fg_sorted.cumsum(0)
        union = gtsum + (1 - fg_sorted).cumsum(0)
        jacc = 1. - intersection / union
        if jacc.numel() > 1:
            grad = jacc.clone()
            grad[1:] = jacc[1:] - jacc[:-1]
        else:
            grad = jacc
        losses.append((errs_sorted * grad).sum())
    if not losses:
        return torch.tensor(0., device=logits.device)
    return sum(losses) / len(losses)


def boundary_loss(logits, mask, num_classes, ignore_index=0):
    with torch.no_grad():
        gt = F.one_hot(mask.clamp(min=0), num_classes)
        gt = gt.permute(0, 3, 1, 2).float()
        gt = gt[:, 1:, :, :]
        kx = torch.tensor(
            [[1.0, 0.0, -1.0],
             [2.0, 0.0, -2.0],
             [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=gt.device
        ).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        edges = []
        for c in range(gt.size(1)):
            g = gt[:, c:c + 1]
            ex = F.conv2d(g, kx, padding=1)
            ey = F.conv2d(g, ky, padding=1)
            edges.append(torch.sqrt(ex ** 2 + ey ** 2))
        boundary_map = torch.clamp(torch.cat(edges, 1), 0, 1)
        boundary_map = boundary_map.max(dim=1, keepdim=True)[0]

    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(mask.clamp(min=0), num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    valid = (mask != ignore_index).unsqueeze(1).float()

    diff = (probs - one_hot).abs() * valid
    b_loss = (diff * boundary_map).sum() / (boundary_map.sum() + 1e-6)
    return b_loss


def combined_loss(logits, mask, num_classes=20,
                  w_ce=0.5, w_dice=0.5, w_lovasz=1.5, w_boundary=1):
    freqs = torch.tensor([
        0.03150183342534689, 0.04260782867450238, 0.00016609538710765,
        0.00039838616015114, 0.00216493982413381, 0.00180705529788636,
        0.00033758327431050, 0.00012711105887399, 0.00003746106399997,
        0.19879647126983287, 0.01471716954988821, 0.14392298360372000,
        0.00390485530374720, 0.13268619447774860, 0.07235922294562230,
        0.26681502148037506, 0.00603501201262603, 0.07814222006271769,
        0.00285549819386317, 0.00061559580861899,
    ], dtype=torch.float32, device=logits.device)

    eps = 1e-6
    median = freqs[1:].median()
    w_c = median / (freqs + eps)
    w_c[0] = 0.0
    w_c = w_c * (len(w_c) / w_c.sum())

    ce = F.cross_entropy(logits, mask, ignore_index=0, reduction='mean', weight=w_c)
    d = dice_loss(logits, mask, num_classes, ignore_index=0)
    l = lovasz_softmax(logits, mask, ignore_index=0)
    b = boundary_loss(logits, mask, num_classes, ignore_index=0)
    return w_ce * ce + w_dice * d + w_lovasz * l + w_boundary * b


# -----------------------------------------------------------------------------
#  METRICS & UTILITY FUNCTIONS (unchanged)
# -----------------------------------------------------------------------------

def evaluate_metrics(model, dataloader, num_classes=20, ignore_index=0, device='cuda'):
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for (in_vol, proj_mask, proj_labels, _, *_rest) in dataloader:
            x = in_vol.to(device, non_blocking=True)
            target = proj_labels.to(device, non_blocking=True)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            valid = target != ignore_index
            t = target[valid].view(-1)
            p = preds[valid].view(-1)
            idx = num_classes * t + p
            cm = torch.bincount(idx, minlength=num_classes ** 2)
            conf_matrix += cm.reshape(num_classes, num_classes)

    total = conf_matrix.sum().item()
    correct = conf_matrix.trace().item()
    pixel_acc = correct / total if total > 0 else 0.0

    class_acc = []
    counts = conf_matrix.sum(dim=1)
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        gt_count = counts[cls].item()
        if gt_count > 0:
            class_acc.append(conf_matrix[cls, cls].item() / gt_count)
    mean_acc = sum(class_acc) / len(class_acc) if class_acc else 0.0

    ious, freqs = [], []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
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


# -----------------------------------------------------------------------------
#  DDP SETUP / CLEANUP (unchanged)
# -----------------------------------------------------------------------------

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


# -----------------------------------------------------------------------------
#  DATA HELPERS (unchanged)
# -----------------------------------------------------------------------------

def initiate_parser(rank, world_size, batch_size):
    ARCH = load_yaml("config/arch/LENet.yaml")
    DATA = load_yaml("config/labels/semantic-kitti.yaml")

    train_dataset = SemanticKitti(
        root="dataset",
        sequences=DATA["split"]["train"],
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        transform=True,
        gt=True,
        drop_few_static_frames=False,
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    valid_dataset = SemanticKitti(
        root="dataset",
        sequences=DATA["split"]["valid"],
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        gt=True,
    )
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    validloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    return trainloader, validloader, train_sampler, val_sampler


# -----------------------------------------------------------------------------
#  TRAINING (gradient accumulation added 👍🏼)
# -----------------------------------------------------------------------------

def train(rank, args):
    setup_ddp(rank, args.world_size)
    seed_torch(1904 + rank)
    train_loader, val_loader, train_sampler, val_sampler = initiate_parser(rank, args.world_size, args.batch_size)

    device = torch.device(f"cuda:{rank}")
    model = SAM2UNet(
        "sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
        stem=args.stem,
        freeze_weight=args.freeze_weight,
        adapter=args.adapter,
        msca=args.msca,
        unify_dim=args.unify_dim,
        use_rfb=args.use_rfb,
        pos_emb=args.pos_emb,
    ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
    )

    backbone_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder.backbone" in n:
            backbone_params.append(p)
        else:
            other_params.append(p)
    assert backbone_params, "No backbone params found — check naming."
    assert other_params, "Everything landed in the backbone group?"

    optimizer = optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr / 10},
        ],
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.save_path, exist_ok=True)

    accum_steps = max(1, args.accum_steps)
    if rank == 0:
        print(f"Gradient accumulation: {accum_steps} step(s) → effective batch = {accum_steps * args.batch_size * args.world_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (in_vol, proj_mask, proj_labels, _, *_rest) in enumerate(train_loader, 1):
            x = in_vol.to(device, non_blocking=True)
            target = proj_labels.to(device, non_blocking=True).long()

            out_main, out_aux = model(x)
            weights = [1.0, 1.0, 0.5, 0.25]
            loss = combined_loss(out_main, target) + sum(w * combined_loss(aux_pred, target) for w, aux_pred in zip(weights, out_aux))
            loss_item = loss.item()

            # --- accumulate gradients ---
            loss = loss / accum_steps
            loss.backward()

            if (i % accum_steps == 0) or (i == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss_item
            if rank == 0 and i % 50 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}]  Batch Loss: {loss_item:.4f}")

        if rank == 0:
            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            # switch to eval mode on the wrapped module
            metrics = evaluate_metrics(model.module, val_loader, num_classes=20, ignore_index=0, device=device)
            if rank == 0:
                print(f"*** Validation metrics after epoch {epoch} ***")
                print(f"  Pixel Acc : {metrics['pixel_acc']:.4f}")
                print(f"  Mean Acc  : {metrics['mean_acc']:.4f}")
                print(f"  mIoU      : {metrics['miou']:.4f}")
                print(f"  fwIoU     : {metrics['fw_iou']:.4f}")
                ckpt = os.path.join(args.save_path, f'SAM2-UNet-epoch{epoch}-rank{rank}.pth')
                torch.save(model.module.state_dict(), ckpt)
                print(f"[Saved Snapshot:] {ckpt}")

    cleanup_ddp()


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM2-UNet DDP with Gradient Accumulation")
    parser.add_argument("--save_path", type=str, required=True, help="path to store the checkpoint")
    # MODEL CONFIG
    parser.add_argument("--stem", required=True, type=str)
    parser.add_argument("--freeze_weight", type=str2bool, required=True)
    parser.add_argument("--adapter", type=str2bool, required=True)
    parser.add_argument("--msca", type=str2bool, required=True)

    # TRAINING CONFIG
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=32, help="number of mini‑batches to accumulate before one optimizer step")
    parser.add_argument("--use_rfb", type=str2bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0.00025)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--unify_dim", type=int, default=256)
    parser.add_argument("--pos_emb", type=str2bool, default=False)

    args = parser.parse_args()

    if args.accum_steps < 1:
        raise ValueError("--accum_steps must be ≥ 1")

    print("Parsed Arguments:")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")

    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
