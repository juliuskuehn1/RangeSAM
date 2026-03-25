#!/usr/bin/env python
# train_ddp.py

import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.multiprocessing as mp
#from kitti_dataloader import MultiSeqNpZDataset, Seq02NpZDataset
from RSAMconv import SAM2UNet
from preprocess.parser import Parser, SemanticKitti
from utils.utils import *
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from schedulefree import AdamWScheduleFree

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, ShardedStateDictConfig
)
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


from sam2.modeling.backbones.hieradet import Hiera

torch.set_float32_matmul_precision('high')

def class_balanced_mean(loss, mask, ignore_index=0, num_classes=None):
    """
    loss:       torch.Tensor of shape [B,H,W], no reduction yet
    mask:       torch.LongTensor of shape [B,H,W]
    ignore_index:  label to ignore
    num_classes:   total # of classes C  (including ignore)
    """
    if num_classes is None:
        num_classes = int(mask.max()) + 1

    # 1) flatten and mask
    valid = (mask != ignore_index)
    loss_flat = loss[valid]                # [N_valid]
    mask_flat = mask[valid]                # [N_valid]

    # 2) compute per-class sums and counts
    C = num_classes
    device = loss.device
    sum_per_class   = torch.zeros(C, device=device)
    count_per_class = torch.zeros(C, device=device)

    sum_per_class   = sum_per_class.scatter_add(0, mask_flat, loss_flat)
    count_per_class = count_per_class.scatter_add(0, mask_flat, torch.ones_like(loss_flat))

    # 3) for each class present, compute its mean
    present = count_per_class > 0
    mean_per_class = sum_per_class[present] / count_per_class[present]

    # 4) average the class-means
    if mean_per_class.numel() == 0:
        return torch.tensor(0.0, device=device)
    return mean_per_class.mean()

# -----------------------------------------------------------------------------
# 1) Dice loss (soft, averaged over classes 1…C-1)
# -----------------------------------------------------------------------------
def focal_cross_entropy(logits, mask, gamma=2.0, alpha=None, ignore_index=0, reduction='mean'):
    """
    Focal loss with optional per-class α weighting.
      logits:     [B, C, H, W]
      mask:       [B, H, W] integer labels in [0..C-1]
      gamma:      focusing parameter
      alpha:      tensor[C] of class weights, or None
    """
    # 1) standard per-pixel CE
    ce = F.cross_entropy(logits, mask,
                         ignore_index=ignore_index,
                         reduction='none')           # [B,H,W]
    p_t = torch.exp(-ce)                       # [B,H,W] prob of true class

    # 2) focal scaling
    loss = (1 - p_t)**gamma * ce               # [B,H,W]

    # 3) α weighting
    if alpha is not None:
        # mask.clamp to avoid indexing -1 for ignore_index
        idx = mask.clamp(min=0)        # still [B,H,W]
        a   = alpha[idx]               # now [B,H,W]
        loss = a * loss
        
    return class_balanced_mean(loss, mask, ignore_index=0, num_classes=20)



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
def new_boundary_loss(logits, mask, theta0=3):
    """
    Compute the boundary F1 loss between prediction and ground-truth,
    ignoring class 0.
    """
    N, C, H, W = logits.shape
    # 1) probabilities
    pred = F.softmax(logits, dim=1)               # [N,C,H,W]

    # 2) one‐hot encode gt
    one_hot = F.one_hot(mask, num_classes=C)      # [N,H,W,C]
    one_hot = one_hot.permute(0,3,1,2).float()    # [N,C,H,W]

    # 3) extract class‐wise boundary maps
    pad = (theta0 - 1) // 2
    gt_b   = F.max_pool2d(1 - one_hot, kernel_size=theta0, stride=1, padding=pad) \
             - (1 - one_hot)
    pred_b = F.max_pool2d(1 - pred,    kernel_size=theta0, stride=1, padding=pad) \
             - (1 - pred)

    # 4) drop class 0
    gt_b   = gt_b[:, 1:, :, :]   # now [N, C-1, H, W]
    pred_b = pred_b[:, 1:, :, :]
    C1 = C - 1

    # 5) flatten spatial dims
    gt_b   = gt_b.view(N, C1, -1)     # [N, C-1, H*W]
    pred_b = pred_b.view(N, C1, -1)

    # 6) precision & recall per class
    eps = 1e-7
    inter = (pred_b * gt_b).sum(dim=2)                   # [N, C1]
    P = inter / (pred_b.sum(dim=2) + eps)
    R = inter / (gt_b.sum(dim=2)   + eps)

    # 7) boundary F1 and loss
    BF1  = 2 * P * R / (P + R + eps)    # [N, C1]
    loss = 1 - BF1

    return loss.mean()

###########
# Dice loss
###########

def dice_loss(logits, mask, num_classes, ignore_index=0, smooth=1e-6):
    probs = F.softmax(logits, 1)
    gt    = F.one_hot(mask.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()
    valid = (mask != ignore_index).unsqueeze(1).float()
    probs, gt = probs * valid, gt * valid
    inter = (probs * gt).sum((0, 2, 3))
    union = probs.sum((0, 2, 3)) + gt.sum((0, 2, 3))
    dice  = (2 * inter + smooth) / (union + smooth)
    dice  = dice[1:]
    return 1. - dice.mean()

def generalized_dice_loss(logits,
                           mask,
                           weights=None,      # 1-D tensor [C]
                           ignore_index=0,
                           smooth=1e-6):
    """
    Sudre et al., ‘Generalised Dice overlap…’ (2017)
    """
    B, C, H, W = logits.shape
    device = logits.device

    # ----- one-hot ground truth, skip ignore -----
    valid = mask != ignore_index
    gt = F.one_hot(mask.clamp(min=0), C).permute(0,3,1,2).float() * valid.unsqueeze(1)
    prob = F.softmax(logits, 1) * valid.unsqueeze(1)

    # ----- class weights -----
    if weights is None:
        freq = gt.sum((0,2,3)) + smooth          # per-class voxel count
        w = 1.0 / (freq * freq)                  # 1 / freq²
    else:
        w = weights.to(device).float()
    w = w / (w.max() + 1e-6)                     # optional normalisation
    if ignore_index < C:
        w[ignore_index] = 0                      # make sure ignore class is zero-weighted

    # ----- numerator & denominator (scalars) -----
    intersect = torch.sum(w * (prob * gt).sum((0,2,3)))     # scalar
    union     = torch.sum(w * (prob + gt).sum((0,2,3)))     # scalar

    dice = (2 * intersect + smooth) / (union + smooth)
    return 1.0 - dice

# -----------------------------------------------------------------------------
# 4) Combined loss
# -----------------------------------------------------------------------------
def combined_loss(logits, mask, num_classes=20,
                  w_ce=1, w_lovasz=1, w_dice=1, w_boundary=1):
    """
    A mix of four losses:
      - standard cross-entropy
      - soft Dice
      - Lovasz-Softmax
      - boundary-sensitive
    """
    w_c = torch.tensor([
    30.767496,   # 0: unlabeled
    22.931664,   # 1: car
   857.562744,   # 2: bicycle
   715.110046,   # 3: motorcycle
   315.961761,   # 4: truck
   356.245178,   # 5: other-vehicle
   747.617004,   # 6: person
   887.223938,   # 7: bicyclist
   963.891541,   # 8: motorcyclist
     5.005093,   # 9: road
    63.624687,   # 10: parking
     6.900217,   # 11: sidewalk
   203.879608,   # 12: other-ground
     7.480204,   # 13: building
    13.631550,   # 14: fence
     3.733921,   # 15: vegetation
   142.146164,   # 16: trunk
    12.635481,   # 17: terrain
   259.369873,   # 18: pole
   618.966736    # 19: traffic-sign
    ], dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    w_c = w_c/w_c.mean()
    w_c = w_c.to(logits.dtype)
    ce = F.cross_entropy(logits, mask,
                         weight=w_c,
                         ignore_index=0,
                         reduction='mean')
    # 3) Lovasz
    l  = lovasz_softmax(logits, mask, ignore_index=0)
    # 4) Boundary
    b  = new_boundary_loss(logits, mask)
    return w_ce*ce + w_lovasz*l + w_boundary*b


def evaluate_metrics(model, dataloader, num_classes=20,
                     ignore_index=0, device='cuda'):
    # build confusion matrix
    conf_matrix = torch.zeros((num_classes, num_classes),
                              dtype=torch.long, device=device)
    model.eval()
    # optimizer.eval()
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
    
@torch.no_grad()
def evaluate_metrics_ddp(model,
                         dataloader,
                         num_classes: int = 20,
                         ignore_index: int = 0,
                         device: torch.device | None = None):
    """
    Computes pixel-accuracy, mean-accuracy, mIoU and fwIoU on the *full*
    validation set in a DDP run.

    • Each rank accumulates its local confusion-matrix.  
    • `dist.all_reduce` sums those matrices across ranks.  
    • Rank 0 returns/prints the metrics (others still return a dict so the
      call is symmetric).

    NOTE: the DataLoader may still use DistributedSampler; that is fine as
    long as we aggregate here.
    """
    if device is None:
        device = next(model.parameters()).device

    # full C×C confusion matrix on *this* GPU
    conf = torch.zeros(num_classes,
                       num_classes,
                       dtype=torch.long,
                       device=device)

    model.eval()
    for batch in dataloader:
        in_vol      = batch[0].to(device, non_blocking=True)   # projected voxels
        target      = batch[2].to(device, non_blocking=True)   # proj_labels
        logits, _   = model(in_vol)                            # forward
        pred        = logits.argmax(1)

        valid = target != ignore_index
        t     = target[valid].view(-1)
        p     = pred[valid].view(-1)
        idx   = num_classes * t + p
        conf += torch.bincount(idx,
                               minlength=num_classes**2
                               ).view(num_classes, num_classes)

    # ---- gather from every GPU ----
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(conf, op=dist.ReduceOp.SUM)

    # ---- derive metrics ----
    total   = conf.sum().item()
    correct = conf.trace().item()
    pixel_acc = correct / total if total else 0.0

    class_acc = []
    ious, freqs = [], []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        gt = conf[cls].sum().item()
        if gt:
            class_acc.append(conf[cls, cls].item() / gt)

        tp = conf[cls, cls].item()
        fp = conf[:, cls].sum().item() - tp
        fn = conf[cls, :].sum().item() - tp
        denom = tp + fp + fn
        if denom:
            iou = tp / denom
            ious.append(iou)
            freqs.append(conf[cls, :].sum().item() / total)
            
    cls_list = [c for c in range(num_classes) if c != ignore_index]
    per_class_iou = {
        cls: float(iou)
        for cls, iou in zip(cls_list, ious)
    }
    
    mean_acc = sum(class_acc) / len(class_acc) if class_acc else 0.0
    miou     = sum(ious)      / len(ious)      if ious      else 0.0
    fw_iou   = sum(f * i for f, i in zip(freqs, ious))       if ious else 0.0

    return {
        "pixel_acc": pixel_acc,
        "mean_acc":  mean_acc,
        "miou":      miou,
        "fw_iou":    fw_iou,
        "per_class_iou": per_class_iou
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


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()

def initiate_parser(rank, world_size, batch_size):
    ARCH = load_yaml("config/arch/LENet.yaml")
    DATA = load_yaml("config/labels/semantic-kitti.yaml")
    train_dataset =SemanticKitti(root="dataset",
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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   drop_last=False)
    valid_dataset = SemanticKitti(root="dataset",
                                  sequences=DATA["split"]["valid"],
                                  labels=DATA["labels"],
                                  color_map=DATA["color_map"],
                                  learning_map=DATA["learning_map"],
                                  learning_map_inv=DATA["learning_map_inv"],
                                  sensor=ARCH["dataset"]["sensor"],
                                  max_points=ARCH["dataset"]["max_points"],
                                  gt=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    validloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   sampler=val_sampler,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   drop_last=True)
    return trainloader, validloader, train_sampler, val_sampler

def auto_wrap_policy(module, recurse, nonwrapped_numel):
    if isinstance(module, Hiera):
        return False
    return size_based_auto_wrap_policy(
        module,
        recurse,
        nonwrapped_numel,
        min_num_params=int(1e7),
    )
    
class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer
    def state_dict(self):
        opts = StateDictOptions(
            full_state_dict=True,      # gather the full tensor on each param
            cpu_offload=True,          # offload to CPU as you go to save GPU memory
        )
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer, options=opts)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

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
        unify_dim=args.unify_dim,   # e.g. 256
        use_rfb=args.use_rfb,        # True / False
        pos_emb=args.pos_emb,
    )
    backbone_params = []
    other_params    = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # 1) pos-embeddings of the backbone → other_params
        if "encoder.backbone" in n and "pos_embed" in n:
            other_params.append(p)

        # 2) remaining backbone weights → backbone_params
        elif "encoder.backbone" in n:
            backbone_params.append(p)

        # 3) everything else → other_params
        else:
            other_params.append(p)
    # mixed_precision = MixedPrecision(
    #     param_dtype=torch.bfloat16,   # fp16 params & grads
    #     reduce_dtype=torch.bfloat16,
    #     buffer_dtype=torch.bfloat16,
    # )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  
        device_id=rank,         #mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        forward_prefetch=True,           # helps throughput for encoder/decoder nets
        use_orig_params=True,            # keeps original Param semantics (PyTorch ≥2.2)
    )
    # assert backbone_params, "didn't catch any backbone parameters — check the name!"
    # assert other_params,    "everything landed in the backbone group?"
    optimizer = optim.AdamW(
    [
        {"params": other_params,    "lr": args.lr},   # heads, adapters, decoder
        {"params": backbone_params, "lr": args.backbone_lr},   # fine‑tune SAM2 trunk
    ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)            # batches per epoch on this GPU
    warmup_epochs   = args.warmup_epochs
    warmup_steps    = warmup_epochs * steps_per_epoch
    total_steps     = args.epochs * steps_per_epoch
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,     # e.g. 0.1
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.min_lr,                       # e.g. 1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],                 # switch after warm-up
    )
    mIoU = 0
    os.makedirs(args.save_path, exist_ok=True)
    weights = [1.0, 1.0, 1.0, 1.0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        # optimizer.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, (in_vol, _, proj_labels, _, _, _,
                _, _, _, _, _, _, _, _, _) in enumerate(train_loader, 1):
            x = in_vol.to(device, non_blocking=True)
            target = proj_labels.to(device, non_blocking=True).long()
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_main, out_aux= model(x)
            loss = combined_loss(out_main, target)  + sum(w * combined_loss(aux_pred, target) for w, aux_pred in zip(weights, out_aux))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if rank == 0 and i % 50 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}]  Batch Loss: {loss.item():.4f}")
                
        if rank == 0:
            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}")
        
        if epoch >= 0 or epoch % 5 == 0 or epoch == args.epochs:
            metrics = evaluate_metrics_ddp(model, val_loader, num_classes=20, ignore_index=0, device=device)
            if rank == 0:
                print("Matrix:")
                print(f"*** Validation metrics after epoch {epoch} ***")
                print(f"  Pixel Acc: {metrics['pixel_acc']:.4f}")
                print(f"  Mean  Acc: {metrics['mean_acc']:.4f}")
                print(f"  mIoU     : {metrics['miou']:.4f}")
                print(f"  fwIoU    : {metrics['fw_iou']:.4f}")
                print("Per-class IoUs:")
                for cls, iou in metrics['per_class_iou'].items():
                    print(f"  Class {cls:2d}: {iou:.4f}")
                if metrics['miou'] > mIoU:
                    print("New best mIoU!")
                    #ckpt = os.path.join(args.save_path, f'SAM2-UNet-epoch{epoch}-rank{rank}.pth')
                    mIoU = metrics['miou']
                    # fsdp_state = get_state_dict(
                    #     model,
                    #     optimizer,
                    #     options=opts
                    # )
                    # torch.save(fsdp_state, ckpt)
                    # print(f"[Saved Snapshot:] {ckpt}")
            ckpt = os.path.join(args.save_path, f'SAM2-UNet-epoch{epoch}-rank{rank}.pth')
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                        model, StateDictType.FULL_STATE_DICT, save_policy
                    ):
                        cpu_state = model.state_dict()
            if rank == 0:
                torch.save(cpu_state, ckpt)
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM2-UNet DDP")
    # parser.add_argument("--hiera_path", type=str, default=None,
    #                     help="path to the sam2 pretrained hiera (default: sam2.1_hiera_tiny.pt)")
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
    parser.add_argument("--use_rfb", default=False, type=str2bool)
    parser.add_argument("--weight_decay", default=0.00025, type=float)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(),
                        help="number of GPUs for DDP")
    parser.add_argument("--unify_dim", type=int, default=256)
    parser.add_argument("--pos_emb", default=False, type=str2bool)
    parser.add_argument("--warmup_epochs",       type=int,   default=5,
                    help="linear warm-up duration (in epochs)")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                        help="initial LR = base_lr × start_factor during warm-up")
    parser.add_argument("--min_lr",              type=float, default=1e-6,
                        help="η_min for cosine annealing phase")
    parser.add_argument("--backbone_lr",              type=float, default=0.001,
                        help="lr for backbone")
    parser.add_argument("--offload",          type=str2bool, required=False,
                        help="Whether to offload to CPU")
    ##########
    args = parser.parse_args()
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg:20}: {value}")
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
