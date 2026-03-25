#!/usr/bin/env python
# train_ddp.py  –  AMP + checkpoint + efficient DDP (patched)

import os, argparse, random, yaml, shutil
from datetime import datetime
import numpy as np
import torch, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist, torch.multiprocessing as mp

from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DistributedSampler

# ------------------------------------------------------------------ #
# your project imports                                               #
# ------------------------------------------------------------------ #
from RSAM import SAM2UNet
from preprocess.parser import SemanticKitti
from utils.utils import load_yaml, str2bool
# ------------------------------------------------------------------ #

# ======================  LOSSES  ===================================

def soft_iou_loss(logits, mask, num_classes, smooth=1e-6, ignore_index=0):
    probs = F.softmax(logits, 1)
    with torch.no_grad():
        gt = torch.eye(num_classes, device=mask.device)[mask].permute(0, 3, 1, 2).float()
    valid = (mask != ignore_index).unsqueeze(1).float()
    probs, gt = probs * valid, gt * valid
    inter = (probs * gt).sum((0, 2, 3))
    union = probs.add(gt).sum((0, 2, 3)) - inter
    iou   = (inter + smooth) / (union + smooth)
    iou   = torch.cat([iou[:ignore_index], iou[ignore_index + 1:]])
    return 1. - iou.mean()

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

def lovasz_softmax(logits, mask, ignore_index=0):
    C = logits.size(1)
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = mask.view(-1)
    valid = labels_flat != ignore_index
    logits_flat, labels_flat = logits_flat[valid], labels_flat[valid]
    if labels_flat.numel() == 0:
        return torch.tensor(0., device=logits.device)
    prob = F.softmax(logits_flat, 1)
    gt   = F.one_hot(labels_flat, C).float()
    losses = []
    for c in range(1, C):
        fg = gt[:, c]
        if fg.sum() == 0:
            continue
        err = (prob[:, c] - fg).abs()
        err_sorted, perm = torch.sort(err, descending=True)
        fg_sorted = fg[perm]
        gtsum = fg.sum()
        inter = gtsum - fg_sorted.cumsum(0)
        union = gtsum + (1 - fg_sorted).cumsum(0)
        jacc = 1. - inter / union
        grad = jacc.clone()
        if jacc.numel() > 1:
            grad[1:] = jacc[1:] - jacc[:-1]
        losses.append((err_sorted * grad).sum())
    return sum(losses) / len(losses) if losses else torch.tensor(0., device=logits.device)

def boundary_loss(logits, mask, num_classes, ignore_index=0):
    """
    Vectorised Sobel edge-loss (no slow Python loop & correct conv2d signature).
    """
    with torch.no_grad():
        # (B,C,H,W) without ignore class
        gt = F.one_hot(mask.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()[:, 1:]
        B, C, H, W = gt.shape
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], device=gt.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        ky = kx.transpose(2, 3)
        # depth-wise conv (groups=C)
        ex = F.conv2d(gt, kx, bias=None, padding=1, groups=C)
        ey = F.conv2d(gt, ky, bias=None, padding=1, groups=C)
        edge = torch.sqrt(ex ** 2 + ey ** 2).clamp(0, 1)      # (B,C,H,W)
        boundary_map = edge.max(1, keepdim=True)[0]            # (B,1,H,W)
    probs = F.softmax(logits, 1)
    one_hot = F.one_hot(mask.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()
    valid = (mask != ignore_index).unsqueeze(1).float()
    diff  = (probs - one_hot).abs() * valid
    return (diff * boundary_map).sum() / (boundary_map.sum() + 1e-6)

def combined_loss(logits, mask, num_classes=20,
                  w_ce=0.25, w_dice=0.2, w_lovasz=0.25,
                  w_boundary=0.25, w_iou=0.25):
    ce   = F.cross_entropy(logits, mask, ignore_index=0)
    dice = dice_loss(logits, mask, num_classes)
    lov  = lovasz_softmax(logits, mask)
    bnd  = boundary_loss(logits, mask, num_classes)
    iou  = soft_iou_loss(logits, mask, num_classes)
    return w_ce * ce + w_dice * dice + w_lovasz * lov + w_boundary * bnd + w_iou * iou

# ======================  UTILS  =====================================

def seed_torch(seed=1904):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = '127.0.0.1', '29500'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp(): dist.destroy_process_group()

# ======================  DATA  ======================================

def get_dataloaders(rank, world_size, batch):
    ARCH = load_yaml("config/arch/LENet.yaml")
    DATA = load_yaml("config/labels/semantic-kitti.yaml")

    train_set = SemanticKitti("dataset", DATA["split"]["train"],
                              DATA["labels"], DATA["color_map"],
                              DATA["learning_map"], DATA["learning_map_inv"],
                              sensor=ARCH["dataset"]["sensor"],
                              max_points=ARCH["dataset"]["max_points"],
                              transform=True, gt=True)

    valid_set = SemanticKitti("dataset", DATA["split"]["valid"],
                              DATA["labels"], DATA["color_map"],
                              DATA["learning_map"], DATA["learning_map_inv"],
                              sensor=ARCH["dataset"]["sensor"],
                              max_points=ARCH["dataset"]["max_points"],
                              gt=True)

    train_samp = DistributedSampler(train_set, world_size, rank)
    valid_samp = DistributedSampler(valid_set, world_size, rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch,
                                               sampler=train_samp,
                                               num_workers=8,
                                               pin_memory=True,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch,
                                               sampler=valid_samp,
                                               num_workers=8,
                                               pin_memory=True,
                                               drop_last=True)
    return train_loader, valid_loader, train_samp

# ======================  CKPT & GRAD FIXES  =========================

def enable_checkpointing(model):
    def wrap(block):
        orig = block.forward
        def fwd_cp(*args, **kwargs):
            return checkpoint(orig, *args, **kwargs, use_reentrant=False)  # ← FIX
        block.forward = fwd_cp
    for blk in model.encoder.backbone.blocks:
        wrap(blk)

def make_grads_contiguous(model):
    for p in model.parameters():
        if p.grad is None:
            continue
        # Catch both the non-contiguous and the “contiguous but stride mismatch” cases
        if p.grad.stride() != p.stride():
            p.grad = p.grad.contiguous()

# ======================  TRAIN  =====================================

def train(rank, args):
    setup_ddp(rank, args.world_size)
    seed_torch(1904 + rank)

    train_loader, val_loader, train_samp = get_dataloaders(
        rank, args.world_size, args.batch_size)

    device = torch.device(f"cuda:{rank}")

    model = SAM2UNet("sam2.1_hiera_t.yaml",
                     args.hiera_path, args.stem,
                     args.freeze_weight, args.adapter, args.msca).to(device)

    enable_checkpointing(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank],
        find_unused_parameters=True,
        gradient_as_bucket_view=True)

    optimiser = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scaler = GradScaler()

    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); train_samp.set_epoch(epoch); running = 0.0
        for step, (in_vol, _, labels, *_) in enumerate(train_loader, 1):
            x   = in_vol.to(device, non_blocking=True)
            tgt = labels.to(device, non_blocking=True).long()

            optimiser.zero_grad(set_to_none=True)
            with autocast():
                main, aux = model(x)
                loss = combined_loss(main, tgt) + \
                       1 * sum(F.cross_entropy(a, tgt, ignore_index=0) for a in aux)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser); make_grads_contiguous(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser); scaler.update()

            running += loss.item()
            if rank == 0 and step % 50 == 0:
                print(f"E{epoch} [{step}/{len(train_loader)}]  Loss {loss.item():.4f}")

        if rank == 0:
            print(f"Epoch {epoch}  mean loss {running/len(train_loader):.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            if rank == 0: print("⇒  validating…")
            metrics = evaluate_metrics(model.module, val_loader, device=device)
            if rank == 0:
                print(f"★ mIoU {metrics['miou']:.4f}  pixAcc {metrics['pixel_acc']:.4f}")
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_path, f"SAM2-UNet-e{epoch}.pth"))
    cleanup_ddp()

# ======================  METRICS  ===================================

def evaluate_metrics(model, loader, num_classes=20, ignore=0, device='cuda'):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for in_vol, _, lbl, *_ in loader:
            x = in_vol.to(device, non_blocking=True)
            y = lbl.to(device, non_blocking=True)
            out, _ = model(x); pred = out.argmax(1)
            valid = y != ignore
            idx = num_classes * y[valid].view(-1) + pred[valid].view(-1)
            cm += torch.bincount(idx, minlength=num_classes**2).view(num_classes, num_classes)
    total, correct = cm.sum().item(), cm.trace().item()
    pixel_acc = correct / total if total else 0.
    ious, freqs = [], []
    for c in range(num_classes):
        if c == ignore: continue
        tp = cm[c, c].item(); fp = cm[:, c].sum().item() - tp; fn = cm[c].sum().item() - tp
        d = tp + fp + fn
        if d: ious.append(tp / d); freqs.append(cm[c].sum().item() / total)
    miou  = sum(ious) / len(ious) if ious else 0.
    fw_iou = sum(f * i for f, i in zip(freqs, ious)) if ious else 0.
    return {"pixel_acc": pixel_acc, "miou": miou, "fw_iou": fw_iou}

# ======================  CLI  =======================================

if __name__ == "__main__":
    p = argparse.ArgumentParser("SAM2-UNet DDP (AMP+checkpoint, patched)")
    p.add_argument("--hiera_path", required=True)
    p.add_argument("--save_path",  required=True)
    p.add_argument("--stem", required=True)
    p.add_argument("--freeze_weight", required=True, type=str2bool)
    p.add_argument("--adapter", required=True, type=str2bool)
    p.add_argument("--msca", required=True, type=str2bool)
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--batch_size", default=12, type=int)
    p.add_argument("--weight_decay", default=5e-4, type=float)
    p.add_argument("--world_size", default=torch.cuda.device_count(), type=int)
    args = p.parse_args()

    if args.world_size > 1:
        mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
    else:
        train(0, args)
