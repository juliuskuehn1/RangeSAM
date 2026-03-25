#!/usr/bin/env python3
"""
Pre‑compute SemanticKITTI range‑image tuples and store them on disk
==================================================================

This script loads your existing `SemanticKitti` Dataset class, runs the 3‑D → 2‑D
projection once **offline**, and stores the resulting tensors to compressed
`.npz` archives—one file per LiDAR scan.  Each file contains exactly the tuple
that your `__getitem__` returns, so you can later plug a tiny wrapper dataset
that just does `numpy.load()` and skips the heavy projection.

Typical usage
-------------
$ python precompute_semantickitti.py \
      --data-root   /data/semantic_kitti \
      --sequences   00 01 02 03 04 05 06 07 09 10 \
      --out-root    /data/semk_cache_5ch \
      --n-workers   8 \
      --batch-size  4

The script is **restart‑safe**: if an `.npz` already exists it will be skipped.
That lets you distribute the work across several machines or resume after an
interruption.

Storage footprint
-----------------
• 5‑channel `float32` image (5·64·2048)       ≈  2.6 MB
• bool mask + labels + xyz + range + remissions ~ 4.2 MB
→ ~6.8 MB per scan × 43 552 scans  ≈ 300 GB (train+val)

Feel free to switch to `float16` for the image and range/proj_xyz arrays if
disk space is tight: add `--fp16-image`.
"""
import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from preprocess.parser import SemanticKitti


# -----------------------------------------------------------------------------
# import your code -------------------------------------------------------------
# -----------------------------------------------------------------------------
# from semantic_kitti_dataset import SemanticKitti   # <-- adjust import path

# -----------------------------------------------------------------------------
# command‑line arguments -------------------------------------------------------
# -----------------------------------------------------------------------------

def load_yaml(path):
    try:
        print(f"\033[32m Opening arch config file {path}\033[0m")
        yaml_data = yaml.safe_load(open(path, 'r'))
        return yaml_data
    except Exception as e:
        print(e)
        print(f"Error opening {path} yaml file.")
        quit()

def _parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Pre‑compute SemanticKITTI range‑image tensors → .npz")

    p.add_argument("--data-root", required=True, type=Path,
                   help="Path to SemanticKITTI 'sequences' folder")
    p.add_argument("--sequences", nargs="*", default=["00"],
                   help="Sequence IDs to process (e.g. 00 01 02 …)")
    p.add_argument("--out-root", required=True, type=Path,
                   help="Folder where .npz files will be written")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-workers", type=int, default=mp.cpu_count())
    p.add_argument("--fp16-image", action="store_true",
                   help="Store proj_full as float16 to halve disk usage")
    return p.parse_args()

# -----------------------------------------------------------------------------
# main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    args = _parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # build dataset exactly as in your training config -----------------
    # ------------------------------------------------------------------
    # from my_config import (labels, color_map, learning_map, learning_map_inv,
    #                        sensor)                        # adjust import path
    ARCH = load_yaml("preprocess/LENet.yaml")
    DATA = load_yaml("preprocess/semantic-kitti.yaml")
    dataset = SemanticKitti(
        root=str(args.data_root.parent),
        sequences=[int(s) for s in args.sequences],
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        gt=True,
        transform=False,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False)

    # ------------------------------------------------------------------
    # iterate and dump --------------------------------------------------
    # ------------------------------------------------------------------
    for batch in loader:
        (proj_full, proj_mask, proj_labels, unproj_labels,
         path_seq, path_name, proj_x, proj_y,
         proj_range, unproj_range, proj_xyz, unproj_xyz,
         proj_remission, unproj_remissions, unproj_n_points) = batch

        # batch dimension is B; iterate items because shapes vary per scan
        B = proj_full.size(0)
        for b in range(B):
            seq = path_seq[b]
            fname = Path(path_name[b]).stem  # e.g. 000034
            out_dir = args.out_root / f"seq_{seq}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{fname}.npz"
            if out_file.exists():
                continue  # already processed

            save_dict = {
                "proj_full": proj_full[b].numpy().astype(np.float16 if args.fp16_image else np.float32),
                "proj_mask": proj_mask[b].numpy(),
                "proj_labels": proj_labels[b].numpy(),
                "unproj_labels": unproj_labels[b].numpy(),
                "proj_x": proj_x[b].numpy(),
                "proj_y": proj_y[b].numpy(),
                "proj_range": proj_range[b].numpy(),
                "unproj_range": unproj_range[b].numpy(),
                "proj_xyz": proj_xyz[b].numpy(),
                "unproj_xyz": unproj_xyz[b].numpy(),
                "proj_remission": proj_remission[b].numpy(),
                "unproj_remissions": unproj_remissions[b].numpy(),
                "unproj_n_points": unproj_n_points[b].item(),
            }
            np.savez_compressed(out_file, **save_dict)
            print(f"✔ saved {out_file.relative_to(args.out_root)}")

if __name__ == "__main__":
    main()
