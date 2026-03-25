#!/usr/bin/env python3
"""
verify_semk_cache.py
====================
Walk through a cached SemanticKITTI directory (created by
precompute_semantickitti.py) and verify that every `.npz` file contains exactly
*the tuple your training loop expects*.

Usage
-----
$ python verify_semk_cache.py  /path/to/semk_cache02/seq_02

Exit status is **0** if all files pass; otherwise the script prints a report of
violations and exits 1.
"""
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# expected template ------------------------------------------------------------
# -----------------------------------------------------------------------------
EXPECTED_KEYS = [
    "proj_full",
    "proj_mask",
    "proj_labels",
    "unproj_labels",
    "proj_x",
    "proj_y",
    "proj_range",
    "unproj_range",
    "proj_xyz",
    "unproj_xyz",
    "proj_remission",
    "unproj_remissions",
    "unproj_n_points",
]

# Shapes that are *fixed*, independent of the scan
FIXED_SHAPES: Dict[str, Tuple[int, ...]] = {
    "proj_mask": (64, 2048),
    "proj_labels": (64, 2048),
    "proj_range": (64, 2048),
    "proj_xyz": (64, 2048, 3),
    # proj_full depends on channel count – we'll validate H,W only.
}

# -----------------------------------------------------------------------------
# validation helpers -----------------------------------------------------------
# -----------------------------------------------------------------------------

def validate_npz(path: Path) -> Tuple[bool, str]:
    """Return (is_valid, message)."""
    try:
        data = np.load(path)
    except Exception as e:
        return False, f"Cannot open: {e}"

    # --- keys present? -------------------------------------------------------
    missing = [k for k in EXPECTED_KEYS if k not in data]
    extra   = [k for k in data.files if k not in EXPECTED_KEYS]
    if missing or extra:
        return False, f"keys mismatch – missing {missing}, unexpected {extra}"

    # --- fixed shapes --------------------------------------------------------
    for k, shp in FIXED_SHAPES.items():
        if data[k].shape != shp:
            return False, f"{k} shape {data[k].shape} ≠ expected {shp}"

    # --- proj_full check -----------------------------------------------------
    pf = data["proj_full"]
    if pf.ndim != 3 or pf.shape[1:] != (64, 2048):
        return False, f"proj_full shape {pf.shape} – H×W should be 64×2048"

    # channel sanity – 5, 8, 13, 6 … depending on config
    if pf.shape[0] not in {5, 6, 8, 13}:
        return False, f"proj_full has {pf.shape[0]} channels – unexpected"

    # --- unproj length consistency ------------------------------------------
    n = int(data["unproj_n_points"])
    for k in ("unproj_xyz", "unproj_range", "unproj_remissions", "unproj_labels"):
        if data[k].shape[0] < n:
            return False, f"{k} length {data[k].shape[0]} ≠ unproj_n_points {n}"

    return True, "ok"

# -----------------------------------------------------------------------------
# main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_semk_cache.py /path/to/seq_02", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    npz_files = sorted(root.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files under {root}")
        sys.exit(1)

    bad = []
    for f in npz_files:
        ok, msg = validate_npz(f)
        if not ok:
            bad.append((f, msg))
            print(f"✗ {f.relative_to(root)} – {msg}")

    if bad:
        print("\nSummary: FAILED –", len(bad), "problematic files")
        sys.exit(1)
    else:
        print("All", len(npz_files), "files passed ✔")
        sys.exit(0)

if __name__ == "__main__":
    main()
