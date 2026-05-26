#!/usr/bin/env python3
"""Full horizon sweep for Mode 1 / Mode 2 RMSE across all 3 environments
and all splits. Used to build the appendix RMSE table in the paper.

Re-uses helpers from eval_qsweep.py and eval_qsweep_ball3d.py but with
HORIZONS = [2, 4, 8, 16, 32, 64].

Output: ablations/mode2_qsweep/results/full_horizons/{env}_{split}.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HORIZONS_FULL = [2, 4, 8, 16, 32, 64]

# Reuse helpers
sys.path.insert(0, str(HERE))
qsweep_mod = importlib.import_module("eval_qsweep")
qsweep_mod.HORIZONS = HORIZONS_FULL    # monkey-patch

ball3d_mod = importlib.import_module("eval_qsweep_ball3d")
ball3d_mod.HORIZONS = HORIZONS_FULL    # monkey-patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="oregonator,euler,ball3d")
    ap.add_argument("--splits", default="test,ood_near,ood_far")
    args = ap.parse_args()

    out_root = HERE / "results" / "full_horizons"
    out_root.mkdir(parents=True, exist_ok=True)

    for env in args.envs.split(","):
        for split in args.splits.split(","):
            print(f"\n=== {env} {split}  (horizons {HORIZONS_FULL}) ===",
                    flush=True)
            t0 = time.time()
            if env == "oregonator":
                qsweep_mod.run_oregonator(split)
                src = HERE / "results" / f"oregonator_{split}.json"
            elif env == "euler":
                qsweep_mod.run_euler(split)
                src = HERE / "results" / f"euler_{split}.json"
            elif env == "ball3d":
                ball3d_mod.run(split)
                src = HERE / "results" / f"ball3d_{split}.json"
            else:
                continue
            # Move to full_horizons subdirectory
            dst = out_root / src.name
            if src.exists():
                dst.write_text(src.read_text())
                print(f"  → {dst}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
