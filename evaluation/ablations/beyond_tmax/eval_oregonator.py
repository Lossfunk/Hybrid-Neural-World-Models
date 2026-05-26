#!/usr/bin/env python3
"""Beyond-T_max audit: trust signal AUROC at horizons exceeding the trained
ladder T_max=64. Tests whether step-doubling continues to discriminate
when the surrogate is queried outside its training horizon support.

Outputs per-horizon AUROC + RMSE for h ∈ {64, 80, 96, 128, 160} on each split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "models"))

from eval_utils import (load_model, predict, step_doubling_estimator,
                          true_error)

CKPT = ROOT / "checkpoints" / "oregonator" / "best.pt"
DATA_DIR = ROOT / "data" / "oregonator"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT_BASE = 0.05

# h=64 is the trained T_max; 80/96/128/160 are extrapolation
HORIZONS = [64, 80, 96, 128, 160]
SPLITS = ["test", "ood_near", "ood_far"]


def auroc_per_traj(scores: np.ndarray, true_errs: np.ndarray) -> float:
    if len(scores) < 4: return float("nan")
    thr = float(np.quantile(true_errs, 0.75))
    bin_lbl = (true_errs > thr).astype(int)
    if bin_lbl.sum() == 0 or bin_lbl.sum() == len(bin_lbl):
        return float("nan")
    return float(roc_auc_score(bin_lbl, scores))


@torch.no_grad()
def eval_split(model, ds_path: Path, n_per_h: int, seed: int):
    rng = np.random.RandomState(seed)
    out = {}
    with h5py.File(ds_path, "r") as f:
        N, T = f["states"].shape[:2]
        for h in HORIZONS:
            if h >= T:
                out[h] = {"auroc": float("nan"),
                          "mean_true_rmse": float("nan"),
                          "mean_ehat": float("nan"),
                          "n_pairs": 0,
                          "skipped": "h >= T"}
                continue
            t0_ = time.time()
            scores, true_errs, ehats = [], [], []
            for _ in range(n_per_h):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s = torch.from_numpy(np.array(f["states"][i, t0])
                                       ).unsqueeze(0).to(DEVICE)
                tgt = torch.from_numpy(np.array(f["states"][i, t0 + h])
                                         ).unsqueeze(0).to(DEVICE)
                dt = torch.tensor([h * DT_BASE], dtype=torch.float32,
                                   device=DEVICE)
                e_map, pred = step_doubling_estimator(model, s, dt)
                te_map = true_error(pred, tgt)
                scores.append(float(e_map.mean().cpu().numpy()))
                ehats.append(float(e_map.mean().cpu().numpy()))
                true_errs.append(float(te_map.mean().cpu().numpy()))
            sc = np.array(scores); te = np.array(true_errs)
            out[h] = {
                "auroc": auroc_per_traj(sc, te),
                "mean_true_rmse": float(np.mean(te)),
                "mean_ehat": float(np.mean(ehats)),
                "n_pairs": len(sc),
                "elapsed_s": time.time() - t0_,
            }
            print(f"  h={h:3d}  AUROC={out[h]['auroc']:.4f}  "
                  f"true_rmse={out[h]['mean_true_rmse']:.4f}  "
                  f"ehat={out[h]['mean_ehat']:.4f}  "
                  f"({out[h]['elapsed_s']:.1f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_h", type=int, default=80)
    args = ap.parse_args()

    print(f"[beyond_tmax_oreg] device={DEVICE}", flush=True)
    print(f"[beyond_tmax_oreg] horizons={HORIZONS}  n_per_h={args.n_per_h}",
          flush=True)
    model = load_model(str(CKPT), device=DEVICE)
    print(f"[beyond_tmax_oreg] loaded {CKPT.name}", flush=True)

    all_results = {}
    for split in SPLITS:
        sp = DATA_DIR / f"oregonator_{split}.h5"
        print(f"\n=== {split} ===", flush=True)
        all_results[split] = eval_split(model, sp, args.n_per_h, seed=42)

    out_path = RESULTS / "oregonator_beyond_tmax.json"
    out_path.write_text(json.dumps({
        "config": {"ckpt": str(CKPT), "horizons": HORIZONS,
                    "n_per_h": args.n_per_h, "auroc_threshold": "q75",
                    "trained_tmax": 64},
        "results": all_results,
    }, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
