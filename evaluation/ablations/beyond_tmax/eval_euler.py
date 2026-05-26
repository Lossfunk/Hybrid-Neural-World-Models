#!/usr/bin/env python3
"""Beyond-T_max audit for Euler. T=100 limits us to h≤99; we test h ∈ {64, 80, 96}."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))
sys.path.insert(0, str(ROOT / "models"))

from eval_utils_euler import (load_model, predict, step_doubling_estimator,
                                 true_error)
from data_utils_2d import Euler2DDataset

CKPT = (ROOT / "checkpoints" / "euler2d" / "best.pt")
DATA_DIR = ROOT / "data" / "euler2d"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HORIZONS = [64, 80, 96]
SPLITS = ["test", "ood_near", "ood_far"]


def auroc_per_traj(scores, true_errs):
    if len(scores) < 4: return float("nan")
    thr = float(np.quantile(true_errs, 0.75))
    lbl = (true_errs > thr).astype(int)
    if lbl.sum() == 0 or lbl.sum() == len(lbl): return float("nan")
    return float(roc_auc_score(lbl, scores))


@torch.no_grad()
def eval_split(model, ds: Euler2DDataset, n_per_h: int, seed: int):
    rng = np.random.RandomState(seed)
    out = {}
    for h in HORIZONS:
        if h >= ds.T:
            out[h] = {"auroc": float("nan"), "n_pairs": 0,
                       "skipped": "h >= T"}; continue
        t0_ = time.time()
        scores, true_errs = [], []
        dt_target = h * ds.dt
        for _ in range(n_per_h):
            i = int(rng.randint(0, ds.N))
            t0 = int(rng.randint(0, ds.T - h))
            u0 = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
            ut = ds.frame(i, t0 + h).to(DEVICE).unsqueeze(0)
            e_map, pred = step_doubling_estimator(model, u0, dt_target)
            te_map = true_error(pred, ut)
            scores.append(float(e_map.mean().cpu().numpy()))
            true_errs.append(float(te_map.mean().cpu().numpy()))
        sc = np.array(scores); te = np.array(true_errs)
        out[h] = {
            "auroc": auroc_per_traj(sc, te),
            "mean_true_rmse": float(np.mean(te)),
            "mean_ehat": float(np.mean(sc)),
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

    print(f"[beyond_tmax_euler] horizons={HORIZONS}", flush=True)
    model = load_model(str(CKPT), device=DEVICE)

    all_results = {}
    for split in SPLITS:
        sp = DATA_DIR / f"euler2d_v2_{split}.h5"
        ds = Euler2DDataset(str(sp))
        print(f"\n=== {split} (N={ds.N} T={ds.T}) ===", flush=True)
        all_results[split] = eval_split(model, ds, args.n_per_h, seed=42)

    out_path = RESULTS / "euler_beyond_tmax.json"
    out_path.write_text(json.dumps({
        "config": {"ckpt": str(CKPT), "horizons": HORIZONS,
                    "n_per_h": args.n_per_h, "auroc_threshold": "q75",
                    "trained_tmax": 64},
        "results": all_results,
    }, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
