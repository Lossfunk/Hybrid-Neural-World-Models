#!/usr/bin/env python3
"""C3 verification for Euler v2: step-doubling estimator vs true error.

For each horizon h:
  1. Sample N_pairs random (s, GT(s,h)) pairs.
  2. Compute pred = f(s, dt) and pred_chain = f(f(s, dt/2), dt/2).
  3. ê per cell = ‖pred − pred_chain‖ across 4 channels.
  4. e per cell = ‖pred − GT‖ across 4 channels.
  5. Pearson r and AUROC between ê and e (per-cell ranking).

Usage:
  python eval_c3_euler.py --ckpt path/best.pt --split test
"""
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
ROOT = HERE.parent.parent
NEURIPS = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))

from eval_utils_euler import (load_model, step_doubling_estimator,             # noqa: E402
                                  true_error, pearson_r)
from data_utils_2d import Euler2DDataset                  # noqa: E402

DEFAULT_HORIZONS = [2, 4, 8, 16, 32, 64]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_pairs_per_horizon", type=int, default=100)
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "euler2d"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--auroc_thresh", type=float, default=0.75,
                     help="quantile of true error per pair that defines high-error")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = [int(x) for x in args.horizons.split(",")]
    ds_path = Path(args.data_dir) / f"euler2d_v2_{args.split}.h5"
    print(f"[c3_euler] split={args.split}  device={device}", flush=True)
    model = load_model(args.ckpt, device=device)
    ds = Euler2DDataset(str(ds_path))
    print(f"[c3_euler] N={ds.N} T={ds.T} base_dt={ds.dt}", flush=True)

    rng = np.random.RandomState(0)
    per_h_results = {}

    for h in horizons:
        if h >= ds.T:
            print(f"[c3_euler] skip h={h} (T={ds.T} too short)", flush=True)
            continue
        dt_target = h * ds.dt
        scores_all = []
        labels_all = []
        e_hat_pooled = []
        e_true_pooled = []
        t_start = time.time()
        for _ in range(args.n_pairs_per_horizon):
            i = int(rng.randint(0, ds.N))
            t0 = int(rng.randint(0, ds.T - h))
            u0 = ds.frame(i, t0).to(device).unsqueeze(0)
            ut = ds.frame(i, t0 + h).to(device).unsqueeze(0)
            with torch.no_grad():
                e_hat_map, pred_full = step_doubling_estimator(model, u0, dt_target)
                e_true_map = true_error(pred_full, ut)
            ehat_np = e_hat_map[0].cpu().numpy().ravel()
            etrue_np = e_true_map[0].cpu().numpy().ravel()
            # Sample-level threshold (q75 within this pair) for AUROC
            thr = float(np.quantile(etrue_np, args.auroc_thresh))
            lbl = (etrue_np > thr).astype(int)
            if lbl.sum() == 0 or lbl.sum() == len(lbl):
                continue
            scores_all.append(ehat_np)
            labels_all.append(lbl)
            e_hat_pooled.append(ehat_np)
            e_true_pooled.append(etrue_np)
        if not scores_all:
            print(f"[c3_euler] h={h}: no usable pairs", flush=True)
            continue
        sc = np.concatenate(scores_all)
        lb = np.concatenate(labels_all)
        eh = np.concatenate(e_hat_pooled)
        et = np.concatenate(e_true_pooled)
        auroc = float(roc_auc_score(lb, sc))
        r = pearson_r(eh, et)
        per_h_results[h] = {
            "n_pairs": len(scores_all),
            "auroc_q75": auroc,
            "pearson_r": r,
            "pearson_r2": float(r * r) if not np.isnan(r) else float("nan"),
        }
        print(f"  h={h:3d}  dt={dt_target:.4f}  auroc={auroc:.4f}  "
              f"pearson_r={r:.4f}  ({time.time()-t_start:.1f}s, n={len(scores_all)})",
              flush=True)

    out_path = Path(args.out) if args.out else (
        ROOT / "evaluation" / "euler2d_eval" / "results" /
        f"euler_c3_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split,
        "horizons": horizons,
        "n_pairs_per_horizon": args.n_pairs_per_horizon,
        "auroc_threshold_quantile": args.auroc_thresh,
        "per_h_results": per_h_results,
    }, indent=2))
    print(f"\n[c3_euler] wrote {out_path}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
