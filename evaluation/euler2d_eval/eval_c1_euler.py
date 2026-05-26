#!/usr/bin/env python3
"""C1 verification for Euler v2: e-map lights up at shocks (high |∇p|).

For 50 random test trajectories at a given horizon:
  1. Compute e-map per cell: e(x, y) = ‖f(s, dt) − Φ(s, dt)‖ across 4 channels
  2. Compute shock mask per cell: |∇p| > q90 (top 10% pressure gradient)
  3. Aggregate: ratio of mean(e | shock) to mean(e | smooth)
  4. Spatial correlation between e-map and |∇p|

Outputs results/euler_c1_{split}_h{h}.json.

Usage:
  python eval_c1_euler.py --ckpt path/best.pt --split test --horizon 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
NEURIPS = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))

from eval_utils_euler import (load_model, predict, true_error,                 # noqa: E402
                                  pressure_from_cons, pearson_r)
from data_utils_2d import Euler2DDataset                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n_trajs", type=int, default=50)
    ap.add_argument("--front_pct", type=float, default=0.9)
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "euler2d"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_path = Path(args.data_dir) / f"euler2d_v2_{args.split}.h5"
    print(f"[c1_euler] split={args.split} h={args.horizon} n={args.n_trajs}",
          flush=True)
    model = load_model(args.ckpt, device=device)
    ds = Euler2DDataset(str(ds_path))
    print(f"[c1_euler] N={ds.N}, T={ds.T}, base_dt={ds.dt}", flush=True)

    h = args.horizon
    if h >= ds.T:
        print(f"[c1_euler] horizon {h} too large for T={ds.T}")
        return 1
    dt_target = h * ds.dt

    rng = np.random.RandomState(0)
    front_means = []
    smooth_means = []
    spatial_corrs = []
    n_skipped = 0
    for k in range(args.n_trajs):
        i = int(rng.randint(0, ds.N))
        t0 = int(rng.randint(0, ds.T - h))
        u0 = ds.frame(i, t0).to(device).unsqueeze(0)        # (1, 4, H, W)
        ut = ds.frame(i, t0 + h).to(device).unsqueeze(0)
        with torch.no_grad():
            pred = predict(model, u0, dt_target)
        e_map = true_error(pred, ut)[0].cpu().numpy()        # (H, W)
        # Shock mask on the TARGET state (where the model is predicting to)
        ut_np = ut[0].cpu().numpy()
        p = pressure_from_cons(ut_np)
        gy, gx = np.gradient(p)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        # Strict comparison + small floor to handle piecewise-constant edge cases
        thr = float(np.quantile(grad_mag, args.front_pct))
        thr = max(thr, 1e-6 * float(grad_mag.max() + 1e-12))
        f_mask = grad_mag > thr
        if f_mask.sum() < 10 or (~f_mask).sum() < 10:
            n_skipped += 1
            continue
        e_front = float(e_map[f_mask].mean())
        e_smooth = float(e_map[~f_mask].mean())
        front_means.append(e_front)
        smooth_means.append(e_smooth)
        spatial_corrs.append(pearson_r(e_map.ravel(), grad_mag.ravel()))

    front_means = np.array(front_means)
    smooth_means = np.array(smooth_means)
    spatial_corrs = np.array(spatial_corrs)
    if len(front_means) == 0:
        print("[c1_euler] no usable pairs (all skipped)")
        ds.close()
        return 1

    ratio = front_means.mean() / max(smooth_means.mean(), 1e-12)

    print()
    print(f"[c1_euler] Aggregate over {len(front_means)} trajectories at h={h}, dt={dt_target:.4f}:")
    print(f"  mean error at fronts:   {front_means.mean():.5f} +- {front_means.std():.5f}")
    print(f"  mean error at smooth:   {smooth_means.mean():.5f} +- {smooth_means.std():.5f}")
    print(f"  ratio front / smooth:   {ratio:.2f}x  (target > 2)")
    print(f"  spatial corr(e, |grad p|):  mean = {spatial_corrs.mean():.3f}  "
          f"std = {spatial_corrs.std():.3f}")
    print(f"  n_skipped (degenerate mask): {n_skipped}")

    out_path = Path(args.out) if args.out else (
        ROOT / "evaluation" / "euler2d_eval" / "results" /
        f"euler_c1_{args.split}_h{args.horizon}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split, "horizon": h, "dt": dt_target,
        "n_trajs": len(front_means), "n_skipped": n_skipped,
        "front_pct_threshold": args.front_pct,
        "mean_e_at_front": float(front_means.mean()),
        "mean_e_at_smooth": float(smooth_means.mean()),
        "ratio": float(ratio),
        "spatial_corr_mean": float(spatial_corrs.mean()),
        "spatial_corr_std": float(spatial_corrs.std()),
        "front_means_per_traj": front_means.tolist(),
        "smooth_means_per_traj": smooth_means.tolist(),
        "spatial_corrs_per_traj": spatial_corrs.tolist(),
    }, indent=2))
    print(f"\n[c1_euler] wrote {out_path}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
