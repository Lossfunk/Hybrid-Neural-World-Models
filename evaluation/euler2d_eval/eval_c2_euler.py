#!/usr/bin/env python3
"""C2 verification for Euler v2: power-law calibration in smooth regions.

For each horizon h in the trained set:
  1. Sample N pairs from given split
  2. Mask to SMOOTH cells only (|∇p| < q10)
  3. Compute mean error in those cells
  4. Save per-horizon array; downstream figure fits power law and plots ratio

Outputs results/euler_c2_{split}.json.

Usage:
  python eval_c2_euler.py --ckpt path/best.pt --split test
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
                                  pressure_from_cons)
from data_utils_2d import Euler2DDataset                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--horizons", default="1,2,4,8,16,32,64")
    ap.add_argument("--n_trajs", type=int, default=50)
    ap.add_argument("--smooth_pct", type=float, default=0.10)
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "euler2d"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = [int(x) for x in args.horizons.split(",")]
    ds_path = Path(args.data_dir) / f"euler2d_v2_{args.split}.h5"
    model = load_model(args.ckpt, device=device)
    ds = Euler2DDataset(str(ds_path))
    print(f"[c2_euler] split={args.split} horizons={horizons} n={args.n_trajs}",
          flush=True)
    print(f"[c2_euler] N={ds.N}, T={ds.T}, base_dt={ds.dt}", flush=True)

    rng = np.random.RandomState(0)
    per_h_mean_err = {}
    per_h_dt = {}
    for h in horizons:
        if h >= ds.T: continue
        dt_target = h * ds.dt
        per_h_dt[h] = dt_target
        smooth_errs = []
        for k in range(args.n_trajs):
            i = int(rng.randint(0, ds.N))
            t0 = int(rng.randint(0, ds.T - h))
            u0 = ds.frame(i, t0).to(device).unsqueeze(0)
            ut = ds.frame(i, t0 + h).to(device).unsqueeze(0)
            with torch.no_grad():
                pred = predict(model, u0, dt_target)
            e_map = true_error(pred, ut)[0].cpu().numpy()
            ut_np = ut[0].cpu().numpy()
            p = pressure_from_cons(ut_np)
            gy, gx = np.gradient(p)
            gm = np.sqrt(gx**2 + gy**2)
            thr = float(np.quantile(gm, args.smooth_pct))
            smooth_mask = gm <= thr
            if smooth_mask.sum() < 100:
                continue
            smooth_errs.append(float(e_map[smooth_mask].mean()))
        if smooth_errs:
            per_h_mean_err[h] = float(np.mean(smooth_errs))
            print(f"  h={h:3d}  dt={dt_target:.4f}  smooth_e={np.mean(smooth_errs):.5f}  "
                  f"+- {np.std(smooth_errs):.5f}  n={len(smooth_errs)}", flush=True)

    out_path = Path(args.out) if args.out else (
        ROOT / "evaluation" / "euler2d_eval" / "results" /
        f"euler_c2_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split, "horizons": horizons,
        "smooth_pct": args.smooth_pct,
        "per_h_dt": per_h_dt,
        "per_h_mean_err": per_h_mean_err,
    }, indent=2))
    print(f"\n[c2_euler] wrote {out_path}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
