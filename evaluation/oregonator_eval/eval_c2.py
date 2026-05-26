#!/usr/bin/env python3
"""C2 verification: power-law calibration in smooth regions.

For each horizon Δt in the trained set:
  1. Sample N pairs from val/test
  2. Mask to SMOOTH regions only (|∇u| < q10 — bottom 10% of gradient magnitude)
  3. Compute mean |error| in those cells
  4. Plot mean error vs Δt on log-log axes; fit power law e ~ Δt^p

A power-law exponent p > 1 indicates super-linear convergence (consistent
with Taylor remainder bound for a smooth approximator).

Usage:
  python eval_c2.py --ckpt path/to/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from eval_utils import load_model, load_pair    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--horizons", default="1,2,4,8,16,32,64")
    ap.add_argument("--n_trajs", type=int, default=50)
    ap.add_argument("--smooth_pct", type=float, default=0.10,
                     help="bottom percentile of |∇u| considered smooth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = [int(x) for x in args.horizons.split(",")]
    ds_path = ROOT / "data" / "oregonator" / f"oregonator_{args.split}.h5"
    model = load_model(args.ckpt, device=device)

    with h5py.File(ds_path, "r") as f:
        N, T, _, H, W = f["states"].shape
        dt_save = float(f.attrs["dt_save"])
    print(f"[c2] split={args.split} horizons={horizons} n_trajs={args.n_trajs}")

    rng = np.random.RandomState(0)
    per_h_mean_err = {}
    per_h_dt = {}
    for h in horizons:
        if h >= T: continue
        dt = h * dt_save
        per_h_dt[h] = dt
        smooth_errs = []
        for k in range(args.n_trajs):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - h))
            u0, ut, _ = load_pair(str(ds_path), i, t0, h)
            u0_t = torch.from_numpy(u0).to(device).unsqueeze(0)
            ut_t = torch.from_numpy(ut).to(device).unsqueeze(0)
            dt_t = torch.tensor([dt], device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = model(u0_t, dt_t)
            e_map = torch.sqrt(((pred - ut_t) ** 2).sum(dim=1))[0].cpu().numpy()
            u_t = ut[0]
            gy, gx = np.gradient(u_t)
            gm = np.sqrt(gx**2 + gy**2)
            smooth_mask = gm <= np.quantile(gm, args.smooth_pct)
            if smooth_mask.sum() < 100: continue
            smooth_errs.append(float(e_map[smooth_mask].mean()))
        per_h_mean_err[h] = float(np.mean(smooth_errs)) if smooth_errs else float("nan")
        print(f"  h={h:2d}  dt={dt:.3f}  mean_e_smooth={per_h_mean_err[h]:.6f}  "
              f"(n={len(smooth_errs)} trajs)")

    # Power-law fit log(e) = p * log(dt) + b, exclude h=1 (might saturate at base error floor)
    dts = np.array([per_h_dt[h] for h in horizons if h in per_h_dt])
    errs = np.array([per_h_mean_err[h] for h in horizons if h in per_h_dt])
    valid = (errs > 1e-12) & np.isfinite(errs)
    if valid.sum() >= 3:
        log_dt = np.log(dts[valid])
        log_e = np.log(errs[valid])
        p, b = np.polyfit(log_dt, log_e, 1)
        e_pred_at_dt = np.exp(b) * dts ** p
    else:
        p = float("nan"); b = float("nan"); e_pred_at_dt = errs.copy()

    print()
    print(f"[c2] Power-law fit: e ~ {np.exp(b):.4e} · Δt^{p:.3f}  (target p > 1)")

    # Save results
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"c2_{args.split}.json"
    out_path.write_text(json.dumps({
        "split": args.split,
        "horizons": horizons,
        "smooth_pct": args.smooth_pct,
        "per_horizon": {str(h): {"dt": per_h_dt.get(h), "mean_e_smooth": per_h_mean_err.get(h)}
                          for h in horizons if h in per_h_dt},
        "power_law_p": float(p),
        "power_law_log_intercept": float(b),
    }, indent=2))
    print(f"[c2] results: {out_path}")

    # Figure: log-log error vs dt with power-law fit
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.loglog(dts, errs, "ko", markersize=8, label="measured smooth-region error")
    if not np.isnan(p):
        ax.loglog(dts, e_pred_at_dt, "r--", lw=1.5,
                   label=f"fit: e ~ Δt^{p:.2f}")
    ax.set_xlabel("Δt"); ax.set_ylabel("mean |error| in smooth cells")
    ax.set_title(f"C2: smooth-region error scaling  ({args.split})  fit p={p:.2f}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"c2_{args.split}.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    print(f"[c2] figure: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
