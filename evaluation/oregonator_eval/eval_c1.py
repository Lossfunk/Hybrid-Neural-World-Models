#!/usr/bin/env python3
"""C1 verification: e-map lights up at fronts.

For 50 random test trajectories:
  1. Compute e-map per cell: e(x, y) = ‖f(s, Δt) − Φ(s, Δt)‖ across channels
  2. Compute front mask per cell: |∇u| > p90 percentile
  3. Aggregate: ratio of mean(e | front) to mean(e | smooth)
  4. Spatial correlation between e-map and |∇u|

A high ratio (>>1) and high correlation confirm C1: the e-map structurally
flags discontinuities. Visualize e-map + front overlay for sample trajs.

Usage:
  python eval_c1.py --ckpt path/to/best.pt --split test --horizon 16
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

from eval_utils import load_model, load_pair, front_mask, pearson_r   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n_trajs", type=int, default=50)
    ap.add_argument("--front_pct", type=float, default=0.9,
                     help="quantile threshold on |∇u| for front mask")
    ap.add_argument("--n_visualize", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_path = ROOT / "data" / "oregonator" / f"oregonator_{args.split}.h5"
    print(f"[c1] split={args.split} horizon={args.horizon} n_trajs={args.n_trajs}")
    model = load_model(args.ckpt, device=device)

    with h5py.File(ds_path, "r") as f:
        N, T, _, H, W = f["states"].shape
        dt_save = float(f.attrs["dt_save"])

    rng = np.random.RandomState(0)
    h = args.horizon
    if h >= T:
        print(f"[c1] horizon {h} too large for T={T}")
        return 1
    dt = h * dt_save

    front_means = []      # mean error in front cells
    smooth_means = []     # mean error in smooth cells
    spatial_corrs = []    # Pearson(e, |∇u|) over the spatial grid
    sample_figures = []
    for k in range(args.n_trajs):
        i = int(rng.randint(0, N))
        t0 = int(rng.randint(0, T - h))
        u0, ut, _ = load_pair(str(ds_path), i, t0, h)   # (C, H, W) each
        u0_t = torch.from_numpy(u0).to(device).unsqueeze(0)
        ut_t = torch.from_numpy(ut).to(device).unsqueeze(0)
        dt_t = torch.tensor([dt], device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = model(u0_t, dt_t)
        # per-cell error map
        e_map = torch.sqrt(((pred - ut_t) ** 2).sum(dim=1))[0].cpu().numpy()  # (H, W)
        # front detection on the TARGET state (front at the time the model is predicting to)
        u_target = ut[0]   # (H, W) — u channel
        gy, gx = np.gradient(u_target)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        f_mask = grad_mag >= np.quantile(grad_mag, args.front_pct)
        if f_mask.sum() < 10 or (~f_mask).sum() < 10:
            continue
        e_front = float(e_map[f_mask].mean())
        e_smooth = float(e_map[~f_mask].mean())
        front_means.append(e_front)
        smooth_means.append(e_smooth)
        spatial_corrs.append(pearson_r(e_map.ravel(), grad_mag.ravel()))
        # Save first n_visualize for figure
        if len(sample_figures) < args.n_visualize:
            sample_figures.append(dict(
                e_map=e_map, grad_mag=grad_mag, f_mask=f_mask,
                u0=u0, ut=ut, pred=pred[0].cpu().numpy(),
                i=i, t0=t0,
            ))

    front_means = np.array(front_means)
    smooth_means = np.array(smooth_means)
    spatial_corrs = np.array(spatial_corrs)
    ratio = front_means.mean() / max(smooth_means.mean(), 1e-12)

    print()
    print(f"[c1] Aggregate over {len(front_means)} trajectories at h={h}, dt={dt:.3f}:")
    print(f"  mean error at fronts:   {front_means.mean():.5f} ± {front_means.std():.5f}")
    print(f"  mean error at smooth:   {smooth_means.mean():.5f} ± {smooth_means.std():.5f}")
    print(f"  ratio front / smooth:   {ratio:.2f}× (target > 2)")
    print(f"  spatial corr(e, |∇u|):  mean = {spatial_corrs.mean():.3f}  "
          f"std = {spatial_corrs.std():.3f}")

    # Save results
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"c1_{args.split}_h{args.horizon}.json"
    out_path.write_text(json.dumps({
        "split": args.split, "horizon": args.horizon, "dt": dt,
        "n_trajs": len(front_means),
        "front_pct_threshold": args.front_pct,
        "mean_e_at_front": float(front_means.mean()),
        "mean_e_at_smooth": float(smooth_means.mean()),
        "ratio_front_over_smooth": float(ratio),
        "spatial_corr_mean": float(spatial_corrs.mean()),
        "spatial_corr_std": float(spatial_corrs.std()),
        "front_means_per_traj": front_means.tolist(),
        "smooth_means_per_traj": smooth_means.tolist(),
        "spatial_corrs_per_traj": spatial_corrs.tolist(),
    }, indent=2))
    print(f"[c1] results: {out_path}")

    # Figure: 4 sample trajs × 4 panels (u_target, e_map, |∇u|, e_map overlaid with front mask)
    n_fig = len(sample_figures)
    fig, axes = plt.subplots(n_fig, 4, figsize=(4 * 4, 4 * n_fig))
    if n_fig == 1:
        axes = axes[None, :]
    for r, sample in enumerate(sample_figures):
        e_map = sample["e_map"]; gm = sample["grad_mag"]; m = sample["f_mask"]
        ut = sample["ut"][0]
        axes[r, 0].imshow(ut, origin="lower", cmap="inferno", vmin=0, vmax=1)
        axes[r, 0].set_title(f"u target  (traj {sample['i']}, t0={sample['t0']})")
        axes[r, 1].imshow(e_map, origin="lower", cmap="viridis")
        axes[r, 1].set_title("e-map (true error)")
        axes[r, 2].imshow(gm, origin="lower", cmap="magma")
        axes[r, 2].set_title("|∇u| (front detector)")
        # overlay front mask (red contour) on e-map
        axes[r, 3].imshow(e_map, origin="lower", cmap="viridis")
        axes[r, 3].contour(m.astype(float), levels=[0.5], colors="red", linewidths=0.7)
        axes[r, 3].set_title("e-map + front contour")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"C1: e-map at fronts  ({args.split}, h={h}, dt={dt:.2f})  "
                 f"front/smooth ratio={ratio:.2f}×", fontsize=12)
    fig.tight_layout()
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"c1_{args.split}.png"
    fig.savefig(fig_path, dpi=110)
    print(f"[c1] figure: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
