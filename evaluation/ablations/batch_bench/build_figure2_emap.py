#!/usr/bin/env python3
"""Build Figure 2: cross-env e-map visualization showing the trust signal
spatially aligns with discontinuities (Oregonator, Euler) and contact
events (Ball 3D).

Layout: 3 rows (envs) × 3 cols (input, e-hat, true error).
For Ball 3D we use a 2D xy-projection of the two balls' trajectories,
colored by e-hat / true error along the prediction window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
OUT_DIR = ROOT / "NeurIPS_final_paper_figures"
OUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def panel_oreg(ax_input, ax_ehat, ax_err, traj_idx: int = 17, t0: int = 40, h: int = 64):
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils import load_model, predict, step_doubling_estimator, true_error

    ckpt = ROOT / "checkpoints" / "oregonator" / "best.pt"
    data = ROOT / "data" / "oregonator" / "oregonator_test.h5"
    model = load_model(str(ckpt), device=DEVICE)
    DT_BASE = 0.05
    target_dt = h * DT_BASE

    with h5py.File(data, "r") as f:
        s = torch.from_numpy(np.array(f["states"][traj_idx, t0])).unsqueeze(0).to(DEVICE)
        tgt = torch.from_numpy(np.array(f["states"][traj_idx, t0 + h])).unsqueeze(0).to(DEVICE)
    dt = torch.tensor([target_dt], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        e_map, pred = step_doubling_estimator(model, s, dt)
        te_map = true_error(pred, tgt)

    inp = s[0, 0].cpu().numpy()
    eh = e_map[0].cpu().numpy()
    te = te_map[0].cpu().numpy()
    ax_input.imshow(inp, cmap="viridis"); ax_input.set_title("Input $u$ (channel 0)", fontsize=10)
    ax_ehat.imshow(eh, cmap="hot"); ax_ehat.set_title("$\\hat{e}$ (step-doubling)", fontsize=10)
    ax_err.imshow(te, cmap="hot"); ax_err.set_title("True RMSE per cell", fontsize=10)
    for ax in [ax_input, ax_ehat, ax_err]:
        ax.set_xticks([]); ax.set_yticks([])


def panel_euler(ax_input, ax_ehat, ax_err, traj_idx: int = 7, t0: int = 4, h: int = 64):
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils_euler import (load_model, predict, step_doubling_estimator,
                                       true_error)
    from data_utils_2d import Euler2DDataset

    ckpt = (ROOT / "checkpoints" / "euler2d" / "best.pt")
    data = ROOT / "data" / "euler2d" / "euler2d_v2_test.h5"
    model = load_model(str(ckpt), device=DEVICE)
    ds = Euler2DDataset(str(data))
    BASE_DT = ds.dt
    target_dt = h * BASE_DT

    s = ds.frame(traj_idx, t0).to(DEVICE).unsqueeze(0)
    tgt = ds.frame(traj_idx, t0 + h).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        e_map, pred = step_doubling_estimator(model, s, target_dt)
        te_map = true_error(pred, tgt)

    rho = s[0, 0].cpu().numpy()  # density
    eh = e_map[0].cpu().numpy()
    te = te_map[0].cpu().numpy()
    ax_input.imshow(rho, cmap="viridis"); ax_input.set_title("Input $\\rho$ (density)", fontsize=10)
    ax_ehat.imshow(eh, cmap="hot"); ax_ehat.set_title("$\\hat{e}$ (step-doubling)", fontsize=10)
    ax_err.imshow(te, cmap="hot"); ax_err.set_title("True RMSE per cell", fontsize=10)
    for ax in [ax_input, ax_ehat, ax_err]:
        ax.set_xticks([]); ax.set_yticks([])


def panel_ball3d(ax_input, ax_ehat, ax_err, traj_idx: int = 18, h: int = 64):
    """Ball 3D: state is 12-D. Visualize as: column-1 trajectory projection,
    column-2 e-hat per prediction step (bar / line), column-3 true RMSE per step."""
    sys.path.insert(0, str(ROOT / "training" / "ball3d"))
    sys.path.insert(0, str(ROOT / "data_generation" / "ball3d"))
    from shortcut_ball3d import ShortcutBall3D

    ckpt = ROOT / "checkpoints" / "ball3d" / "best.pt"
    data = ROOT / "data" / "ball3d" / "ball3d_test.h5"

    ck = torch.load(str(ckpt), map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    model = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                              emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                              ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    DT_BASE = 0.01
    with h5py.File(data, "r") as f:
        traj = np.array(f["states"][traj_idx], dtype=np.float32)  # (T, 9)
        T = traj.shape[0]

    # For each candidate t0, run step-doubling at h=64 and collect e-hat + true err
    horizons = list(range(1, min(64, T - 1) + 1, 4))
    e_hats, true_errs, dts = [], [], []
    s = torch.from_numpy(traj[0]).unsqueeze(0).to(DEVICE)
    for h_step in horizons:
        if h_step >= T: continue
        target_dt = h_step * DT_BASE
        dt_t = torch.full((1,), target_dt, dtype=torch.float32, device=DEVICE)
        half_t = dt_t * 0.5
        with torch.no_grad():
            pf = model(s, dt_t)
            pm = model(s, half_t)
            pc = model(pm, half_t)
            e = float(torch.sqrt(((pf - pc) ** 2).sum() + 1e-12).item())
        gt = torch.from_numpy(traj[h_step]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            te = float(torch.sqrt(((pf - gt) ** 2).mean()).item())
        e_hats.append(e); true_errs.append(te); dts.append(target_dt)

    # ── Column 1: 2D projection of two balls' trajectories
    ax_input.plot(traj[:, 0], traj[:, 1], "-", color="#1f77b4", label="ball 1", linewidth=1.5)
    ax_input.plot(traj[:, 3], traj[:, 4], "-", color="#d62728", label="ball 2", linewidth=1.5)
    ax_input.scatter(traj[0, 0], traj[0, 1], color="#1f77b4", s=60, marker="o", zorder=5)
    ax_input.scatter(traj[0, 3], traj[0, 4], color="#d62728", s=60, marker="o", zorder=5)
    ax_input.set_xlim(-2.2, 2.2); ax_input.set_ylim(-2.2, 2.2)
    ax_input.set_aspect("equal")
    ax_input.legend(loc="upper right", fontsize=8)
    ax_input.set_xlabel("x", fontsize=9); ax_input.set_ylabel("y", fontsize=9)
    ax_input.set_title("Trajectories $(x, y)$ projection", fontsize=10)
    ax_input.grid(alpha=0.25)

    # ── Column 2: e-hat per horizon (bars)
    ax_ehat.bar(range(len(horizons)), e_hats, color="#cc4444", alpha=0.85)
    ax_ehat.set_xticks([0, len(horizons)-1])
    ax_ehat.set_xticklabels([f"h={horizons[0]}", f"h={horizons[-1]}"])
    ax_ehat.set_ylabel("$\\hat{e}$", fontsize=10)
    ax_ehat.set_title("$\\hat{e}$ per horizon", fontsize=10)
    ax_ehat.grid(alpha=0.25, axis="y")

    # ── Column 3: true RMSE per horizon (bars)
    ax_err.bar(range(len(horizons)), true_errs, color="#444444", alpha=0.85)
    ax_err.set_xticks([0, len(horizons)-1])
    ax_err.set_xticklabels([f"h={horizons[0]}", f"h={horizons[-1]}"])
    ax_err.set_ylabel("True RMSE", fontsize=10)
    ax_err.set_title("True RMSE per horizon", fontsize=10)
    ax_err.grid(alpha=0.25, axis="y")


def main():
    fig = plt.figure(figsize=(11.5, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    # Row labels
    row_labels = ["Oregonator\n(reaction-diffusion)",
                   "Euler 2D\n(compressible flow)",
                   "Ball 3D\n(rigid body)"]

    # Build panels
    panels = []
    for i in range(3):
        row = []
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        panels.append(row)

    panel_oreg(panels[0][0], panels[0][1], panels[0][2])
    panel_euler(panels[1][0], panels[1][1], panels[1][2])
    panel_ball3d(panels[2][0], panels[2][1], panels[2][2])

    # Add row labels on left side
    for i, lbl in enumerate(row_labels):
        panels[i][0].annotate(lbl, xy=(-0.34, 0.5),
                                xycoords="axes fraction",
                                ha="center", va="center",
                                rotation=90, fontsize=12, fontweight="bold")

    fig.suptitle("$\\hat{e}$ spatially aligns with the surrogate's true-error structure across all three environments",
                  fontsize=13, y=0.995)

    out_pdf = OUT_DIR / "figure2_emap_cross_env.pdf"
    out_png = OUT_DIR / "figure2_emap_cross_env.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
