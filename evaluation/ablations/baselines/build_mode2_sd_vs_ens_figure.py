#!/usr/bin/env python3
"""Mode 2 SD-gate vs ENS-gate — comparison figure.

For each (env, split, h):
  - SD-gate Mode 2 RMSE vs ENS-gate Mode 2 RMSE at fixed q75 trigger rate.
Lower = better gate. The paper claim: SD-gate produces lower Mode 2 RMSE.

Figure: 1 row per env × 3 splits, x = horizon, y = Mode 2 RMSE for both gates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "figures"))
from _paper_theme import (apply_theme, COLORS, ENV_SHORT, SPLIT_DISPLAY,
                            SPLIT_COLORS)
apply_theme()

RES = HERE / "results"
HORIZONS = [2, 4, 8, 16, 32, 64]
ENVS = ["oregonator", "euler", "ball3d"]
SPLITS = ["test", "ood_near", "ood_far"]


def _load(env, split):
    p = RES / f"mode2_sd_vs_ens_{env}_{split}.json"
    if not p.exists(): return None
    return json.load(open(p))


def fig_winloss_heatmap(out_path: Path):
    """Win/loss heatmap of (Mode 2 ENS RMSE − Mode 2 SD RMSE).
    Positive = SD gate wins (less RMSE). Color positive = green."""
    rows = [(env, split) for env in ENVS for split in SPLITS]
    n_rows = len(rows); n_cols = len(HORIZONS)
    delta = np.full((n_rows, n_cols), np.nan)
    row_labels = []
    for ri, (env, split) in enumerate(rows):
        row_labels.append(f"{ENV_SHORT[env]} • {SPLIT_DISPLAY[split]}")
        d = _load(env, split)
        if d is None: continue
        for ci, h in enumerate(HORIZONS):
            v = d.get(str(h), {})
            if not v: continue
            sd_rmse = v.get("m2_sd_gate_rmse", np.nan)
            en_rmse = v.get("m2_ens_gate_rmse", np.nan)
            if np.isnan(sd_rmse) or np.isnan(en_rmse): continue
            delta[ri, ci] = en_rmse - sd_rmse        # positive = SD wins

    cmap = LinearSegmentedColormap.from_list(
        "rg_div",
        [(0.78, 0.10, 0.10), (0.94, 0.38, 0.38), (0.99, 0.85, 0.85),
         (1.00, 1.00, 1.00),
         (0.85, 0.95, 0.85), (0.40, 0.78, 0.40), (0.10, 0.55, 0.10)],
        N=256,
    )
    abs_max = float(np.nanmax(np.abs(delta)))
    if abs_max < 1e-10: abs_max = 0.01
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    im = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto")
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = delta[ri, ci]
            if np.isnan(v): continue
            txt_color = "black" if abs(v) < abs_max * 0.5 else "white"
            ax.text(ci, ri, f"{v:+.4f}", ha="center", va="center",
                      fontsize=10, fontweight="bold", color=txt_color)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for ri, label in enumerate(row_labels):
        rm = float(np.nanmean(delta[ri]))
        ax.text(n_cols + 0.05, ri, f"  mean = {rm:+.4f}",
                  va="center", ha="left", fontsize=10, fontweight="bold",
                  color="green" if rm > 0 else "red")
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.04, shrink=0.85)
    cb.set_label(r"Mode 2 RMSE  (ENS gate $-$ SD gate)",
                    fontsize=10, fontweight="bold")
    cb.ax.text(0.5, 1.02, "SD gate better", transform=cb.ax.transAxes,
                  fontsize=9, ha="center", color="green", fontweight="bold")
    cb.ax.text(0.5, -0.02, "ENS gate better", transform=cb.ax.transAxes,
                  fontsize=9, ha="center", va="top", color="red", fontweight="bold")
    fig.suptitle(
        "Mode 2 RMSE: SD gate vs ENS gate   ·   "
        "Green = SD gate produces lower RMSE",
        fontsize=13, fontweight="bold", y=1.02)

    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.name}", flush=True)


def fig_lines_per_env(out_path: Path):
    """Per-env line plot of Mode 2 RMSE vs h, SD gate vs ENS gate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True,
                                  sharey=False)
    for ax, env in zip(axes, ENVS):
        for split in SPLITS:
            d = _load(env, split)
            if d is None: continue
            sd_rmse = []; en_rmse = []
            for h in HORIZONS:
                v = d.get(str(h), {})
                sd_rmse.append(v.get("m2_sd_gate_rmse", np.nan))
                en_rmse.append(v.get("m2_ens_gate_rmse", np.nan))
            ax.plot(HORIZONS, sd_rmse, "o-", color=SPLIT_COLORS[split],
                      linewidth=2.0, label=f"SD gate   {SPLIT_DISPLAY[split]}")
            ax.plot(HORIZONS, en_rmse, "s--", color=SPLIT_COLORS[split],
                      linewidth=2.0, alpha=0.7,
                      label=f"ENS gate {SPLIT_DISPLAY[split]}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(HORIZONS); ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.minorticks_off()
        ax.set_xlabel(r"Horizon $h$  (multi-step)")
        ax.set_ylabel("Mode 2 RMSE  (lower is better)")
        ax.set_title(ENV_SHORT[env], fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.name}", flush=True)


def main():
    OUT = ROOT / "evaluation" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    fig_winloss_heatmap(OUT / "mode2_sd_vs_ens_winloss")
    fig_lines_per_env(OUT / "mode2_sd_vs_ens_lines")


if __name__ == "__main__":
    main()
