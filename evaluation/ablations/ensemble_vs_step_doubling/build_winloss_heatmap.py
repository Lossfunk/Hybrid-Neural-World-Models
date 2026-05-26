#!/usr/bin/env python3
"""Win/loss heatmap of SD - Ensemble Δ AUROC.
Rows = (env × split) cells, columns = horizons.
Green cells = SD wins (positive Δ), Red cells = ensemble wins.
Number inside each cell = the Δ value with sign.
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
from _paper_theme import (apply_theme, ENV_SHORT, SPLIT_DISPLAY)
apply_theme()

RES = HERE / "results"
HORIZONS = [2, 4, 8, 16, 32, 64]
ENVS = ["euler", "oregonator"]
SPLITS = ["test", "ood_near", "ood_far"]


def _load(env, split):
    p = RES / f"{env}_{split}.json"
    if not p.exists(): return None
    return json.load(open(p))


def fig_winloss_heatmap(out_path: Path):
    """Two heatmaps stacked: top = per-cell Δ, bottom = per-pair Δ.
    Diverging colormap centred at 0. Annotated with Δ values inside cells."""
    rows = [(env, split) for env in ENVS for split in SPLITS]
    n_rows = len(rows)
    n_cols = len(HORIZONS)

    delta_cell = np.full((n_rows, n_cols), np.nan)
    delta_pair = np.full((n_rows, n_cols), np.nan)

    row_labels = []
    for ri, (env, split) in enumerate(rows):
        row_labels.append(f"{ENV_SHORT[env]}  •  {SPLIT_DISPLAY[split]}")
        d = _load(env, split)
        if d is None: continue
        ph = d["per_h_results"]
        for ci, h in enumerate(HORIZONS):
            v = ph.get(str(h), {})
            sd_c = v.get("auroc_step_doubling", np.nan)
            en_c = v.get("auroc_ensemble", np.nan)
            sd_p = v.get("per_pair_auroc_step_doubling", np.nan)
            en_p = v.get("per_pair_auroc_ensemble", np.nan)
            if not np.isnan(sd_c) and not np.isnan(en_c):
                delta_cell[ri, ci] = sd_c - en_c
            if not np.isnan(sd_p) and not np.isnan(en_p):
                delta_pair[ri, ci] = sd_p - en_p

    # diverging cmap: red = ensemble wins (negative Δ), green = SD wins (positive Δ)
    # centred at 0
    cmap = LinearSegmentedColormap.from_list(
        "rg_div",
        [(0.78, 0.10, 0.10), (0.94, 0.38, 0.38), (0.99, 0.85, 0.85),
         (1.00, 1.00, 1.00),
         (0.85, 0.95, 0.85), (0.40, 0.78, 0.40), (0.10, 0.55, 0.10)],
        N=256,
    )

    vmax_c = float(np.nanmax(np.abs(delta_cell)))
    vmax_p = float(np.nanmax(np.abs(delta_pair)))
    vmax = max(vmax_c, vmax_p, 0.05)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.0),
                                  constrained_layout=True,
                                  gridspec_kw={"hspace": 0.18})

    for ax, data, panel_title in [
        (axes[0], delta_cell,
            r"Per-cell $\Delta$ AUROC  (spatial trust map)"),
        (axes[1], delta_pair,
            r"Per-trajectory $\Delta$ AUROC  (Mode 2 gate signal)"),
    ]:
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        # Annotate each cell with its value
        for ri in range(n_rows):
            for ci in range(n_cols):
                v = data[ri, ci]
                if np.isnan(v):
                    continue
                txt_color = "black" if abs(v) < 0.07 else "white"
                fmt = f"{v:+.3f}" if abs(v) > 1e-3 else "0.000"
                ax.text(ci, ri, fmt, ha="center", va="center",
                          fontsize=10, fontweight="bold", color=txt_color)
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([f"h={h}" for h in HORIZONS], fontsize=10)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=4)
        # White grid between cells
        ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        # Add row mean to right
        for ri, label in enumerate(row_labels):
            row_mean = float(np.nanmean(data[ri]))
            ax.text(n_cols + 0.05, ri, f"  mean = {row_mean:+.3f}",
                      va="center", ha="left", fontsize=10, fontweight="bold",
                      color="green" if row_mean > 0 else "red")

    # Single colorbar at the right
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.06, shrink=0.85)
    cb.set_label(r"$\Delta$ AUROC  (Step-doubling $-$ Ensemble K=3)",
                    fontsize=10, fontweight="bold")
    cb.ax.text(0.5, 1.02, "SD wins", transform=cb.ax.transAxes,
                  fontsize=9, ha="center", color="green", fontweight="bold")
    cb.ax.text(0.5, -0.02, "Ens wins", transform=cb.ax.transAxes,
                  fontsize=9, ha="center", va="top", color="red", fontweight="bold")

    fig.suptitle(
        "Step-doubling vs K=3 Ensemble disagreement   ·   "
        "Green = step-doubling wins   Red = ensemble wins",
        fontsize=13, fontweight="bold", y=1.02)

    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    OUT = ROOT / "evaluation" / "figures"
    fig_winloss_heatmap(OUT / "sd_vs_ensemble_winloss_heatmap")
