#!/usr/bin/env python3
"""Generate the step-doubling vs error-head AUROC comparison figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT_ALL = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT_ALL / "evaluation" / "figures"))

from _theme import apply_theme, COLORS, SPLIT_DISPLAY  # noqa: E402

apply_theme()

SPLITS = ["test", "ood_near", "ood_far"]
HORIZONS = [2, 4, 8, 16, 32, 64]


def build(no_title: bool = False):
    j = json.loads((HERE / "results" / "auroc_compare.json").read_text())
    sd = j["auroc_step_doubling"]
    eh = j["auroc_error_head"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True,
                              sharey=True)
    n_h = len(HORIZONS)
    width = 0.4
    x = np.arange(n_h, dtype=float)

    for ax, split in zip(axes, SPLITS):
        sd_vals = np.array([sd[split][str(h)] for h in HORIZONS])
        eh_vals = np.array([eh[split][str(h)] for h in HORIZONS])
        ax.bar(x - width/2, sd_vals, width=width*0.92,
                color=COLORS["mode1"], edgecolor="black", linewidth=0.6,
                label=f"Step-doubling  (mean = {sd_vals.mean():.2f})")
        ax.bar(x + width/2, eh_vals, width=width*0.92,
                color=COLORS["mode3"], edgecolor="black", linewidth=0.6,
                label=f"Error head  (mean = {eh_vals.mean():.2f})")
        ax.axhline(0.5, color="#888", linestyle=":", linewidth=1.2,
                    alpha=0.7, zorder=0)
        ax.axhline(0.9, color=COLORS["highlight"], linestyle="--",
                    linewidth=1.4, alpha=0.6, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}\n{h*0.05:.2f}s" for h in HORIZONS])
        ax.set_xlabel(r"Prediction horizon  ($h \times \Delta t_{save}$ = 0.05s)")
        ax.set_ylim(0.5, 1.04)
        ax.set_title(SPLIT_DISPLAY[split], fontsize=12, loc="left")
        ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel(r"AUROC@q75 (per-cell, $n{=}80$ pairs per cell)")

    if not no_title:
        fig.suptitle(
            "Trust signal alternatives: step-doubling beats learned error head at every cell",
            fontsize=13, y=1.04,
        )

    out_dir = ROOT_ALL / "evaluation" / "figures"
    suffix = "_notitle" if no_title else ""
    for ext in ("pdf", "png"):
        p = (out_dir / f"figures{suffix}" / f"fig_error_head_compare.{ext}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig_error_head_compare ({'no-title' if no_title else 'with title'})")


if __name__ == "__main__":
    build(no_title=False)
    build(no_title=True)
    print("done.")
