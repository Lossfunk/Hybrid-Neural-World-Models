#!/usr/bin/env python3
"""Headline SD-vs-Ensemble figure: per-pair (Mode 2 gate) AUROC bars across
6 cells (2 envs × 3 splits), with per-horizon overlay below."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "figures"))
from _paper_theme import (apply_theme, COLORS, ENV_SHORT, SPLIT_DISPLAY,
                            SPLIT_COLORS, BASELINE_DISPLAY)
apply_theme()

RES = HERE / "results"
HORIZONS = [2, 4, 8, 16, 32, 64]
ENVS = ["euler", "oregonator"]
SPLITS = ["test", "ood_near", "ood_far"]


def _load(env, split):
    p = RES / f"{env}_{split}.json"
    if not p.exists(): return None
    return json.load(open(p))


def fig_headline(out_path: Path):
    """Two-panel: top = per-pair AUROC mean bars per (env × split) for SD vs ENS;
    bottom = per-horizon line plot for h=64 (the 'long-horizon win' headline)."""
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.4])

    # ─── Top: bar chart of per-pair mean AUROC across splits ─────────────
    ax0 = fig.add_subplot(gs[0, 0])
    cells = [(env, split) for env in ENVS for split in SPLITS]   # 6 cells
    sd_means = []; en_means = []; cell_labels = []
    for env, split in cells:
        d = _load(env, split)
        if d is None:
            sd_means.append(np.nan); en_means.append(np.nan)
        else:
            ph = d["per_h_results"]
            sds = [ph[h].get("per_pair_auroc_step_doubling", np.nan) for h in ph]
            ens = [ph[h].get("per_pair_auroc_ensemble", np.nan) for h in ph]
            sd_means.append(float(np.nanmean(sds)))
            en_means.append(float(np.nanmean(ens)))
        cell_labels.append(f"{ENV_SHORT[env]}\n{SPLIT_DISPLAY[split]}")

    x = np.arange(len(cells), dtype=float)
    w = 0.36
    bar_sd = ax0.bar(x - w/2, sd_means, width=w * 0.92, color=COLORS["mode1"],
                          edgecolor="black", linewidth=0.5,
                          label=fr"Step-doubling $\hat{{e}}$ (ours, K=1 model)")
    bar_en = ax0.bar(x + w/2, en_means, width=w * 0.92, color=COLORS["ensemble"],
                          edgecolor="black", linewidth=0.5,
                          label="K=3 ensemble disagreement (3 models)")
    # Annotate winning bars
    for i, (sd, en) in enumerate(zip(sd_means, en_means)):
        if not np.isnan(sd) and not np.isnan(en):
            d = sd - en
            xv, yv = (i - w/2, sd) if sd > en else (i + w/2, en)
            color = COLORS["mode1"] if sd > en else COLORS["ensemble"]
            ax0.annotate(f"{'+' if d > 0 else ''}{d:.3f}",
                            xy=(xv, yv), ha="center", va="bottom",
                            fontsize=9, fontweight="bold",
                            color=color, xytext=(0, 4),
                            textcoords="offset points")
    ax0.set_xticks(x); ax0.set_xticklabels(cell_labels, fontsize=9)
    ax0.axhline(0.7, color="#444", linestyle="--", linewidth=1.0,
                  label="Useful threshold ($AUROC=0.7$)")
    ax0.axhline(0.5, color="#888", linestyle=":", linewidth=1.0,
                  label="Random ($AUROC=0.5$)")
    ax0.set_ylabel(r"Per-trajectory AUROC  (Mode 2 gate signal)")
    ax0.set_ylim(0.5, 1.05)
    ax0.legend(loc="lower left", fontsize=10, ncol=2, framealpha=0.95)

    # ─── Bottom: line plot per horizon, focused on h=64 long-horizon win ─
    ax1 = fig.add_subplot(gs[1, 0])
    # Per-horizon SD - ENS Δ for each cell
    for env in ENVS:
        for split in SPLITS:
            d = _load(env, split)
            if d is None: continue
            ph = d["per_h_results"]
            deltas = []
            for h in HORIZONS:
                v = ph.get(str(h), {})
                sd = v.get("per_pair_auroc_step_doubling", np.nan)
                en = v.get("per_pair_auroc_ensemble", np.nan)
                deltas.append(sd - en)
            ls = "-" if env == "oregonator" else "--"
            marker = "o" if env == "oregonator" else "s"
            ax1.plot(HORIZONS, deltas, marker + ls,
                       color=SPLIT_COLORS[split],
                       linewidth=2.0, markersize=7,
                       label=f"{ENV_SHORT[env]} {SPLIT_DISPLAY[split]}")
    ax1.axhline(0.0, color="#444", linewidth=1.5)
    ax1.fill_between([1, 100], 0, 1, color=COLORS["mode1"], alpha=0.07,
                       zorder=0, label="Step-doubling wins")
    ax1.fill_between([1, 100], -1, 0, color=COLORS["ensemble"], alpha=0.07,
                       zorder=0, label="Ensemble wins")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(HORIZONS); ax1.set_xticklabels([str(h) for h in HORIZONS])
    ax1.minorticks_off()
    ax1.set_xlim(1.7, 80)
    ax1.set_ylim(-0.15, 0.15)
    ax1.set_xlabel(r"Prediction horizon $h$  (multi-step)")
    ax1.set_ylabel(r"Per-pair $\Delta$ AUROC  (SD $-$ Ens)")
    ax1.legend(loc="upper left", fontsize=9, ncol=2)

    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    OUT = ROOT / "evaluation" / "figures"
    fig_headline(OUT / "sd_vs_ensemble_headline")
