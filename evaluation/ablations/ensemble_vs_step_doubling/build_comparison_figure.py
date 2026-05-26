#!/usr/bin/env python3
"""Build the SD vs Ensemble comparison figure (per-cell + per-pair)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "figures"))
from _paper_theme import (apply_theme, COLORS, ENV_SHORT, SPLIT_DISPLAY,
                            SPLIT_COLORS, BASELINE_DISPLAY)
apply_theme()

RES = HERE / "results"
OUT_BASE = ROOT / "evaluation" / "figures"
OUT_BASE.mkdir(parents=True, exist_ok=True)
HORIZONS = [2, 4, 8, 16, 32, 64]
ENVS = ["euler", "oregonator"]
SPLITS = ["test", "ood_near", "ood_far"]


def _load(env, split):
    p = RES / f"{env}_{split}.json"
    if not p.exists(): return None
    return json.load(open(p))


def fig_per_cell_vs_per_pair(out_path: Path):
    """Two-row x three-col grid: rows = {per-cell, per-pair}, cols = splits.
    Each panel: bars of SD vs ENS for both envs across horizons.

    Actually simpler: 2 rows × 2 cols (per-cell vs per-pair × Euler vs Oreg),
    overlay 3 splits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True,
                                sharex=True)
    # rows: 0 = per-cell, 1 = per-pair
    # cols: 0 = Euler, 1 = Oregonator
    for col, env in enumerate(ENVS):
        for row, key_suffix in enumerate(["", "per_pair_"]):
            ax = axes[row, col]
            for split in SPLITS:
                d = _load(env, split)
                if d is None: continue
                m = d["per_h_results"]
                sds = [m.get(str(h), {}).get(f"{key_suffix}auroc_step_doubling",
                                                  m.get(str(h), {}).get(f"auroc_step_doubling", float("nan")))
                          for h in HORIZONS]
                ens = [m.get(str(h), {}).get(f"{key_suffix}auroc_ensemble",
                                                  m.get(str(h), {}).get(f"auroc_ensemble", float("nan")))
                          for h in HORIZONS]
                ax.plot(HORIZONS, sds, "o-", color=SPLIT_COLORS[split],
                          linewidth=2.0, markersize=7,
                          label=f"SD {SPLIT_DISPLAY[split]}")
                ax.plot(HORIZONS, ens, "s--", color=SPLIT_COLORS[split],
                          linewidth=2.0, markersize=7, alpha=0.7,
                          label=f"Ens {SPLIT_DISPLAY[split]}")
            ax.set_xscale("log", base=2)
            ax.set_xticks(HORIZONS); ax.set_xticklabels([str(h) for h in HORIZONS])
            ax.minorticks_off()
            ax.axhline(0.5, color="#888", linestyle=":", linewidth=1.0)
            ax.axhline(0.7, color="#444", linestyle="--", linewidth=1.0)
            ax.set_ylim(0.4, 1.02)
            row_label = ("Per-cell  (spatial trust map)"
                            if row == 0 else "Per-trajectory  (Mode 2 gate)")
            col_label = ENV_SHORT[env]
            if row == 0:
                ax.set_title(f"{col_label}", fontsize=12, fontweight="bold")
            ax.set_ylabel(row_label if col == 0 else "")
            if row == 1:
                ax.set_xlabel("Prediction horizon $h$  (multi-step)")

    # Build a single legend on the upper-right axis (clean approach)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    # Dedup by label
    seen = set(); h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l in seen: continue
        seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                  ncol=3, fontsize=10, frameon=True)

    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}", flush=True)


def fig_summary_table_text(out_path: Path):
    """Print a markdown table of all SD vs Ens means, save as text."""
    lines = ["# SD vs Ensemble — full numerical summary",
              "",
              "## Per-cell AUROC", ""]
    lines.append("| Env | Split | h=2 | h=4 | h=8 | h=16 | h=32 | h=64 | mean |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for env in ENVS:
        for split in SPLITS:
            d = _load(env, split)
            if d is None: continue
            m = d["per_h_results"]
            sd_row = [m.get(str(h), {}).get("auroc_step_doubling", float("nan")) for h in HORIZONS]
            en_row = [m.get(str(h), {}).get("auroc_ensemble", float("nan")) for h in HORIZONS]
            sd_mean = float(np.nanmean(sd_row))
            en_mean = float(np.nanmean(en_row))
            lines.append(f"| {ENV_SHORT[env]} | {split} SD | "
                            + " | ".join(f"{v:.3f}" for v in sd_row)
                            + f" | **{sd_mean:.3f}** |")
            lines.append(f"| {ENV_SHORT[env]} | {split} Ens | "
                            + " | ".join(f"{v:.3f}" for v in en_row)
                            + f" | **{en_mean:.3f}** |")

    lines += ["", "## Per-trajectory (per-pair) AUROC  — Mode 2 gate signal", ""]
    lines.append("| Env | Split | h=2 | h=4 | h=8 | h=16 | h=32 | h=64 | mean |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for env in ENVS:
        for split in SPLITS:
            d = _load(env, split)
            if d is None: continue
            m = d["per_h_results"]
            sd_row = [m.get(str(h), {}).get("per_pair_auroc_step_doubling", float("nan")) for h in HORIZONS]
            en_row = [m.get(str(h), {}).get("per_pair_auroc_ensemble", float("nan")) for h in HORIZONS]
            sd_mean = float(np.nanmean(sd_row))
            en_mean = float(np.nanmean(en_row))
            lines.append(f"| {ENV_SHORT[env]} | {split} SD | "
                            + " | ".join(f"{v:.3f}" if not np.isnan(v) else "—" for v in sd_row)
                            + f" | **{sd_mean:.3f}** |")
            lines.append(f"| {ENV_SHORT[env]} | {split} Ens | "
                            + " | ".join(f"{v:.3f}" if not np.isnan(v) else "—" for v in en_row)
                            + f" | **{en_mean:.3f}** |")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path.name}", flush=True)


def main():
    print("Building SD vs Ensemble comparison...", flush=True)
    fig_per_cell_vs_per_pair(OUT_BASE / "sd_vs_ensemble_per_cell_per_pair")
    fig_summary_table_text(OUT_BASE / "sd_vs_ensemble_summary.md")


if __name__ == "__main__":
    main()
