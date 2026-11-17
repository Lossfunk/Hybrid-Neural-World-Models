#!/usr/bin/env python3
"""Per-horizon Mode 1 vs Mode 2 RMSE across all 54 cells (3 envs x 6 h x 3 splits)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SRC = ROOT / "ablations" / "quadrant" / "results"  # quadrant_{env}.json
OUT = Path(__file__).resolve().parents[1]  # camera_ready_out/

ENV_LABEL = {"oregonator": "Oregonator", "euler": "Euler 2D", "ball3d": "Ball 3D"}
ENV_COLOR = {"oregonator": "#1b8a8a", "euler": "#d97717", "ball3d": "#7c3aed"}
SPLIT_LABEL = {"test": "test", "ood_near": "OOD-near", "ood_far": "OOD-far"}
HORIZONS = [2, 4, 8, 16, 32, 64]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

    for r, env in enumerate(("oregonator", "euler", "ball3d")):
        for c, split in enumerate(("test", "ood_near", "ood_far")):
            ax = axes[r, c]
            d = json.loads((SRC / f"quadrant_{env}.json").read_text())["cells"][split]
            m1 = [d[str(h)]["m1_rmse"] for h in HORIZONS]
            m2 = [d[str(h)]["m2_gated_rmse"] for h in HORIZONS]
            color = ENV_COLOR[env]
            ax.plot(HORIZONS, m1, marker="o", markersize=7, linewidth=1.8,
                    color=color, alpha=0.45, linestyle="--", label="Mode 1")
            ax.plot(HORIZONS, m2, marker="s", markersize=7, linewidth=2.0,
                    color=color, label="Mode 2 ($q{=}0.75$)")
            for h, a, b in zip(HORIZONS, m1, m2):
                if a > 1e-6:
                    cut = (a - b) / a * 100
                    ax.annotate(f"-{cut:.0f}%",
                                (h, b), textcoords="offset points",
                                xytext=(0, -14), ha="center", fontsize=7,
                                color="#2a8a2a", fontweight="bold")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(HORIZONS); ax.set_xticklabels([str(h) for h in HORIZONS])
            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if r == 0:
                ax.set_title(SPLIT_LABEL[split], fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{ENV_LABEL[env]}\nTrajectory RMSE", fontsize=10)
            if r == 2:
                ax.set_xlabel("Horizon $h$")
            ax.legend(loc="best", fontsize=8, frameon=False)

    plt.tight_layout()
    out_pdf = OUT / "appendix_m1_vs_m2.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "appendix_m1_vs_m2.png", bbox_inches="tight", dpi=150)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
