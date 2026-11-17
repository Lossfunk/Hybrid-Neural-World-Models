#!/usr/bin/env python3
"""DAgger weight ablation: supervised-only vs DAgger-only vs hybrid (10% DAgger)."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = Path(__file__).resolve().parents[1]  # camera_ready_out/

ENVS = ["Oregonator", "Euler 2D", "Ball 3D"]
ENV_COLORS = ["#1b8a8a", "#d97717", "#7c3aed"]

SUP_ONLY = [0.131, 0.021, 0.025]
HYBRID   = [0.104, 0.016, 0.024]
DAGGER_ONLY = [0.378, 0.156, 0.216]


def main():
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(ENVS))
    w = 0.27
    bars1 = ax.bar(x - w, SUP_ONLY, w, label="Supervised only ($\\lambda{=}0$)",
                    color="#cccccc", edgecolor="black", linewidth=0.7)
    bars2 = ax.bar(x, HYBRID, w, label="Hybrid (default $\\lambda{=}0.1$)",
                    color=ENV_COLORS[0], edgecolor="black", linewidth=0.9)
    bars3 = ax.bar(x + w, DAGGER_ONLY, w, label="DAgger only ($\\lambda{=}1$)",
                    color="#444444", edgecolor="black", linewidth=0.7, alpha=0.7)

    for bars in (bars1, bars2, bars3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h * 1.05,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(ENVS, fontsize=11)
    ax.set_yscale("log")
    ax.set_ylabel("Final val MSE (log scale)", fontsize=11)
    ax.set_title("DAgger weight ablation", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(top=1.6)  # headroom so the legend clears the tallest bar label
    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)

    plt.tight_layout()
    out_pdf = OUT / "appendix_dagger.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "appendix_dagger.png", bbox_inches="tight", dpi=150)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
