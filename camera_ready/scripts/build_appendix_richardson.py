#!/usr/bin/env python3
"""Richardson comparison: AUROC and wall-clock cost on Euler 2D."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = Path(__file__).resolve().parents[1]  # camera_ready_out/
ROOT = HERE.parent.parent
SRC = ROOT / "ablations" / "richardson_auroc" / "results" / "richardson_auroc_test_only.json"


def main():
    d = json.loads(SRC.read_text())["data"]["test"]
    horizons = sorted(int(h) for h in d.keys())

    ours_auroc = [d[str(h)]["ours"]["auroc"] for h in horizons]
    ours_lo = [d[str(h)]["ours"]["lo"] for h in horizons]
    ours_hi = [d[str(h)]["ours"]["hi"] for h in horizons]

    rich_auroc = [d[str(h)]["richardson_fixed"]["auroc"] for h in horizons]
    rich_lo = [d[str(h)]["richardson_fixed"]["lo"] for h in horizons]
    rich_hi = [d[str(h)]["richardson_fixed"]["hi"] for h in horizons]
    rich_wall = [d[str(h)]["wall"] for h in horizons]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ours_err = [
        [a - lo for a, lo in zip(ours_auroc, ours_lo)],
        [hi - a for a, hi in zip(ours_auroc, ours_hi)],
    ]
    rich_err = [
        [a - lo for a, lo in zip(rich_auroc, rich_lo)],
        [hi - a for a, hi in zip(rich_auroc, rich_hi)],
    ]
    ax.errorbar(horizons, ours_auroc, yerr=ours_err, marker="o", markersize=8,
                linewidth=2.0, capsize=4, color="#cc4444", label="Step-doubling (ours)")
    ax.errorbar(horizons, rich_auroc, yerr=rich_err, marker="s", markersize=8,
                linewidth=2.0, capsize=4, color="#444", label="Classical Richardson")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons); ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Horizon $h$")
    ax.set_ylabel("AUROC against true error")
    ax.set_title("AUROC (Oregonator, test split)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.3, 1.05); ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    ax = axes[1]
    ax.bar([str(h) for h in horizons], rich_wall, color="#444", alpha=0.7,
           edgecolor="black", linewidth=0.8, label="Richardson wall (s)")
    for h, w in zip(horizons, rich_wall):
        ax.text(str(h), w + 30, f"{w:.0f}s", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Horizon $h$")
    ax.set_ylabel("Wall time per cell (s)")
    ax.set_title("Richardson cost (50 pairs/cell)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.92, "Step-doubling: $<$1 s/cell\n(2 surrogate forward passes)",
            transform=ax.transAxes, fontsize=10, color="#cc4444",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cc4444"))

    plt.tight_layout()
    out_pdf = OUT / "appendix_richardson.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "appendix_richardson.png", bbox_inches="tight", dpi=150)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
