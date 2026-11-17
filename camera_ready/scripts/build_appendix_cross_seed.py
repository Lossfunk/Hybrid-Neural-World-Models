#!/usr/bin/env python3
"""Cross-seed AUROC stability: 3 seeds × 3 envs × 3 splits."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SRC = ROOT / "ablations" / "quadrant" / "results"  # cross_seed_indep_{env}.json
OUT = Path(__file__).resolve().parents[1]  # camera_ready_out/

ENV_LABEL = {"oregonator": "Oregonator", "euler": "Euler 2D", "ball3d": "Ball 3D"}
ENV_COLOR = {"oregonator": "#1b8a8a", "euler": "#d97717", "ball3d": "#7c3aed"}


def main():
    d = {e: json.loads((SRC / f"cross_seed_indep_{e}.json").read_text())
         for e in ("oregonator", "euler", "ball3d")}
    HORIZONS = [2, 4, 8, 16, 32, 64]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)

    for c, env in enumerate(("oregonator", "euler", "ball3d")):
        ax = axes[c]
        for split, marker in [("test", "o"), ("ood_near", "s"), ("ood_far", "^")]:
            seeds_data = d[env]["seeds"]
            mean_auroc, std_auroc = [], []
            for h in HORIZONS:
                vals = [seeds_data[s][split][str(h)] for s in sorted(seeds_data)]
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                if vals:
                    mean_auroc.append(np.mean(vals)); std_auroc.append(np.std(vals))
                else:
                    mean_auroc.append(np.nan); std_auroc.append(0)
            mean_auroc = np.array(mean_auroc); std_auroc = np.array(std_auroc)
            ax.errorbar(HORIZONS, mean_auroc, yerr=std_auroc,
                          marker=marker, markersize=7, capsize=4,
                          label=split.replace("_", "-"), linewidth=1.6,
                          color=ENV_COLOR[env] if split == "test" else None,
                          alpha=1.0 if split == "test" else 0.55)
        ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
        ax.set_xticks(HORIZONS); ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.set_xlabel("Horizon $h$")
        if c == 0: ax.set_ylabel("AUROC against true error")
        ax.set_title(ENV_LABEL[env], fontsize=12, fontweight="bold")
        ax.set_ylim(0.3, 1.05); ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", fontsize=9, frameon=False)

    plt.tight_layout()
    out_pdf = OUT / "appendix_cross_seed.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "appendix_cross_seed.png", bbox_inches="tight", dpi=150)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
