#!/usr/bin/env python3
"""Parse the killed-mid-run log and reconstruct the JSON + figures.

The actual run was killed by the watcher right after test split completed,
so the data is in the log but no JSON was saved. This parses the per-cell
AUROC + magnitude data and produces the deliverable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
LOG = HERE / "results" / "run.log"
OUT_JSON = HERE / "results" / "richardson_auroc_test_only.json"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse():
    text = LOG.read_text()
    blocks = re.split(r"^\[(\w+)\] N=\d+ T=\d+", text, flags=re.MULTILINE)
    # blocks alternates [..., split_name, content, split_name, content, ...]
    out = {}
    for i in range(1, len(blocks), 2):
        split = blocks[i]
        content = blocks[i + 1] if i + 1 < len(blocks) else ""
        if split not in ("test", "ood_near", "ood_far"):
            continue
        cells = {}
        # Match each h block: "  h= X  wall=Y.Ys  prod ê range [a, b]  fixed ê range [c, d]"
        # followed by   "    AUROC: ours=X.XXX[lo,hi]  rich_prod=X.XXX[lo,hi]  rich_fixed=X.XXX[lo,hi]"
        pat = (r"  h=\s*(\d+)\s+wall=([\d.]+)s\s+prod ê range "
               r"\[([\d.eE+-]+),\s*([\d.eE+-]+)\]\s+fixed ê range "
               r"\[([\d.eE+-]+),\s*([\d.eE+-]+)\]\s*\n"
               r"    AUROC:\s+ours=([\d.]+)\[([\d.]+),([\d.]+)\]\s+"
               r"rich_prod=([\d.]+)\[([\d.]+),([\d.]+)\]\s+"
               r"rich_fixed=([\d.]+)\[([\d.]+),([\d.]+)\]")
        for m in re.finditer(pat, content):
            h = int(m.group(1))
            cells[h] = dict(
                wall=float(m.group(2)),
                prod_min=float(m.group(3)), prod_max=float(m.group(4)),
                fixed_min=float(m.group(5)), fixed_max=float(m.group(6)),
                ours=dict(auroc=float(m.group(7)), lo=float(m.group(8)), hi=float(m.group(9))),
                richardson_prod=dict(auroc=float(m.group(10)), lo=float(m.group(11)), hi=float(m.group(12))),
                richardson_fixed=dict(auroc=float(m.group(13)), lo=float(m.group(14)), hi=float(m.group(15))),
            )
        out[split] = cells
    return out


def main():
    data = parse()
    print(f"Parsed splits: {list(data.keys())}")
    for split, cells in data.items():
        print(f"  [{split}] horizons: {sorted(cells.keys())}")

    # Aggregate
    all_ours, all_rp, all_rf = [], [], []
    for split, cells in data.items():
        for h, c in cells.items():
            all_ours.append(c["ours"]["auroc"])
            all_rp.append(c["richardson_prod"]["auroc"])
            all_rf.append(c["richardson_fixed"]["auroc"])

    summary = {
        "scope": "test split only — process was killed before OOD splits",
        "n_pairs_per_cell": 50,
        "n_bootstrap": 1000,
        "data": data,
        "aggregate": {
            "n_cells": len(all_ours),
            "ours_mean": float(np.mean(all_ours)) if all_ours else None,
            "richardson_prod_mean": float(np.mean(all_rp)) if all_rp else None,
            "richardson_fixed_mean": float(np.mean(all_rf)) if all_rf else None,
        },
        "paper_summary": (
            "On 6 horizons × 1 split (test) = 6 cells. Richardson disagreement "
            "values are at floating-point noise level (1e-11 to 1e-3 range, "
            "with the 1e-3 outlier coming from a single chaotic state at h=8 "
            "and h=32). At h=2, Richardson AUROC matches ours within bootstrap "
            "CI (0.875 vs 0.856 prod, 0.885 vs 0.856 fixed-stencil) — "
            "small-h truncation noise happens to correlate with surrogate "
            "error. Beyond h=2, Richardson AUROC collapses to 0.39-0.55 "
            "(random) while our learned signal holds 0.81-0.97. Mean over "
            f"6 cells: ours={np.mean(all_ours):.3f}, "
            f"richardson_prod={np.mean(all_rp):.3f}, "
            f"richardson_fixed={np.mean(all_rf):.3f}. The structural argument "
            "is supported empirically: classical Richardson is not a viable "
            "trust signal at h≥4 on modern adaptive-substepped solvers."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"saved {OUT_JSON}")

    # Figure: AUROC comparison bar chart for test split
    HORIZONS = [2, 4, 8, 16, 32, 64]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    width = 0.27
    x_idx = np.arange(len(HORIZONS))
    if "test" in data:
        cells = data["test"]
        ours_v = [cells.get(h, {}).get("ours", {}).get("auroc", float("nan")) for h in HORIZONS]
        ours_lo = [cells.get(h, {}).get("ours", {}).get("lo", float("nan")) for h in HORIZONS]
        ours_hi = [cells.get(h, {}).get("ours", {}).get("hi", float("nan")) for h in HORIZONS]
        rp_v = [cells.get(h, {}).get("richardson_prod", {}).get("auroc", float("nan")) for h in HORIZONS]
        rp_lo = [cells.get(h, {}).get("richardson_prod", {}).get("lo", float("nan")) for h in HORIZONS]
        rp_hi = [cells.get(h, {}).get("richardson_prod", {}).get("hi", float("nan")) for h in HORIZONS]
        rf_v = [cells.get(h, {}).get("richardson_fixed", {}).get("auroc", float("nan")) for h in HORIZONS]
        rf_lo = [cells.get(h, {}).get("richardson_fixed", {}).get("lo", float("nan")) for h in HORIZONS]
        rf_hi = [cells.get(h, {}).get("richardson_fixed", {}).get("hi", float("nan")) for h in HORIZONS]

        def errs(vs, los, his):
            lo = [v - l for v, l in zip(vs, los)]
            hi = [h - v for v, h in zip(vs, his)]
            return [lo, hi]
        ax.bar(x_idx - width, ours_v, width=width, yerr=errs(ours_v, ours_lo, ours_hi),
                label="ours (learned signal)", color="tab:blue", alpha=0.85, capsize=4)
        ax.bar(x_idx, rp_v, width=width, yerr=errs(rp_v, rp_lo, rp_hi),
                label="Richardson (production solver)", color="tab:orange", alpha=0.85, capsize=4)
        ax.bar(x_idx + width, rf_v, width=width, yerr=errs(rf_v, rf_lo, rf_hi),
                label="Richardson (fixed-stencil)", color="tab:green", alpha=0.85, capsize=4)
        ax.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.6, label="random")
        ax.axhline(0.75, color="black", lw=0.7, ls=":", alpha=0.5, label="strong (0.75)")
        ax.set_xticks(x_idx); ax.set_xticklabels([f"h={h}" for h in HORIZONS])
        ax.set_ylabel("AUROC@q75")
        ax.set_title("test split — learned signal vs classical Richardson  (n=50 pairs/horizon, "
                      "1000-resample bootstrap 95% CI)", fontsize=11)
        ax.set_ylim(0.2, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")
        fig.tight_layout()
        out = FIG_DIR / "auroc_comparison_test.pdf"
        fig.savefig(out)
        out_png = FIG_DIR / "auroc_comparison_test.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"saved {out}, {out_png}")

    # Magnitudes figure: log-scale spread
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if "test" in data:
        cells = data["test"]
        for h in HORIZONS:
            if h not in cells: continue
            c = cells[h]
            ax.plot([h, h], [c["prod_min"], c["prod_max"]], "o-", color="tab:orange",
                     alpha=0.7, lw=2, label="Richardson(prod)" if h == 2 else None)
            ax.plot([h, h], [c["fixed_min"], c["fixed_max"]], "s-", color="tab:green",
                     alpha=0.7, lw=2, label="Richardson(fixed)" if h == 2 else None)
    ax.set_yscale("log")
    ax.set_xlabel("horizon h")
    ax.set_ylabel("Richardson disagreement (min-max range)")
    ax.set_title("Richardson disagreement magnitudes — most cells at floating-point noise level")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "richardson_magnitudes_test.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
