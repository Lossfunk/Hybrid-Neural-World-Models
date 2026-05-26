#!/usr/bin/env python3
"""Build Figure 1: surrogate-vs-solver speedup story (2 panels side-by-side).

Left: speedup at B=1 vs horizon (3 lines, one per env).
Right: speedup at h=64 vs batch size (3 lines, one per env).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
RESULTS = HERE / "results"
OUT_DIR = ROOT / "NeurIPS_final_paper_figures"
OUT_DIR.mkdir(exist_ok=True)

ENV_COLORS = {"oregonator": "#1f77b4", "euler": "#ff7f0e", "ball3d": "#2ca02c"}
ENV_LABELS = {"oregonator": "Oregonator (PDE)", "euler": "Euler 2D (PDE)",
              "ball3d": "Ball 3D (rigid body)"}
ENV_MARKERS = {"oregonator": "o", "euler": "s", "ball3d": "^"}


def main():
    horizon = json.load(open(RESULTS / "horizon_sweep.json"))
    h64 = {
        "oregonator": json.load(open(RESULTS / "oregonator_h64.json")),
        "euler":      json.load(open(RESULTS / "euler_h64.json")),
        "ball3d":     json.load(open(RESULTS / "ball3d_h64.json")),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # ── LEFT: speedup vs horizon at B=1
    ax = axes[0]
    for env in ["oregonator", "euler", "ball3d"]:
        d = horizon[env]
        hs = sorted(int(h) for h in d.keys())
        speedups = [d[str(h)]["speedup"] for h in hs]
        ax.plot(hs, speedups, marker=ENV_MARKERS[env], color=ENV_COLORS[env],
                label=ENV_LABELS[env], linewidth=2, markersize=7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6,
                label="parity (1×)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon h (steps)", fontsize=12)
    ax.set_ylabel("Speedup vs.\\ reference solver", fontsize=12)
    ax.set_title("Single-trajectory speedup ($B{=}1$)", fontsize=13)
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    # ── RIGHT: speedup vs batch at h=64
    ax = axes[1]
    for env in ["oregonator", "euler", "ball3d"]:
        d = h64[env]["speedup_by_batch"]
        Bs = sorted(int(b) for b in d.keys())
        speedups = [d[str(b)] for b in Bs]
        ax.plot(Bs, speedups, marker=ENV_MARKERS[env], color=ENV_COLORS[env],
                label=ENV_LABELS[env], linewidth=2, markersize=7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size $B$", fontsize=12)
    ax.set_ylabel("Speedup vs.\\ reference solver", fontsize=12)
    ax.set_title("Batch-mode speedup at $h{=}64$", fontsize=13)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out_pdf = OUT_DIR / "figure1_speedup_two_panel.pdf"
    out_png = OUT_DIR / "figure1_speedup_two_panel.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
