#!/usr/bin/env python3
"""Build Figure 3: Mode 1 to Mode 2 Pareto curve across envs.

For each env at h=64, plot the curve (RMSE, effective speedup) parameterised
by q in {0.5, 0.6, 0.75, 0.85, 0.9}, plus Mode 1 endpoint (q=1) and full-
solver endpoint (q=0).

Effective speedup at q with batch B=1:
    speedup(q) = solver_cost / (3 * surrogate_cost + (1-q) * solver_cost)

The 3x is for ê computation (3 surrogate forward passes); when q=1 (no
fallback) we'd instead just be running Mode 1, which costs 1 forward pass.
We report two curves on the same axes when distinguishing matters.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
RESULTS = HERE / "results"
QSWEEP = ROOT / "ablations" / "mode2_qsweep" / "results"
OUT_DIR = ROOT / "NeurIPS_final_paper_figures"

ENV_COLORS = {"oregonator": "#1f77b4", "euler": "#ff7f0e", "ball3d": "#2ca02c"}
ENV_LABELS = {"oregonator": "Oregonator", "euler": "Euler 2D",
              "ball3d": "Ball 3D"}
ENV_MARKERS = {"oregonator": "o", "euler": "s", "ball3d": "^"}


def main():
    bench = {
        "oregonator": json.load(open(RESULTS / "oregonator_h64.json")),
        "euler":      json.load(open(RESULTS / "euler_h64.json")),
        "ball3d":     json.load(open(RESULTS / "ball3d_h64.json")),
    }
    qsweep = {
        "oregonator": json.load(open(QSWEEP / "oregonator_test.json")),
        "euler":      json.load(open(QSWEEP / "euler_test.json")),
        "ball3d":     json.load(open(QSWEEP / "ball3d_test.json")),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for env in ["oregonator", "euler", "ball3d"]:
        b = bench[env]
        solver_ms = b["solver_ms_per_traj"]
        sur_ms = b["surrogate_ms_per_pair_by_batch"]["1"]
        qs = qsweep[env]["64"]
        m1_rmse = qs["m1_rmse_mean"]

        # Mode 1 point: speedup = solver/sur, RMSE = M1 RMSE
        m1_speedup = solver_ms / sur_ms
        # Mode 2 points at each q
        rmses = [m1_rmse]
        speedups = [m1_speedup]
        for q in [0.9, 0.85, 0.75, 0.6, 0.5]:
            r = qs["qsweep"][str(q)]["mode2_rmse"]
            # cost = 3 * sur (always) + (1-q) * solver
            cost = 3 * sur_ms + (1 - q) * solver_ms
            spd = solver_ms / cost
            rmses.append(r); speedups.append(spd)
        # full-solver endpoint: speedup=1, RMSE~0
        rmses.append(1e-4); speedups.append(1.0)

        ax.plot(rmses, speedups, marker=ENV_MARKERS[env],
                color=ENV_COLORS[env], linewidth=2, markersize=8,
                label=ENV_LABELS[env])

        # Annotate mode 1 endpoint
        ax.annotate("Mode 1", xy=(m1_rmse, m1_speedup),
                     xytext=(8, -10), textcoords="offset points",
                     fontsize=8, color=ENV_COLORS[env])

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(0.95, 1.05, "solver parity", fontsize=8, color="gray", ha="right",
             transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mode 2 RMSE  (lower is better)", fontsize=12)
    ax.set_ylabel("Effective speedup vs.\\ reference solver", fontsize=12)
    ax.set_title("Pareto frontier: dial $q$ from Mode 2 (cheap solver fallback) to Mode 1 (pure surrogate)",
                  fontsize=11)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    out_pdf = OUT_DIR / "figure3_pareto_cross_env.pdf"
    out_png = OUT_DIR / "figure3_pareto_cross_env.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
