#!/usr/bin/env python3
"""Figure 1 v2: 4-line Pareto. Each env contributes a Mode 1 curve
(parametrised by horizon) and a Mode 2 curve (parametrised by q at
h=64). Total 4 lines for clarity.

Panel A (left): CPU-CPU speedup vs horizon, 4 lines (Mode 1 and Mode 2
for each env).

Panel B (right): Pareto frontier in (RMSE, speedup), 4 lines:
  - Oregonator Mode 1 across horizons {2, 4, 8, 16, 32, 64}
  - Oregonator Mode 2 across q {0.9, 0.85, 0.75, 0.6, 0.5} at h=64
  - Euler Mode 1 across horizons
  - Euler Mode 2 across q at h=64
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ablations" / "batch_bench" / "results"
QSWEEP = ROOT / "ablations" / "mode2_qsweep" / "results"
OUT = Path(__file__).resolve().parents[1] / "fig1_unspliced"
OUT.mkdir(exist_ok=True)

ENV_COLOR = {"oregonator": "#1f77b4", "euler": "#ff7f0e"}
ENV_LABEL = {"oregonator": "Oregonator", "euler": "Euler 2D"}

HORIZONS = [2, 4, 8, 16, 32, 64]
Q_VALUES = [0.9, 0.85, 0.75, 0.6, 0.5]
Q_DEFAULT = 0.75


def m2_speedup(solver_ms, sur_ms, q_keep):
    cost = 3 * sur_ms + (1 - q_keep) * solver_ms
    return solver_ms / cost


def main():
    cpu = json.load(open(RESULTS / "cpu_h64.json"))
    qs = {
        "oregonator": json.load(open(QSWEEP / "oregonator_test.json")),
        "euler":      json.load(open(QSWEEP / "euler_test.json")),
    }
    m1_rmse = json.load(open(RESULTS / "m1_rmse_horizons.json"))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ====================================================================
    # Panel A: speedup vs horizon (CPU-CPU). 4 lines: M1 and M2 per env.
    # ====================================================================
    sweep = json.load(open(RESULTS / "cpu_horizon_sweep.json"))  # measured at each h
    for env in ["oregonator", "euler"]:
        m1_speedups = []
        m2_speedups = []
        for h in HORIZONS:
            solver_h = sweep[env][str(h)]["solver_ms"]
            sur_h = sweep[env][str(h)]["surrogate_ms"]
            m1_speedups.append(solver_h / sur_h)
            m2_speedups.append(m2_speedup(solver_h, sur_h, q_keep=Q_DEFAULT))

        c = ENV_COLOR[env]
        axA.plot(HORIZONS, m1_speedups, marker="o", color=c, linewidth=2.4,
                 markersize=8, label=f"{ENV_LABEL[env]}, Mode 1")
        axA.plot(HORIZONS, m2_speedups, marker="s", color=c, linewidth=2.0,
                 markersize=6.5, linestyle="--", alpha=0.8,
                 label=f"{ENV_LABEL[env]}, Mode 2 ($q{{=}}0.75$)")

        # Speedup arrow at h=64 for Mode 1: place above for Oreg, below for Euler
        if env == "oregonator":
            offset = (-55, 26)
        else:
            offset = (-12, -32)
        axA.annotate(f"{m1_speedups[-1]:.0f}$\\times$",
                     xy=(64, m1_speedups[-1]),
                     xytext=offset,
                     textcoords="offset points",
                     fontsize=13, fontweight="bold", color=c,
                     arrowprops=dict(arrowstyle="->",
                                      color=c, lw=1.6,
                                      shrinkA=0, shrinkB=4))

    axA.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, zorder=0)
    axA.text(64, 1.0, "solver ($1\\times$)",
             color="gray", fontsize=9, ha="right", va="top",
             bbox=dict(facecolor="white", edgecolor="none", pad=2.0))
    axA.set_xscale("log", base=2)
    axA.set_yscale("log")
    axA.set_xticks(HORIZONS)
    axA.set_xticklabels([str(h) for h in HORIZONS])
    axA.set_xlabel("Horizon $h$ (steps)", fontsize=11)
    axA.set_ylabel("Speedup over reference solver (log)", fontsize=11)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=9, framealpha=0.95)
    axA.grid(True, which="both", alpha=0.25)
    axA.set_axisbelow(True)
    axA.set_ylim(0.2, 200)

    # ====================================================================
    # Panel B: clean Pareto at h=64, 2 lines (one per env), q sweep.
    # Mode 1 endpoint at top; q values walking down to solver.
    # ====================================================================
    for env in ["oregonator", "euler"]:
        d = cpu[env]
        solver_ms_64 = d["solver_ms_per_traj"]
        sur_ms = d["surrogate_batch_ms"]["1"]

        # Build full curve: Mode 1 (q=1 endpoint) -> Mode 2 q values -> solver
        env_qs = qs[env]["64"]
        m1_rmse_64 = env_qs["m1_rmse_mean"]
        rmses = [m1_rmse_64]
        speeds = [solver_ms_64 / sur_ms]
        for q in Q_VALUES:
            r = env_qs["qsweep"][str(q)]["mode2_rmse"]
            s = m2_speedup(solver_ms_64, sur_ms, q_keep=q)
            rmses.append(r)
            speeds.append(s)
        rmses.append(1e-4)
        speeds.append(1.0)

        c = ENV_COLOR[env]
        # Single bold curve from Mode 1 to solver, passing through q values
        axB.plot(rmses, speeds, color=c, linewidth=2.6, alpha=0.85,
                 zorder=2, label=f"{ENV_LABEL[env]}")
        # Mode 1 endpoint as big circle
        axB.plot([rmses[0]], [speeds[0]], marker="o", color=c,
                 markersize=14, linestyle="None", zorder=3,
                 markeredgecolor="white", markeredgewidth=1.5)
        # Mode 2 q points as squares
        for qi in range(1, len(Q_VALUES) + 1):
            axB.plot([rmses[qi]], [speeds[qi]], marker="s",
                     color=c, markersize=8, linestyle="None",
                     zorder=3, markeredgecolor="white",
                     markeredgewidth=1.0)

        # Annotate Mode 1 endpoint
        axB.annotate("Mode 1",
                      xy=(rmses[0], speeds[0]),
                      xytext=(-8, 14), textcoords="offset points",
                      fontsize=10.5, color=c, fontweight="bold",
                      ha="right")
        # Annotate q=0.75
        idx_75 = 1 + Q_VALUES.index(0.75)
        axB.annotate(f"$q{{=}}0.75$",
                      xy=(rmses[idx_75], speeds[idx_75]),
                      xytext=(10, -2), textcoords="offset points",
                      fontsize=9.5, color=c)

    # Solver endpoint (gray triangle, shared for both envs)
    axB.plot([1e-4], [1.0], marker="^", color="gray", markersize=11,
             markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    axB.annotate("reference solver\n($1\\times$, exact)",
                  xy=(1e-4, 1.0), xytext=(15, 8),
                  textcoords="offset points",
                  fontsize=9.5, color="gray", fontweight="bold",
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    axB.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5,
                zorder=0)
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(5e-5, 0.6)
    axB.set_ylim(0.6, 200)
    axB.set_xlabel("RMSE  (lower is better)", fontsize=11)
    axB.set_ylabel("Effective speedup (log)", fontsize=11)
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=9, framealpha=0.95)
    axB.grid(True, which="both", alpha=0.25)
    axB.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT / "fig1_speedup_two_panel.pdf", bbox_inches="tight", dpi=200)
    plt.savefig(OUT / "fig1_speedup_two_panel.png", bbox_inches="tight", dpi=200)
    print(f"Wrote {OUT / 'fig1_speedup_two_panel.pdf'}")
    print(f"Wrote {OUT / 'fig1_speedup_two_panel.png'}")


if __name__ == "__main__":
    main()
