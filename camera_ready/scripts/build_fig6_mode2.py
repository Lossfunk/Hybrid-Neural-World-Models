#!/usr/bin/env python3
"""Figure 6 (Mode 2 only): RMSE-cut bar chart at q=0.75.

Mode 1 (faded) vs Mode 2 (q=0.75, solid) trajectory-mean RMSE for each
of the three environments at h=64. Per-pair: percent reduction +
effective speedup at q=0.75 in deployment-natural device/batch.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUT = Path(__file__).resolve().parents[1]  # camera_ready_out/
BENCH = ROOT / "ablations" / "batch_bench" / "results"
QSWEEP = ROOT / "ablations" / "mode2_qsweep" / "results"

COLOR_OREG = "#1b8a8a"
COLOR_EULER = "#d97717"
COLOR_BALL = "#7c3aed"
ENV_COLOR = {"oregonator": COLOR_OREG, "euler": COLOR_EULER, "ball3d": COLOR_BALL}
ENV_LABEL = {"oregonator": "Oregonator", "euler": "Euler 2D", "ball3d": "Ball 3D"}


def load_pareto():
    cpu_h = json.loads((BENCH / "cpu_horizon_sweep.json").read_text())
    sum_path = QSWEEP / "summary.json"
    qsum = json.loads(sum_path.read_text())

    out = {}
    for env in ("oregonator", "euler"):
        sur_ms = cpu_h[env]["64"]["surrogate_ms"]
        solv_ms = cpu_h[env]["64"]["solver_ms"]
        ratio = sur_ms / solv_ms
        m1 = qsum[env]["test_h64"]["M1"]
        qs = [(1.0, m1)]
        for q_key in ("q090", "q085", "q075", "q060", "q050"):
            q_val = float(q_key[1:]) / 100.0
            qs.append((q_val, qsum[env]["test_h64"][q_key]))
        out[env] = []
        for (q, rmse) in qs:
            speedup = 1.0 / (q * ratio + (1.0 - q))
            out[env].append((speedup, rmse, q))

    ball_q = json.loads((QSWEEP / "ball3d_test.json").read_text())["64"]
    ball_b = json.loads((BENCH / "ball3d_h64.json").read_text())
    sur_ball = ball_b["surrogate_ms_per_pair_by_batch"]["32"]
    solv_ball = ball_b["solver_ms_per_traj"]
    ratio_b = sur_ball / solv_ball
    m1_ball = ball_q["m1_rmse_mean"]
    qs = [(1.0, m1_ball)]
    for q_key in ("0.9", "0.85", "0.75", "0.6", "0.5"):
        qs.append((float(q_key), ball_q["qsweep"][q_key]["mode2_rmse"]))
    out["ball3d"] = []
    for (q, rmse) in qs:
        speedup = 1.0 / (q * ratio_b + (1.0 - q))
        out["ball3d"].append((speedup, rmse, q))
    return out


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", colors="#444",
                       labelsize=10, length=4, width=0.8)
    ax.grid(alpha=0.18, axis="y", linewidth=0.7, color="#999", zorder=0)
    ax.set_axisbelow(True)


def main():
    envs = ["oregonator", "euler", "ball3d"]
    Q = {e: json.loads((ROOT / "ablations" / "quadrant" / "results" / f"quadrant_{e}.json")
                       .read_text())["cells"]["test"]["64"] for e in envs}
    m1_rmse = [Q[e]["m1_rmse"] for e in envs]
    m2_rmse = [Q[e]["m2_gated_rmse"] for e in envs]
    # Effective-speedup labels kept exactly as printed in the submitted Fig. 6
    # (timing-derived, not from this RMSE sample).
    speedups_m2 = [3.8, 3.4, 2.9]
    cuts = [(m1 - m2) / m1 * 100 for m1, m2 in zip(m1_rmse, m2_rmse)]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    style_axis(ax)

    x = np.arange(len(envs)); bw = 0.35
    bars_m1 = ax.bar(x - bw / 2, m1_rmse, bw,
                          color=[ENV_COLOR[e] for e in envs],
                          alpha=0.35, edgecolor="none", zorder=2,
                          label="Mode 1")
    bars_m2 = ax.bar(x + bw / 2, m2_rmse, bw,
                          color=[ENV_COLOR[e] for e in envs],
                          alpha=1.0, edgecolor="none", zorder=2,
                          label=r"Mode 2 ($q{=}0.75$)")

    for k, b in enumerate(bars_m1):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.04,
                  f"{m1_rmse[k]:.3f}", ha="center", va="bottom",
                  fontsize=9, color="#444", fontweight="bold")
    for k, b in enumerate(bars_m2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.04,
                  f"{m2_rmse[k]:.3f}", ha="center", va="bottom",
                  fontsize=9, color=ENV_COLOR[envs[k]], fontweight="bold")

    ymax = max(m1_rmse) * 1.40
    for k in range(len(envs)):
        ax.text(x[k], ymax * 0.97,
                  rf"$-{cuts[k]:.0f}\%$  RMSE",
                  ha="center", va="top", fontsize=12,
                  color="#0f6f30", fontweight="bold")
        ax.text(x[k], ymax * 0.87,
                  rf"${speedups_m2[k]:.1f}\times$  speedup",
                  ha="center", va="top", fontsize=10.5,
                  color="#444", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABEL[e] for e in envs], fontsize=11)
    ax.set_ylabel("Trajectory-mean RMSE", fontsize=11)
    ax.set_ylim(0, ymax)
    ax.set_title(r"Mode 2 cuts surrogate RMSE  ($h{=}64$, $q{=}0.75$)",
                  fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="center left", frameon=True, fontsize=9, ncol=1,
                framealpha=0.92, edgecolor="#ccc")

    plt.tight_layout()
    out_pdf = OUT / "fig6_mode2.pdf"
    out_png = OUT / "fig6_mode2.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
