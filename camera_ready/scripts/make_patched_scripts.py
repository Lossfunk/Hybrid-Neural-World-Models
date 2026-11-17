#!/usr/bin/env python3
"""Create camera-ready copies of the original figure builders with minimal patches.

Each original script in neurips_FINAL_FIGS/ is copied here and patched with exact
string replacements (each must match exactly once, or this script aborts). Only
data sources, output paths and the specific changes requested in the camera-ready
brief are touched; styling, sizes and fonts are untouched.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[1] / "neurips_FINAL_FIGS"
OUT_DECL = 'OUT = Path(__file__).resolve().parents[1]  # camera_ready_out/\n'


def patch(src_rel, dst_name, repls):
    text = (FIGS / src_rel).read_text()
    for old, new in repls:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{src_rel}: expected 1 match, found {n}: {old[:70]!r}")
        text = text.replace(old, new)
    (HERE / dst_name).write_text(text)
    print(f"wrote {dst_name}")


# Task 2 -- Fig 9 from the independent-draw cross-seed files (a).
patch("appendix_figures/build_appendix_cross_seed.py", "build_appendix_cross_seed.py", [
    ('SRC = ROOT / "ablations" / "ensemble_vs_step_doubling" / "results" / "cross_seed_auroc.json"\n',
     'SRC = ROOT / "ablations" / "quadrant" / "results"  # cross_seed_indep_{env}.json\n' + OUT_DECL),
    ('    d = json.loads(SRC.read_text())\n',
     '    d = {e: json.loads((SRC / f"cross_seed_indep_{e}.json").read_text())\n'
     '         for e in ("oregonator", "euler", "ball3d")}\n'),
    ('            seeds_data = d[env][split]\n', '            seeds_data = d[env]["seeds"]\n'),
    ('                vals = [seeds_data[s][str(h)].get("cell_auroc")\n'
     '                          if not np.isnan(seeds_data[s][str(h)].get("cell_auroc", float("nan")))\n'
     '                          else seeds_data[s][str(h)].get("pair_auroc")\n'
     '                          for s in seeds_data if str(h) in seeds_data[s]]\n',
     '                vals = [seeds_data[s][split][str(h)] for s in sorted(seeds_data)]\n'),
    ('    out_pdf = HERE / "appendix_cross_seed.pdf"', '    out_pdf = OUT / "appendix_cross_seed.pdf"'),
    ('plt.savefig(HERE / "appendix_cross_seed.png"', 'plt.savefig(OUT / "appendix_cross_seed.png"'),
])

# Task 4 -- Fig 10 from the matched-sample quadrant files (b).
patch("appendix_figures/build_appendix_m1_vs_m2.py", "build_appendix_m1_vs_m2.py", [
    ('SRC = ROOT / "ablations" / "mode2_qsweep" / "results" / "full_horizons"\n',
     'SRC = ROOT / "ablations" / "quadrant" / "results"  # quadrant_{env}.json\n' + OUT_DECL),
    ('            d = json.loads((SRC / f"{env}_{split}.json").read_text())\n'
     '            m1 = [d[str(h)]["m1_rmse_mean"] for h in HORIZONS]\n'
     '            m2 = [d[str(h)]["qsweep"]["0.75"]["mode2_rmse"] for h in HORIZONS]\n',
     '            d = json.loads((SRC / f"quadrant_{env}.json").read_text())["cells"][split]\n'
     '            m1 = [d[str(h)]["m1_rmse"] for h in HORIZONS]\n'
     '            m2 = [d[str(h)]["m2_gated_rmse"] for h in HORIZONS]\n'),
    ('    out_pdf = HERE / "appendix_m1_vs_m2.pdf"', '    out_pdf = OUT / "appendix_m1_vs_m2.pdf"'),
    ('plt.savefig(HERE / "appendix_m1_vs_m2.png"', 'plt.savefig(OUT / "appendix_m1_vs_m2.png"'),
])

# Task 5 -- Fig 6 RMSE bars from (b), test split, h=64; speedup labels kept as submitted.
patch("build_fig6_mode2.py", "build_fig6_mode2.py", [
    ('ROOT = HERE.parent\n', 'ROOT = HERE.parent.parent\n' + OUT_DECL),
    ('    C = load_pareto()\n'
     '    envs = ["oregonator", "euler", "ball3d"]\n'
     '    m1_rmse = [C[e][0][1] for e in envs]\n'
     '    m2_rmse = [next(p[1] for p in C[e] if abs(p[2] - 0.75) < 1e-6) for e in envs]\n'
     '    speedups_m2 = [next(p[0] for p in C[e] if abs(p[2] - 0.75) < 1e-6) for e in envs]\n',
     '    envs = ["oregonator", "euler", "ball3d"]\n'
     '    Q = {e: json.loads((ROOT / "ablations" / "quadrant" / "results" / f"quadrant_{e}.json")\n'
     '                       .read_text())["cells"]["test"]["64"] for e in envs}\n'
     '    m1_rmse = [Q[e]["m1_rmse"] for e in envs]\n'
     '    m2_rmse = [Q[e]["m2_gated_rmse"] for e in envs]\n'
     '    # Effective-speedup labels kept exactly as printed in the submitted Fig. 6\n'
     '    # (timing-derived, not from this RMSE sample).\n'
     '    speedups_m2 = [3.8, 3.4, 2.9]\n'),
    ('    out_pdf = HERE / "fig6_mode2.pdf"\n    out_png = HERE / "fig6_mode2.png"\n',
     '    out_pdf = OUT / "fig6_mode2.pdf"\n    out_png = OUT / "fig6_mode2.png"\n'),
])

# Task 8 -- Fig 8: one colour per lambda setting; legend moved clear of the bars.
patch("appendix_figures/build_appendix_dagger.py", "build_appendix_dagger.py", [
    ('HERE = Path(__file__).parent\n', 'HERE = Path(__file__).parent\n' + OUT_DECL),
    ('                    color=ENV_COLORS, edgecolor="black", linewidth=0.9)',
     '                    color=ENV_COLORS[0], edgecolor="black", linewidth=0.9)'),
    ('    ax.legend(loc="upper left", fontsize=9, frameon=False)\n',
     '    ax.set_ylim(top=1.6)  # headroom so the legend clears the tallest bar label\n'
     '    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)\n'),
    ('    out_pdf = HERE / "appendix_dagger.pdf"', '    out_pdf = OUT / "appendix_dagger.pdf"'),
    ('plt.savefig(HERE / "appendix_dagger.png"', 'plt.savefig(OUT / "appendix_dagger.png"'),
])

# Task 9 -- Fig 1 panel (a) from the per-horizon measured sweep (panel (b) code untouched;
# its pixels are restored from the submitted PNG by splice_fig1.py).
patch("build_fig1_v2.py", "build_fig1_v2.py", [
    ('ROOT = Path(__file__).parent.parent\n', 'ROOT = Path(__file__).resolve().parents[2]\n'),
    ('OUT = Path(__file__).parent\n', 'OUT = Path(__file__).resolve().parents[1] / "fig1_unspliced"\nOUT.mkdir(exist_ok=True)\n'),
    ('    for env in ["oregonator", "euler"]:\n'
     '        d = cpu[env]\n'
     '        sur_ms = d["surrogate_batch_ms"]["1"]\n'
     '        per_step = d["solver_ms_per_traj"] / 64.0\n'
     '\n'
     '        m1_speedups = []\n'
     '        m2_speedups = []\n'
     '        for h in HORIZONS:\n'
     '            solver_h = per_step * h\n'
     '            m1_speedups.append(solver_h / sur_ms)\n'
     '            m2_speedups.append(m2_speedup(solver_h, sur_ms, q_keep=Q_DEFAULT))\n',
     '    sweep = json.load(open(RESULTS / "cpu_horizon_sweep.json"))  # measured at each h\n'
     '    for env in ["oregonator", "euler"]:\n'
     '        m1_speedups = []\n'
     '        m2_speedups = []\n'
     '        for h in HORIZONS:\n'
     '            solver_h = sweep[env][str(h)]["solver_ms"]\n'
     '            sur_h = sweep[env][str(h)]["surrogate_ms"]\n'
     '            m1_speedups.append(solver_h / sur_h)\n'
     '            m2_speedups.append(m2_speedup(solver_h, sur_h, q_keep=Q_DEFAULT))\n'),
])

# Task 13 -- Fig 14: only the environment named in the panel title is corrected.
patch("appendix_figures/build_appendix_richardson.py", "build_appendix_richardson.py", [
    ('HERE = Path(__file__).parent\n', 'HERE = Path(__file__).parent\n' + OUT_DECL),
    ('    ax.set_title("AUROC (Euler 2D, test split)"', '    ax.set_title("AUROC (Oregonator, test split)"'),
    ('    out_pdf = HERE / "appendix_richardson.pdf"', '    out_pdf = OUT / "appendix_richardson.pdf"'),
    ('plt.savefig(HERE / "appendix_richardson.png"', 'plt.savefig(OUT / "appendix_richardson.png"'),
])
