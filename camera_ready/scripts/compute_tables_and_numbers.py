#!/usr/bin/env python3
"""Camera-ready tables (5, 6, Mode-1 RMSE) and numbers.md, read-only from existing results.

No simulation, training, solver rollout or evaluation is run. Every value is read
from the result files named next to it in numbers.md.
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]          # ~/Shortcut_actions
OUT = Path(__file__).resolve().parents[1]           # camera_ready_out/
Q = ROOT / "ablations" / "quadrant" / "results"
FH = ROOT / "ablations" / "mode2_qsweep" / "results" / "full_horizons"
ENVS = ["oregonator", "euler", "ball3d"]
ENV_LABEL = {"oregonator": "Oregonator", "euler": "Euler 2D", "ball3d": "Ball 3D"}
SPLITS = ["test", "ood_near", "ood_far"]
SPLIT_LABEL = {"test": "test", "ood_near": "OOD-near", "ood_far": "OOD-far"}
H = [2, 4, 8, 16, 32, 64]
rel = lambda p: str(Path(p).relative_to(ROOT))
mism = []  # mismatches against expected values


def check(name, got, exp, tol):
    ok = abs(got - exp) <= tol
    if not ok:
        mism.append(f"{name}: computed {got} vs expected {exp}")
    return "matches expected" if ok else f"**DIFFERS from expected {exp}**"


HEADER = (
    "  \\begin{tabular}{c | ccc | ccc | ccc}\n  \\toprule\n"
    "   & \\multicolumn{3}{c|}{Oregonator} & \\multicolumn{3}{c|}{Euler 2D} & \\multicolumn{3}{c}{Ball 3D} \\\\\n"
    "  $h$ & test & OOD-n & OOD-f & test & OOD-n & OOD-f & test & OOD-n & OOD-f \\\\\n  \\midrule\n")
FOOTER = "  \\bottomrule\n  \\end{tabular}\n"

# ------------------------------------------------------------------ Table 5 (a)
cs = {e: json.loads((Q / f"cross_seed_indep_{e}.json").read_text()) for e in ENVS}
t5 = {}  # (env, split, h) -> (mean, sd, [seed values])
for e in ENVS:
    seeds = cs[e]["seeds"]
    for sp in SPLITS:
        for h in H:
            v = [seeds[s][sp][str(h)] for s in sorted(seeds)]
            t5[(e, sp, h)] = (float(np.mean(v)), float(np.std(v)), v)   # population SD (ddof=0)
rows = []
for h in H:
    cells = [f"${t5[(e, sp, h)][0]:.2f}{{\\pm}}{f'{t5[(e, sp, h)][1]:.2f}'[1:]}$" for e in ENVS for sp in SPLITS]
    rows.append(f"  {h:2d} & " + " & ".join(cells) + " \\\\\n")
(OUT / "table5.tex").write_text(HEADER + "".join(rows) + FOOTER)
means = [m for (m, s, v) in t5.values()]
sds = {k: s for k, (m, s, v) in t5.items()}
kmax = max(sds, key=sds.get)
t5_median, t5_median_sd, t5_max_sd = float(np.median(means)), float(np.median(list(sds.values()))), sds[kmax]
euler_test = [round(t5[("euler", "test", h)][0], 2) for h in H]
b16, b32 = t5[("ball3d", "ood_far", 16)], t5[("ball3d", "ood_far", 32)]

# Oregonator seed-2 truncation (checkpoint metadata, read-only)
s2 = ROOT / "oregonator_pipeline/checkpoints/shortcut_oregonator_v3/seed2"
s2_epochs = sorted(int(p.stem[5:]) for p in s2.glob("epoch*.pt"))
ck2 = torch.load(s2 / "best.pt", map_location="cpu", weights_only=False)
s2_best_ep, s2_best_val = ck2.get("epoch"), ck2.get("val_mse")
s2_hist = (s2 / "history.json").exists()

# ------------------------------------------------------------------ Table 6 (b)
qd = {e: json.loads((Q / f"quadrant_{e}.json").read_text()) for e in ENVS}
cells = {(e, sp, h): qd[e]["cells"][sp][str(h)] for e in ENVS for sp in SPLITS for h in H}
for k, c in cells.items():   # cut from the stored M1/M2 agrees with the stored cut field
    assert abs((c["m1_rmse"] - c["m2_gated_rmse"]) / c["m1_rmse"] * 100 - c["cut_vs_m1_pct"]) < 1e-6, k
    assert abs(c["cut_vs_m1_pct"] - 25.0 - c["delta_above_random_pp"]) < 1e-6, k
t6 = ["  \\begin{tabular}{l l | r r r r r r}\n  \\toprule\n",
      "  Env & Split & $h{=}2$ & $h{=}4$ & $h{=}8$ & $h{=}16$ & $h{=}32$ & $h{=}64$ \\\\\n"]
for e in ENVS:
    t6.append(f"  \\midrule\n  \\multirow{{3}}{{*}}{{{ENV_LABEL[e]}}}\n")
    for sp in SPLITS:
        vals = " & ".join(f"$-{cells[(e, sp, h)]['cut_vs_m1_pct']:.0f}\\%$" for h in H)
        t6.append(f"    & {SPLIT_LABEL[sp] + ' ' * (8 - len(SPLIT_LABEL[sp]))} & {vals} \\\\\n")
t6.append(FOOTER)
(OUT / "table6.tex").write_text("".join(t6))
gaps = {k: c["delta_above_random_pp"] for k, c in cells.items()}
cuts = {k: c["cut_vs_m1_pct"] for k, c in cells.items()}
gmin, gmed, gmax = min(gaps.values()), float(np.median(list(gaps.values()))), max(gaps.values())
kgmin, kgmax = min(gaps, key=gaps.get), max(gaps, key=gaps.get)
env_med = {e: float(np.median([g for k, g in gaps.items() if k[0] == e])) for e in ENVS}
near = sorted([k for k, g in gaps.items() if g <= 10.0], key=lambda k: (ENVS.index(k[0]), SPLITS.index(k[1]), k[2]))
kcut = max(cuts, key=cuts.get)
oreg_cuts = [c for k, c in cuts.items() if k[0] == "oregonator"]
kcmin = min((k for k in cuts if k[0] == "oregonator"), key=cuts.get)
kcmaxo = max((k for k in cuts if k[0] == "oregonator"), key=cuts.get)
fig6 = {e: cells[(e, "test", 64)] for e in ENVS}
fc = {e: [cells[(e, sp, h)]["frac_false_confidence"] * 100 for sp in SPLITS for h in H] for e in ENVS}
fc_med = float(np.median(sum(fc.values(), [])))

# ------------------------------------------------------------------ Mode-1 RMSE table (c)
fh = {(e, sp): json.loads((FH / f"{e}_{sp}.json").read_text()) for e in ENVS for sp in SPLITS}
assert all(fh[(e, sp)][str(h)]["n_pairs"] == 80 for e in ENVS for sp in SPLITS for h in H)
rm = {(e, sp, h): fh[(e, sp)][str(h)]["m1_rmse_mean"] for e in ENVS for sp in SPLITS for h in H}
r_rows = [f"  {h:2d} & " + " & ".join(f"{rm[(e, sp, h)]:.3f}" for e in ENVS for sp in SPLITS) + " \\\\\n" for h in H]
(OUT / "rmse_table.tex").write_text(
    "\\begin{table}[!htb]\n  \\centering\n"
    "  \\caption{Mode~1 trajectory-mean test RMSE per horizon and split, $80$ trajectories per cell.}\n"
    "  \\label{tab:m1_rmse}\n  \\small\n  \\setlength{\\tabcolsep}{4pt}\n"
    + HEADER + "".join(r_rows) + FOOTER + "\\end{table}\n")
EXP_RMSE = {"oregonator": [(.007, .022), (.011, .034), (.018, .058), (.046, .097), (.048, .121), (.062, .131)],
            "euler": [(.127, .252), (.194, .507), (.115, 1.117), (.193, 1.124), (.237, .513), (.282, 1.053)],
            "ball3d": [(.035, .041), (.049, .073), (.045, .112), (.100, .218), (.123, .364), (.277, .368)]}
rmse_ok = 0
for e in ENVS:
    for i, h in enumerate(H):
        for sp, exp in zip(("test", "ood_far"), EXP_RMSE[e][i]):
            got = round(rm[(e, sp, h)], 3)
            if abs(got - exp) < 1e-9:
                rmse_ok += 1
            else:
                mism.append(f"RMSE {e} {sp} h={h}: computed {got:.3f} vs expected {exp:.3f}")

# ------------------------------------------------------------------ DAgger chaining (f)
ch = json.loads((Q / "chaining_dagger.json").read_text())
pct = {e: {k: (ch[e]["lambda0"][k] - ch[e]["lambda0.1"][k]) / ch[e]["lambda0.1"][k] * 100
           for k in ("1", "2", "4", "8")} for e in ("euler", "ball3d")}


def hist_info(p):
    h = json.loads(Path(p).read_text())
    v = [(r["epoch"], r["val_mse"]) for r in h if r.get("val_mse") is not None]
    b = min(v, key=lambda x: x[1])
    return len(v), b[0], b[1]


RUNS = {
    ("oregonator", "0"): "oregonator_pipeline/checkpoints/dagger_ablation/sup_only/seed0/history.json",
    ("oregonator", "1"): "oregonator_pipeline/checkpoints/dagger_ablation/dagger_only/seed0/history.json",
    ("oregonator", "0.1"): "oregonator_pipeline/checkpoints/shortcut_oregonator_v3/seed0/history.json",
    ("euler", "0"): "neurips/training/checkpoints/dagger_ablation/sup_only_euler/seed0/history.json",
    ("euler", "1"): "neurips/training/checkpoints/dagger_ablation/dagger_only_euler/seed0/history.json",
    ("euler", "0.1"): "neurips/training/checkpoints/shortcut_euler2d_v2_dagger/seed0/history.json",
    ("ball3d", "0"): "mujoco_pipeline/checkpoints/dagger_ablation_ball3d/sup_only/seed0/history.json",
    ("ball3d", "1"): "mujoco_pipeline/checkpoints/dagger_ablation_ball3d/dagger_only/seed0/history.json",
    ("ball3d", "0.1"): "mujoco_pipeline/checkpoints/shortcut_ball3d/seed0/history.json",
}
HI = {k: hist_info(ROOT / p) for k, p in RUNS.items()}
ep = lambda lam: ", ".join(str(HI[(e, lam)][0]) for e in ENVS[:2]) + f" and {HI[('ball3d', lam)][0]}"
assert HI[("oregonator", "0")][0] == HI[("oregonator", "1")][0] and HI[("euler", "0")][0] == HI[("euler", "1")][0] \
    and HI[("ball3d", "0")][0] == HI[("ball3d", "1")][0]
lam0_ckpt = torch.load(ROOT / "neurips/training/checkpoints/dagger_ablation/sup_only_euler/seed0/best.pt",
                       map_location="cpu", weights_only=False)

# ------------------------------------------------------------------ numbers.md
cellname = lambda k: f"{ENV_LABEL[k[0]]} {SPLIT_LABEL[k[1]]} h={k[2]}"
L = []
w = L.append
w("# Camera-ready numbers\n")
w("All values are read from existing result files; nothing was simulated, trained, re-timed or re-evaluated. "
  "Paths are relative to `~/Shortcut_actions/`. Outputs are in `camera_ready_out/`; the scripts that produced "
  "them are in `camera_ready_out/scripts/`.\n")
w("## Mismatches against the expected (as-posted) values\n")
w("None. Every expected value in the brief reproduces to the printed precision.\n" if not mism else
  "\n".join(f"- {m}" for m in mism) + "\n")

w("## Task 1: Table 5 (cross-seed AUROC)\n")
w("Source: `ablations/quadrant/results/cross_seed_indep_{oregonator,euler,ball3d}.json` (`seeds[s][split][h]`), "
  "written by `ablations/quadrant/eval_cross_seed_indep.py`. 150 / 150 / 1000 (trajectory, t0) pairs per cell; "
  "each seed scored on its own draw; per-trajectory AUROC at the 75th-percentile true-error threshold.\n")
w(f"- **T5_TABLE**: `camera_ready_out/table5.tex` (drop-in replacement for the `tabular` of `tab:cross_seed_auroc`).\n")
w(f"- **T5_MEDIAN_AUROC**: {t5_median:.3f} (median of the 54 seed-means; {check('T5 median', round(t5_median, 3), 0.906, 1e-9)}).\n")
w(f"- **T5_MEDIAN_SD**: {t5_median_sd:.3f}\n")
w(f"- **T5_MAX_SD**: {t5_max_sd:.3f}\n")
w(f"- **T5_MAX_SD_CELL**: {cellname(kmax)} (mean {t5[kmax][0]:.3f}, seeds {', '.join(f'{x:.3f}' for x in t5[kmax][2])}).\n")
w(f"- Expected-value checks: Euler test column {', '.join(f'{x:.2f}' for x in euler_test)} "
  f"({'matches expected' if euler_test == [0.96, 0.99, 0.96, 0.97, 0.95, 0.91] else '**DIFFERS**'}); "
  f"Ball 3D OOD-far h=16 {b16[0]:.2f} ± {b16[1]:.2f}, h=32 {b32[0]:.2f} ± {b32[1]:.2f} "
  f"({'matches expected' if (round(b16[0],2),round(b16[1],2),round(b32[0],2),round(b32[1],2)) == (0.51,0.02,0.49,0.03) else '**DIFFERS**'}).\n")
w(f"- **SD convention**: population SD (`np.std`, ddof=0), as instructed. For reference, sample SD (ddof=1) would print "
  f"Ball 3D OOD-far h=32 as ±{np.std(b32[2], ddof=1):.2f} instead of ±{b32[1]:.2f}.\n")
w(f"- **Oregonator seed 2 is a truncated run**: `oregonator_pipeline/checkpoints/shortcut_oregonator_v3/seed2/` has "
  f"epoch checkpoints up to epoch {max(s2_epochs)} of the configured 60, history.json present: {s2_hist}; `best.pt` is "
  f"epoch {s2_best_ep} with val MSE {s2_best_val:.3f} (seeds 0 and 1 ran 60 epochs, best val 0.104 and 0.109). "
  "The Oregonator columns of Table 5 and Fig. 9 include this seed.\n")

w("## Task 2: Fig. 9\n")
w("`camera_ready_out/appendix_cross_seed.png` (+ `.pdf`), same layout, size (2234×639 px, 150 dpi) and style as the "
  "submitted figure; only the data source changed (seed-mean ± population SD from the files above). "
  "Script: `scripts/build_appendix_cross_seed.py`.\n")

w("## Task 3: Table 6 (Mode 2 vs Mode 1 at q=0.75)\n")
w("Source: `ablations/quadrant/results/quadrant_{env}.json` → `cells[split][h]` (`m1_rmse`, `m2_gated_rmse`, "
  "`cut_vs_m1_pct`, `delta_above_random_pp`), written by `ablations/quadrant/eval_quadrant.py`; 200 / 200 / 1000 "
  "(trajectory, t0) pairs per cell, one shared draw. Gap above random = cut − 25 pp (checked for every cell).\n")
w("- **T6_TABLE**: `camera_ready_out/table6.tex` (drop-in `tabular` for `tab:m1_m2_horizons`; cuts rounded to integers).\n")
w(f"- **GAP_MIN**: +{gmin:.1f} pp ({cellname(kgmin)}; {check('GAP_MIN', round(gmin, 1), 2.9, 1e-9)})\n")
w(f"- **GAP_MEDIAN**: +{gmed:.1f} pp ({check('GAP_MEDIAN', round(gmed, 1), 28.5, 1e-9)})\n")
w(f"- **GAP_MAX**: +{gmax:.1f} pp ({cellname(kgmax)}; {check('GAP_MAX', round(gmax, 1), 54.0, 1e-9)})\n")
w(f"- **PER_ENV_MEDIANS**: Oregonator +{env_med['oregonator']:.1f} pp, Euler 2D +{env_med['euler']:.1f} pp, "
  f"Ball 3D +{env_med['ball3d']:.1f} pp (median over each environment's 18 cells).\n")
w(f"- **NEAR_FLOOR_CELLS** (gap ≤ 10 pp): {len(near)} cells: " +
  "; ".join(f"{cellname(k)} (−{cuts[k]:.1f}%, +{gaps[k]:.1f} pp)" for k in near) + ".\n")
w(f"- **LARGEST_CUT**: {cellname(kcut)}, −{cuts[kcut]:.1f}% (M1 {cells[kcut]['m1_rmse']:.3f} → M2 "
  f"{cells[kcut]['m2_gated_rmse']:.3f}).\n")
w(f"- **OREG_RANGE**: {min(oreg_cuts):.1f}% to {max(oreg_cuts):.1f}% ({cellname(kcmin)} to {cellname(kcmaxo)}); "
  f"rounded {min(oreg_cuts):.0f}–{max(oreg_cuts):.0f}%.\n")
w(f"- Every one of the 54 cells shows a reduction and exceeds the 25% random floor: {all(g > 0 for g in gaps.values())}.\n")

w("## Task 4: Fig. 10\n")
w("`camera_ready_out/appendix_m1_vs_m2.png` (+ `.pdf`), same layout and size (2232×1481 px, 150 dpi); only the data "
  "source changed (Mode 1 = `m1_rmse`, Mode 2 = `m2_gated_rmse` from the quadrant files). Script: "
  "`scripts/build_appendix_m1_vs_m2.py`. Inherited from the submitted figure and left unchanged: the h=2 reduction "
  "labels sit on the bottom axis line and are partly clipped (the original has the same overlap). A one-line fix, if "
  "wanted, is `ax.margins(y=0.12)` in the panel loop.\n")

w("## Task 5: Fig. 6\n")
w(f"`camera_ready_out/fig6_mode2.png` (+ `.pdf`), same size (1479×895 px, 200 dpi). RMSE bars from the quadrant files, "
  f"test split, h=64: Oregonator {fig6['oregonator']['m1_rmse']:.3f} → {fig6['oregonator']['m2_gated_rmse']:.3f}, "
  f"Euler 2D {fig6['euler']['m1_rmse']:.3f} → {fig6['euler']['m2_gated_rmse']:.3f}, "
  f"Ball 3D {fig6['ball3d']['m1_rmse']:.3f} → {fig6['ball3d']['m2_gated_rmse']:.3f}. "
  "The effective-speedup labels are hard-coded to the values printed in the submitted figure (3.8×, 3.4×, 2.9×), "
  "including the original Ball 3D label. Note: the current `cpu_horizon_sweep.json` would give Euler 3.6×, not 3.4×, "
  "with the original script's formula; the labels were kept as submitted per the brief.\n")
f6 = [fig6[e]["cut_vs_m1_pct"] for e in ENVS]
w(f"- **FIG6_RANGE**: {min(f6):.1f}% to {max(f6):.1f}% (Oregonator −{f6[0]:.1f}%, Euler 2D −{f6[1]:.1f}%, "
  f"Ball 3D −{f6[2]:.1f}%). The main text (Section 5.5) currently says \"43%–52%\".\n")

w("## Task 6: Mode 1 RMSE table\n")
w("Source: `ablations/mode2_qsweep/results/full_horizons/{env}_{split}.json` → `[h].m1_rmse_mean` (`n_pairs` = 80 in "
  "every cell, checked), written by `ablations/mode2_qsweep/eval_qsweep_full_horizons.py`.\n")
w(f"- **RMSE_TABLE**: `camera_ready_out/rmse_table.tex` (full `table` environment, label `tab:m1_rmse`, caption as given). "
  f"Expected test / OOD-far values: {rmse_ok} of 36 match.\n")
w("- The given caption says \"test RMSE\" while the table also has OOD-near and OOD-far columns; kept verbatim as instructed.\n")

w("## Task 7: False-confidence quadrant\n")
w("Source: quadrant files, `frac_false_confidence` = share of all pairs in the cell that the gate keeps at q=0.75 and "
  "that are in the top true-error quartile (definition checked in `eval_quadrant.py`, `quadrant_and_mode2`).\n")
for e, exp in zip(ENVS, [(3.5, 13.0), (0.0, 11.5), (4.0, 17.6)]):
    lo, hi = min(fc[e]), max(fc[e])
    ok = (round(lo, 1), round(hi, 1)) == exp
    if not ok:
        mism.append(f"false-confidence {e}: {lo:.1f}-{hi:.1f} vs expected {exp}")
    w(f"- **FALSECONF_{e.upper()}**: {lo:.1f}% to {hi:.1f}% ({'matches expected' if ok else '**DIFFERS**'})\n")
w(f"- **FALSECONF_MEDIAN**: {fc_med:.1f}% over all 54 cells ({check('FALSECONF_MEDIAN', round(fc_med, 1), 6.5, 1e-9)})\n")

w("## Task 8: Fig. 8\n")
w("`camera_ready_out/appendix_dagger.png` (+ `.pdf`), same data (hard-coded best validation MSE values, traceable to "
  "the histories under Task 10), same size (1259×643 px, 150 dpi). Changes: the λ=0.1 bars now use one colour (teal "
  "#1b8a8a, the swatch the legend already showed) on every environment, so the legend is correct; the legend is a single "
  "row at the top and the y-axis top is raised to 1.6 so it clears every bar and value label. Value labels kept. "
  "Script: `scripts/build_appendix_dagger.py`.\n")

w("## Task 9: Fig. 1\n")
w("- **FIG1_STATUS**: separate runs. The submitted panel (a) scaled one h=64 timing linearly in h "
  "(`ablations/batch_bench/results/cpu_h64.json`, May 3, `bench_cpu.py`: 70.2× / 24.8×); Fig. 5(a) uses the per-horizon "
  "measured sweep (`cpu_horizon_sweep.json`, May 5, `bench_cpu_horizon_sweep_clean.py`: 72.0× / 25.6×). As instructed, "
  "panel (a) was regenerated from `cpu_horizon_sweep.json`: Mode 1 = solver_ms / surrogate_ms and Mode 2 (dashed) = "
  "solver / (3·surrogate + 0.25·solver), both now from the times measured at each h. It reads **72×** and **26×**. "
  "Panel (b) is pixel-identical to the submitted figure: the q-sweep JSONs panel (b) reads were rewritten (May 5 18:40) "
  "after the submitted PNG was made (15:18), so re-rendering would have changed it; `scripts/splice_fig1.py` joins the new "
  "panel (a) to the submitted panel (b) at a 32-column white gap (columns 1323-1355; split at x=1339 of 2676). Panel (b)'s unlabelled Mode 1 "
  "endpoints still sit at the `cpu_h64.json` values (70.2× / 24.8×). Only the PNG is produced (the paper uses the PNG); "
  "`camera_ready_out/fig1_unspliced/` is an intermediate render with a re-rendered panel (b) and should not be used. "
  "The Fig. 1 caption's \"25× and 70×\" needs to become 26× and 72×.\n")

w("## Task 10: DAgger chained measurement\n")
w("Source: `ablations/quadrant/results/chaining_dagger.json`, written by `ablations/quadrant/eval_chaining.py`.\n")
w("- **DAGGER_ENV**: Euler 2D. Relative change of λ=0 vs λ=0.1: " +
  ", ".join(f"k={k}: {pct['euler'][k]:+.1f}%" for k in ("1", "2", "4", "8")) +
  ". Ball 3D: " + ", ".join(f"k={k}: {pct['ball3d'][k]:+.1f}%" for k in ("1", "2", "4", "8")) + ".\n")
w("- **DAGGER_METRIC**: RMSE in physical (de-normalised) state units, Euler: mean over 100 (trajectory, t0) pairs from "
  "the **test** split (`RandomState(7)`) of the per-pair RMSE over all cells and the four conserved-variable channels; "
  "Ball 3D: one RMSE over 500 test pairs × 9 state dims. The total horizon is fixed at 64 base steps and reached with k "
  "chained calls of 64/k each, so k=1 is a single-shot prediction at h=64. This is a **different metric** from Fig. 8, "
  "which plots the best per-epoch **validation** MSE in **normalised** units, averaged over horizons drawn uniformly from "
  "{1,…,64}, read from the training histories. Both use the same seed-0 checkpoints.\n")
w(f"- **DAGGER_EPOCHS_ABLATION**: {ep('0')} (λ=0 and λ=1 runs, Oregonator / Euler 2D / Ball 3D; identical for both λ).\n")
w(f"- **DAGGER_EPOCHS_DEFAULT**: {ep('0.1')} (λ=0.1 main runs). The Oregonator default run was resumed from the "
  "epoch-9 checkpoint of an earlier run (`oregonator_pipeline/logs/train_v3_part1.log`; `best_resumed_from_e9.pt`) and "
  "then trained 60 epochs.\n")
w("- Best epoch / best val MSE per run: " + "; ".join(
  f"{ENV_LABEL[e]} λ={lam}: ep {HI[(e, lam)][1]} / {HI[(e, lam)][2]:.4f}" for e in ENVS for lam in ("0", "0.1", "1")) + ".\n")
w(f"- **DAGGER_LAMBDA0_CKPT**: `neurips/training/checkpoints/dagger_ablation/sup_only_euler/seed0/best.pt` (epoch "
  f"{lam0_ckpt.get('epoch')} of the 30-epoch λ=0 run, `dagger_weight` {lam0_ckpt.get('dagger_weight')}, val MSE "
  f"{lam0_ckpt.get('val_mse'):.4f}), i.e. the same checkpoint as the Fig. 8 λ=0 Euler bar; compared against the 40-epoch "
  "λ=0.1 main checkpoint. Ball 3D λ=0: `mujoco_pipeline/checkpoints/dagger_ablation_ball3d/sup_only/seed0/best.pt`.\n")

w("## Task 11: Fig. 11 sample\n")
w("- **QSWEEP_SAMPLE**: 80 (trajectory, t0) pairs per (environment, split) cell at h=64, one model seed (seed 0), "
  "Oregonator and Euler 2D only. Source: `ablations/mode2_qsweep/results/summary.json` (May 2), whose `method` field "
  "states the 80 pairs; the evaluator is `ablations/mode2_qsweep/eval_qsweep.py` (`N_PAIRS = 80`, `RandomState(7)`, "
  "seed-0 checkpoints). The per-environment JSONs that summary.json was built from were overwritten on May 5, and no "
  "script in the repo writes summary.json. This sample differs from Table 6 / Fig. 6 / Fig. 10 (quadrant files, 200 / "
  "200 / 1000 pairs) and from the May 5 `full_horizons/` run.\n")

w("## Task 12: Mode 2 threshold source\n")
w("- **THRESHOLD_SOURCE**: evaluated-set quantile in all five; none uses a validation set.\n"
  "  - Fig. 1(b): `mode2_qsweep/results/{env}_test.json` from `eval_qsweep.py`, `mode2_at_q`: "
  "`thr = np.quantile(sd, q)` over the same 80 evaluated test pairs (the submitted panel (b) came from an earlier "
  "output of the same script).\n"
  "  - Fig. 6, Fig. 10, Table 6 (this camera-ready version, from the quadrant files): `eval_quadrant.py`, "
  "`quadrant_and_mode2`: `gate_thr = np.quantile(eh, q_keep)` over the evaluated sample (so the deferral rate is "
  "exactly 25%). The submitted versions (`full_horizons/`, `summary.json`) used the same evaluated-set rule.\n"
  "  - Fig. 11: `summary.json` from `eval_qsweep.py`, same `mode2_at_q` rule.\n")

if mism:
    L.insert(3, "\n".join(f"- {m}" for m in mism) + "\n")
(OUT / "numbers.md").write_text("\n".join(L))
print("mismatches:", mism or "none")
print(open(OUT / "table5.tex").read()); print(open(OUT / "table6.tex").read())
