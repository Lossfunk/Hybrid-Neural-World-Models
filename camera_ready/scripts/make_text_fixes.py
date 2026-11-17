#!/usr/bin/env python3
"""Write camera_ready_out/text_fixes.md.

Every find string is verified to occur exactly once in the paper source; spans
(start ... end) are verified to be ordered. Corrected numbers are recomputed from
the result files and asserted before they are written.

Paper source checked: the May 27 preprint .tex, whose body is identical to the
submitted source; submitted-source line = preprint line + 22 (offset verified at
\\makeatletter, \\maketitle and \\begin{abstract}). checklist.tex lines are as in
that file.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
PAPER = Path.home() / "Documents/Shortcut_Lossfunk/NeurIPS_submission/Preprint_Lossfunk_of_NeurIPS"
TEX = (PAPER / "neurips_2026.tex").read_text()
CHK = (PAPER / "checklist.tex").read_text()
OFFSET = 22
J = lambda p: json.loads((ROOT / p).read_text())

# ---------------------------------------------------------------- recomputed values
em = J("ablations/baselines/results/energy_momentum_ball3d.json")["test"]
sd32 = em["32"]["auroc_step_doubling"]; assert round(sd32, 2) == 0.82
marg = [em[h]["auroc_energy_residual"] - em[h]["auroc_step_doubling"] for h in ("8", "16")]
assert all(0.025 <= m <= 0.0405 for m in marg), marg
mom = [em[h]["auroc_momentum_residual"] for h in ("2", "4", "8", "16", "32", "64")]
assert sum(m <= 0.515 for m in mom) == 5 and round(mom[-1], 2) == 0.67


def best(p):
    h = J(p)
    return min(r["val_mse"] for r in h if r.get("val_mse") is not None)


B = {(e, l): best(p) for (e, l), p in {
    ("o", 0): "oregonator_pipeline/checkpoints/dagger_ablation/sup_only/seed0/history.json",
    ("o", 1): "oregonator_pipeline/checkpoints/dagger_ablation/dagger_only/seed0/history.json",
    ("o", .1): "oregonator_pipeline/checkpoints/shortcut_oregonator_v3/seed0/history.json",
    ("e", 0): "neurips/training/checkpoints/dagger_ablation/sup_only_euler/seed0/history.json",
    ("e", 1): "neurips/training/checkpoints/dagger_ablation/dagger_only_euler/seed0/history.json",
    ("e", .1): "neurips/training/checkpoints/shortcut_euler2d_v2_dagger/seed0/history.json",
    ("b", 0): "mujoco_pipeline/checkpoints/dagger_ablation_ball3d/sup_only/seed0/history.json",
    ("b", 1): "mujoco_pipeline/checkpoints/dagger_ablation_ball3d/dagger_only/seed0/history.json",
    ("b", .1): "mujoco_pipeline/checkpoints/shortcut_ball3d/seed0/history.json"}.items()}
r0 = {e: (B[(e, 0)] / B[(e, .1)] - 1) * 100 for e in "oeb"}
r1 = [B[(e, 1)] / B[(e, .1)] for e in "oeb"]
assert round(r0["e"]) == 26 and round(r0["b"]) == 6 and round(r0["o"]) == 26
assert round(min(r1), 1) == 3.6 and round(max(r1), 1) == 9.5

sc = {e: J(f"ablations/self_consistency/checkpoints/self_consistency_{e}/seed0/history.json") for e in ("oreg", "euler")}
orders = {e: np.log10(sc[e][0]["train_sc_loss"] / min(r["train_sc_loss"] for r in sc[e])) for e in sc}
ep5 = {e: next(r["train_sc_loss"] for r in sc[e] if r["epoch"] == 5) for e in sc}
valrng = {e: (min(r["val_mse_vs_GT"] for r in sc[e]), max(r["val_mse_vs_GT"] for r in sc[e])) for e in sc}
assert all(3.0 <= o <= 3.6 for o in orders.values()), orders

bt = J("ablations/beyond_tmax/results/summary.json")
o128 = [bt["oregonator"][s][h] for s in ("test", "ood_near", "ood_far") for h in ("h64", "h80", "h96", "h128")]
e_ext = [bt["euler"][s][h] for s in ("test", "ood_near", "ood_far") for h in ("h80", "h96")]
assert (min(o128), max(o128)) == (0.84, 0.97) and (min(e_ext), max(e_ext)) == (0.99, 1.0)

ro = J("ablations/rollout/results/rollout_h64.json")["euler"]
ro_sd = {k: ro[k]["std_rmse"] for k in ("1", "2", "4", "8")}
assert ro_sd["1"] > 0.5 and ro_sd["2"] > 0.5 and ro_sd["4"] < 0.2 and ro_sd["8"] < 0.2, ro_sd

eh = J("ablations/batch_bench/results/euler_h64.json")["speedup_by_batch"]
oh = J("ablations/batch_bench/results/oregonator_h64.json")["speedup_by_batch"]
assert round(eh["1"]) == 123 and round(eh["32"]) == 186 and round(oh["1"]) == 734

ncpu = os.cpu_count(); assert ncpu == 32
osname = next((l.split("=", 1)[1].strip().strip('"') for l in open("/etc/os-release") if l.startswith("PRETTY_NAME=")), "?")

t3 = {"head": [.79, .72, .73, .86, .89, .85, .65, .77, .77], "ens": [.75, .73, .71, .81, .82, .83, .74, .81, .70]}
assert round(np.mean(t3["head"]), 2) == 0.78 and round(np.mean(t3["ens"]), 2) == 0.77

t5 = {e: J(f"ablations/quadrant/results/cross_seed_indep_{e}.json")["seeds"] for e in ("oregonator",)}
oreg_t5 = [np.mean([t5["oregonator"][s][sp][h] for s in "012"]) for sp in ("test", "ood_near", "ood_far")
           for h in ("2", "4", "8", "16", "32", "64")]
assert (round(min(oreg_t5), 2), round(max(oreg_t5), 2)) == (0.81, 0.95)
qd = J("ablations/quadrant/results/quadrant_oregonator.json")["cells"]
oreg_cut = [qd[sp][h]["cut_vs_m1_pct"] for sp in ("test", "ood_near", "ood_far") for h in ("2", "4", "8", "16", "32", "64")]
assert (round(min(oreg_cut)), round(max(oreg_cut))) == (44, 78)

rich = J("ablations/richardson_auroc/results/richardson_auroc_test_only.json")["data"]["test"]
rv = [rich[h][m]["auroc"] for h in ("4", "8", "16", "32", "64") for m in ("richardson_prod", "richardson_fixed")]
assert (round(min(rv), 2), round(max(rv), 2)) == (0.38, 0.56)
assert max(max(rich[h]["fixed_max"] for h in rich), max(rich[h]["prod_max"] for h in rich if h not in ("8", "32"))) < 1e-7
assert min(min(rich[h]["prod_min"], rich[h]["fixed_min"]) for h in rich) > 1e-11
assert round(rich["8"]["prod_max"], 4) == 0.0004 and round(rich["32"]["prod_max"], 3) == 0.003

# ---------------------------------------------------------------- fix list
# (group, id, title, file, start, end_or_None, replacement, what was run / source)
F = []
add = lambda *a: F.append(a)

G1 = "A. Richardson appendix (app:richardson), Fig. 14 and the main-text mention"
add(G1, "A1", "Main text, Section 5.2 (evidence paragraph)", "tex",
    r"and classical Richardson extrapolation (Appendix~\ref{app:richardson}), which directly estimates prediction-difficulty under truncation-order assumptions,",
    r"fails at $h \ge 4$ where step-doubling holds at AUROC $0.81$--$0.97$.",
    r"and a solver-side two-leg comparison on Oregonator, our closest test of classical Richardson extrapolation (Appendix~\ref{app:richardson}), is at chance for $h \ge 4$ where step-doubling holds at AUROC $0.81$--$0.97$ on the same pairs.",
    "`ablations/richardson_auroc/run.py` runs the Oregonator Tyson solver, not Euler/HLL (log: `results/run.log`).")
add(G1, "A2", "Appendix intro", "tex", r"to head on Euler 2D.", None,
    r"to head on the Oregonator test split, using an earlier Oregonator surrogate checkpoint (v1) than the one used elsewhere in the paper.",
    "`run.py:51` `CKPT = .../shortcut_oregonator_v1/seed0/best.pt`; `run.log` line 1 confirms v1. The step-doubling scores and true errors it compares against were read from `ablations/auroc_recompute/results/` while those held v1 outputs (Apr 29 log); they were overwritten with v3 on Apr 30.")
add(G1, "A3", "Setup paragraph", "tex",
    r"For each $(s_0, T)$ pair on the test split, we run the reference HLL",
    r"the two legs as the higher-order estimate.",
    r"For each of $50$ $(s_0, T)$ pairs per horizon, we run the reference Oregonator solver from $s_0$ twice: once directly to horizon $T$, and once as two chained legs of $T/2$, with the same solver step in both. The Richardson signal is the magnitude of the difference between the two end states. We then compute AUROC of this signal against the surrogate's true error on the same pairs, exactly as for step-doubling. We report two solver settings: ``richardson-prod'' uses the production adaptive solver ($\Delta t{=}0.05$) and ``richardson-fixed'' a fixed-stencil solver ($\Delta t{=}0.025$). Because both legs execute the same sequence of solver steps, the two end states agree to floating-point precision: differences are $10^{-11}$ to $10^{-7}$, apart from production-solver outliers of up to $4{\times}10^{-4}$ at $h{=}8$ and $3{\times}10^{-3}$ at $h{=}32$.",
    "`run.py:81-105` (`richardson_disagreement`: solver to T vs. solver to T/2 then T/2, same `dt_solver`); `run.py:6-8` and `:170-173` (\"prod\" = production adaptive solver at dt 0.05, \"fixed\" = fixed-stencil at dt 0.025; no step halving, no order correction, no geometric mean). Magnitudes: `prod_min/prod_max/fixed_min/fixed_max` in `richardson_auroc_test_only.json`. The rest of the original line (\"We were able to complete the test split ...\") is unchanged.")
add(G1, "A4", "Result paragraph", "tex",
    r"vs $0.86$) ; exactly the regime where its truncation-error model is most",
    r"underwrites Richardson is no longer informative; step-doubling holds at",
    r"vs $0.86$). At $h{=}4$ and beyond Richardson is at chance ($\mathrm{AUROC}{\in}[0.38, 0.56]$ across $h \in \{4, 8, 16, 32, 64\}$ and both settings). Because the Richardson differences are at floating-point level (Setup), we do not attribute either result to truncation error; step-doubling holds at",
    "No truncation-error model is measured (see A3). Range over both settings at h ≥ 4: fixed 0.38–0.53, prod 0.39–0.56. The next line (\"$\\mathrm{AUROC}{\\in}[0.81, 0.97]$ across the same horizons.\") is unchanged.")
add(G1, "A5", "Cost paragraph", "tex",
    r"Richardson on Euler 2D requires",
    r"two finite-volume rollouts at full and half step size, which scales with the",
    r"Richardson on Oregonator requires three solver rollouts per pair and setting (one to $T$ and two chained legs of $T/2$), which scales with the",
    "`run.py:102-104`. The wall times (78 s … 1864 s, 61 min) are Oregonator-solver times for both settings together (`run.py:192-211`).")
add(G1, "A6", "Cost paragraph, wall-time unit", "tex",
    r"$50$-trajectory cell: $78$ s at $h{=}2$ rising to $1864$ s at $h{=}64$, a", None,
    r"$50$-pair cell (both settings): $78$ s at $h{=}2$ rising to $1864$ s at $h{=}64$, a",
    "`run.py:192-211` times one horizon's loop, which computes both settings for 50 pairs.")
add(G1, "A7", "Fig. 14 caption, title", "tex",
    r"\caption{\textbf{Step-doubling vs classical Richardson on Euler 2D test", None,
    r"\caption{\textbf{Step-doubling vs a solver two-leg (Richardson-style) comparison on the Oregonator test",
    "The figure panel title is corrected to \"AUROC (Oregonator, test split)\" in the regenerated `camera_ready_out/appendix_richardson.png` (only that label changed; 4,564 pixels inside the title box).")
add(G1, "A8", "Fig. 14 caption, body", "tex",
    r"across all horizons; classical Richardson (black) drops to chance for",
    r"error. Right: per-cell wall-clock cost. Richardson scales with solver step",
    r"across all horizons; the solver comparison (black, richardson-fixed) is at chance for $h{\ge}4$. Both use the earlier v1 Oregonator checkpoint. Right: per-cell wall-clock cost. Richardson scales with solver step",
    "The plotted black line is `richardson_fixed` (`build_appendix_richardson.py:21-24`).")

G2 = "B. Closed-loop rollout appendix (app:rollout) and Fig. 13"
add(G2, "B1", "Appendix intro", "tex",
    r"world model: at deployment, the surrogate can be chained $k$ times at",
    r"without the loop blowing up.",
    r"world model: the surrogate can be chained $k$ times at $h{=}64/k$ each to reach a fixed total horizon of $64$ steps without the loop blowing up.",
    "`ablations/rollout/eval_rollout.py:34` `H_TOTAL = 64`; `closed_loop_chain` applies f k times at `target_dt / k_chain`.")
add(G2, "B2", "Setup paragraph", "tex",
    r"For each environment, $32$ trajectories are sampled and rolled out in two",
    r"$64, 128, 256, 512$ steps.",
    r"For each environment, $32$ (trajectory, $t_0$) pairs are drawn from the test split and rolled out in two ways: (i) the reference solver, providing ground truth at $t_0{+}64$, and (ii) the surrogate evaluated at $h{=}64/k$, with its output fed back as the next input, $k$ times. We sweep $k \in \{1, 2, 4, 8\}$, i.e. step sizes $h \in \{64, 32, 16, 8\}$, all reaching the same total horizon of $64$ steps; $k{=}1$ is the single-shot prediction.",
    "`eval_rollout.py:4-9,34-35` (32 pairs, `RandomState(7)`, `*_test.h5`, ground truth at t0+64). Totals of 128–512 steps were not run (trajectories have 201/100/101 frames).")
add(G2, "B3", "What-this-defends paragraph", "tex",
    r"shows the surrogate can be composed at deployment for trajectories several",
    r"times longer than the training horizon if needed, and that",
    r"shows the surrogate can be composed at deployment from $k$ shorter steps without the loop blowing up, and that",
    "Same script; no horizon beyond 64 steps is evaluated.")
add(G2, "B4", "Fig. 13 caption", "tex",
    r"each at $h{=}64$ effective, on three environments.", None,
    r"each at $h{=}64/k$ (total horizon $64$ steps), on three environments.",
    "The figure's own x-axis label also says \"Chained calls $k$ (each at $h{=}64$ effective)\" (`build_appendix_rollout.py:33`). It names no environment, so per the brief it was not regenerated; a one-line change to `\"Chained calls $k$ (each at $h{=}64/k$)\"` fixes it if wanted.")

G3 = "C. The 10% \"DAgger\" term (no solver is queried during training)"
add(G3, "C1", "Contribution 1", "tex", r"and a $10\%$ DAgger refinement.", None,
    r"and a $10\%$ DAgger-style two-step composition term.",
    "Trainers: `oregonator_pipeline/training/train_shortcut_oregonator.py:60-114`, `neurips/training/training/train_shortcut_2d_dagger.py:65-104`, `mujoco_pipeline/training/train_ball3d.py:67-83`. None calls a solver.")
add(G3, "C2", "Method 3.1, after Eq. 1 (the loss used in every main run)", "tex",
    r"where $\tilde{\cdot}$ denotes per-channel normalisation by training statistics.", None,
    r"where $\tilde{\cdot}$ denotes per-channel normalisation by training statistics. All main runs add a $10\%$ two-step composition term to this loss (Section~\ref{sec:5_setup}).",
    "Eq. 1 contains only L_sup; the trained models also used the composition term.")
add(G3, "C3", "Section 5.1, surrogates and training", "tex",
    r"and a $10\%$ DAgger \citep{ross2011dagger} refinement",
    r"against the reference solver. The",
    r"and a $10\%$ DAgger-style composition term \citep{ross2011dagger}: the surrogate's own prediction at $T/2$ is advanced by a second $T/2$ step and supervised against the reference-solver target at $T$; no solver is queried during training (Appendix~\ref{app:training_hp}). The",
    "Composition f(f(s,T/2),T/2) vs dataset target s_T (`train_shortcut_2d_dagger.py:66-101`).")
add(G3, "C4", "Appendix A.4 intro", "tex",
    r"with a $10\%$ DAgger \citep{ross2011dagger} refinement against the reference solver.", None,
    r"with a $10\%$ DAgger-style composition term \citep{ross2011dagger} (Appendix~\ref{app:training_hp}).",
    "As C3.")
add(G3, "C5", "Appendix A.4, meaning of lambda=1", "tex",
    r"($\lambda{=}0.1$), and pure DAgger ($\lambda{=}1$, no supervised loss).", None,
    r"($\lambda{=}0.1$), and pure DAgger ($\lambda{=}1$: composition term only on $h \ge 2$ samples; Oregonator still supervises its $h{=}1$ samples directly, while Euler 2D and Ball 3D give them no loss).",
    "Oregonator: `use_dagger = rand < dagger_prob & dt > min_dt`, the rest supervised (`train_shortcut_oregonator.py:69-82`). Euler/Ball: `(1-w)·L_sup + w·L_comp`, L_comp only on h ≥ 2 (`train_shortcut_2d_dagger.py:83-101`, `train_ball3d.py:77-81`).")
add(G3, "C6", "Appendix A.4, mechanism sentence", "tex",
    r"solver-in-the-loop refinement compounds its own predictions and drifts off", None,
    r"composition-only objective trains on its own intermediate predictions and drifts off",
    "No solver in the loop (C1).")
add(G3, "C7", "Appendix A.4, choice of lambda", "tex",
    r"is the smallest mixing weight that delivers the Oregonator win without", None,
    r"is, of the three settings tested, the one that delivers the Oregonator win without",
    "Only λ ∈ {0, 0.1, 1} were run (`oregonator_pipeline/configs/training/shortcut_oregonator_dagger_{0_100,100_0}.yaml`, `neurips/configs/training/shortcut_euler2d_dagger_{0_100,100_0}.yaml`).")
add(G3, "C8", "Appendix B.2, training text", "tex",
    r"$h \in \{1, 2, 4, 8, 16, 32, 64\}$, and a $10\%$ DAgger refinement against",
    r"the reference solver. Per-environment hyperparameters are summarised in",
    r"$h \in \{1, 2, 4, 8, 16, 32, 64\}$, and a $10\%$ DAgger-style composition term: two chained $T/2$ predictions supervised against the target at $T$, defined for $h \ge 2$. On Oregonator the term replaces the supervised loss on a random $10\%$ of the $h \ge 2$ samples, with gradients through both steps; on Euler 2D and Ball 3D the loss is $0.9\,\mathcal{L}_{\text{sup}}$ on every sample plus $0.1$ times the composition loss on the $h \ge 2$ samples, with no gradient through the first step. Per-environment hyperparameters are summarised in",
    "Oregonator: Bernoulli routing, realised rate 7.7–9.4% per epoch (`oregonator_pipeline/logs/train_v3*.log`), no `no_grad`. Euler/Ball: `torch.no_grad()` on the first half-step (`train_shortcut_2d_dagger.py:94`, `train_ball3d.py:77`).")
add(G3, "C9", "Table 11, lambda row", "tex",
    r"DAgger weight $\lambda$ & $0.1$ & $0.1$ & $0.1$ \\", None,
    r"DAgger $\lambda$ & $0.1$ (sample prob.) & $0.1$ (loss weight) & $0.1$ (loss weight) \\",
    "Config keys: `dagger_prob` (Oregonator) vs `dagger_weight` (Euler, Ball) in the checkpoints' `run_config.json`.")

G4 = "D. Training-length, data and hyperparameter descriptions"
add(G4, "D1", "Method 3.1, horizon set", "tex",
    r"sampled uniformly from the set $\{2, 4, 8, 16, 32, 64\}$.", None,
    r"sampled uniformly from the set $\{1, 2, 4, 8, 16, 32, 64\}$.",
    "`horizon_set: [1, 2, 4, 8, 16, 32, 64]` in all three `run_config.json` / trainers; matches Table 1 caption and App. B.2.")
add(G4, "D2", "Method 3.1, count of horizons", "tex",
    r"two reasons. It covers two orders of magnitude in horizon with only six values,", None,
    r"two reasons. It covers two orders of magnitude in horizon with only seven values,", "As D1.")
add(G4, "D3", "Method 3.1, doubling property", "tex",
    r"structure ensures every horizon $T$ in the set has its half-horizon $T/2$ also in the set,", None,
    r"structure ensures every horizon $T \ge 2$ in the set has its half-horizon $T/2$ also in the set,",
    "h=1 has no half-horizon in the set.")
add(G4, "D4", "Table 1, Oregonator training set", "tex",
    r"Train trajectories    & $1200$ & $500$  & $1000$ \\", None,
    r"Train trajectories    & $400$ (of $1200$) & $500$  & $1000$ \\",
    "The Oregonator model trained on the first 400 trajectories: `oregonator_train_memmap.npy` shape (400, 201, 2, 256, 256); `train_v3_part1.log` \"train N=400\". The h5 holds 1200.")
add(G4, "D5", "Table 1, val MSE row label", "tex",
    r"Final val MSE         & $0.104$ & $0.016$ & $0.024$ \\", None,
    r"Best val MSE (all $h$) & $0.104$ & $0.016$ & $0.024$ \\",
    "Values are best-epoch validation MSE (epochs 57 / 31 / 59), averaged over horizons drawn from {1,…,64}; last-epoch values are 0.140 / 0.026 / 0.025.")
add(G4, "D6", "Appendix A.2 text (single-horizon)", "tex",
    r"ladder collapsed to a single value, validation MSE at $h{=}64$ is $1.7\times$", None,
    r"ladder collapsed to a single value, validation MSE (at $h{=}64$ for the single-horizon model, averaged over horizons for the multi-horizon model) is $1.7\times$",
    "`validate()` draws h from the configured horizon set (`train_shortcut_2d_dagger.py:39-53`, `train_ball3d.py:86-93`).")
add(G4, "D7", "Table 4 caption", "tex",
    r"only the training-horizon set differs. ``---'' on Oregonator: not run for cost",
    r"identical to the Euler 2D backbone.}",
    r"the training-horizon set differs, and the single-horizon runs were trained for fewer epochs ($30$ on Euler 2D and $40$ on Ball 3D, versus $40$ and $80$). An Oregonator single-horizon run ($30$ epochs) reached validation MSE $0.265$ at $h{=}64$, versus $0.104$ for the multi-horizon model.}",
    "Runs exist: `oregonator_pipeline/checkpoints/shortcut_oregonator_singlehorizon/seed0` (best 0.2649, 30 ep), `neurips/training/checkpoints/shortcut_euler2d_singlehorizon/seed0` (0.0277, 30 ep), `mujoco_pipeline/checkpoints/shortcut_ball3d_singlehorizon/seed0` (0.1427, 40 ep).")
add(G4, "D8", "Table 4 header", "tex",
    r"Environment & Training horizons & Val MSE at $h{=}64$ & Step-doubling probe \\", None,
    r"Environment & Training horizons & Val MSE (multi: all $h$; single: $h{=}64$) & Step-doubling probe \\",
    "As D6.")
add(G4, "D9", "Appendix A.4, what was held fixed", "tex",
    r"Everything else in the recipe (architecture, optimiser, multi-horizon ladder,",
    r"dataset) is held fixed.",
    r"Everything else in the recipe (architecture, optimiser, multi-horizon ladder, dataset) is held fixed except training length: the $\lambda{=}0$ and $\lambda{=}1$ runs were trained for $30$, $30$ and $40$ epochs (Oregonator, Euler 2D, Ball 3D), the default for $60$, $40$ and $80$.",
    "History lengths of the `dagger_ablation*` runs vs. the main runs (also in numbers.md, DAGGER_EPOCHS_*).")
add(G4, "D10", "Fig. 8 caption, quantity", "tex",
    r"\caption{\textbf{DAgger weight ablation.} Final validation MSE (log scale)", None,
    r"\caption{\textbf{DAgger weight ablation.} Best validation MSE (log scale)",
    "The bars are best-epoch values (`build_appendix_dagger.py:12-14` = history minima). The figure's own y-label also says \"Final val MSE\"; it names no environment, so it was not regenerated.")
add(G4, "D11", "Fig. 8 caption, comparison", "tex",
    r"worst; the hybrid wins or ties on every environment.}", None,
    r"worst. The $\lambda{=}0$ and $\lambda{=}1$ runs were trained for fewer epochs ($30$/$30$/$40$) than the default ($60$/$40$/$80$).}",
    "The comparison is epoch-unmatched (D9).")
add(G4, "D12", "Cross-seed appendix, seed 2", "tex",
    r"retrain each surrogate from scratch with three seeds ($\{0, 1, 2\}$) and", None,
    r"retrain each surrogate from scratch with three seeds ($\{0, 1, 2\}$; Oregonator seed~2 stopped at epoch $35$ of $60$) and",
    "`oregonator_pipeline/checkpoints/shortcut_oregonator_v3/seed2/`: last `epoch035.pt`, no history.json, `best.pt` epoch 34 (val 0.144).")
add(G4, "D13", "Appendix B.2, early stopping text", "tex",
    r"hyperparameters were tuned per seed. We use early stopping on validation",
    r"MSE with patience $15$ epochs.",
    r"hyperparameters were tuned per seed. On Oregonator we stop early after $15$ epochs without improvement in validation MSE or validation trust-signal $R^2$, and the run was resumed once (weights only) from an epoch-9 checkpoint; Euler 2D and Ball 3D run the full epoch budget. All three keep the best-validation-MSE checkpoint.",
    "Early stopping exists only in `train_shortcut_oregonator.py:300-390`; none in the Euler or Ball trainers. Oregonator resume: `train_v3_part1.log` (9 epochs) + `best_resumed_from_e9.pt`, `--resume` loads weights only (`:257-261`).")
add(G4, "D14", "Table 11, LR schedule", "tex",
    r"LR schedule         & cosine & cosine & cosine \\", None,
    r"LR schedule         & cosine & constant & cosine \\",
    "`train_shortcut_2d_dagger.py` has no scheduler; Ball uses `CosineAnnealingLR` (`train_ball3d.py:128`).")
add(G4, "D15", "Table 11, early stopping", "tex",
    r"Early-stopping patience & $15$ & $15$ & $15$ \\", None,
    r"Early-stopping patience & $15$ & none & none \\", "As D13.")
add(G4, "D16", "Table 11, Oregonator epochs", "tex",
    r"Epochs              & $60$ & $40$ & $80$ \\", None,
    r"Epochs              & $9{+}60$ & $40$ & $80$ \\", "As D13.")
add(G4, "D17", "Table 11, Oregonator samples", "tex",
    r"Total samples seen  & $300{,}000$ & $80{,}000$ & ${\approx}\,2{\times}10^{6}$ \\", None,
    r"Total samples seen  & $345{,}000$ & $80{,}000$ & ${\approx}\,2{\times}10^{6}$ \\",
    "69 epochs × 5,000 samples.")
add(G4, "D18", "Appendix B.2, recipe choice", "tex",
    r"The same hyperparameters are used across all three environments without",
    r"per-environment tuning.",
    r"Hyperparameters were not tuned per environment beyond the differences listed in Table~\ref{tab:training_hp}.",
    "Table 11 itself lists different LR, batch size, warmup, samples per epoch and epochs.")

G5 = "E. Solvers, architecture and benchmark setup"
add(G5, "E1", "Section 5.1, Oregonator parameters", "tex",
    r"$(\varepsilon, q, f, D) = (0.05, 0.002, 2.0, 1.0)$.", None,
    r"$q{=}0.002$ and $D{=}1$, with $\varepsilon$ and $f$ sampled per trajectory (Table~\ref{tab:oreg_params}).",
    "`oregonator_pipeline/scripts/generate_dataset.py:92-93`: ε ~ U(0.02, 0.08), f ~ U(0.5, 2.0) per trajectory. The preceding line ends \"Tyson parameters\".")
add(G5, "E2", "Section 5.1, Ball 3D collisions", "tex",
    r"and analytic axis-aligned wall collisions;", None,
    r"and axis-aligned wall collisions resolved at every substep (position projection and normal-velocity reflection with restitution);",
    "`mujoco_pipeline/envs/ball3d_env.py:76-93`.")
add(G5, "E3", "Section 5.1, architecture changes", "tex",
    r"backbone is dictated by data shape and we make no environment-specific architectural changes.", None,
    r"backbone is dictated by data shape, and the only environment-specific architectural change is the U-Net's boundary padding (circular for the periodic Oregonator grid, zero for Euler 2D).",
    "`models/shortcut_oregonator_2d.py:43-44,80,102` `padding_mode=\"circular\"`; `models/shortcut_pde_2d.py` default zero padding.")
add(G5, "E4", "Section 5.4, GPU speedups", "tex",
    r"the PDE surrogates saturate near $B{=}1$ at",
    r"$734\times$ on Oregonator and $186\times$ on Euler against the unbatched CPU reference;",
    r"the PDE surrogates reach $734\times$ on Oregonator at $B{=}1$ and $186\times$ on Euler at $B{=}32$ ($123\times$ at $B{=}1$) against the unbatched CPU reference;",
    f"`ablations/batch_bench/results/euler_h64.json` speedup_by_batch: B=1 {eh['1']:.1f}, B=32 {eh['32']:.1f}; `oregonator_h64.json` B=1 {oh['1']:.1f}.")
add(G5, "E5", "Appendix B.3, thread configuration", "tex",
    r"\texttt{torch.set\_num\_threads(8)} and \texttt{OMP\_NUM\_THREADS=8} to", None,
    r"\texttt{torch.set\_num\_threads(8)} (the NumPy solver runs with default threading) to",
    "`ablations/batch_bench/bench_cpu_horizon_sweep_clean.py:14` sets torch threads only; `OMP_NUM_THREADS` is set nowhere.")

G6 = "F. Metric and protocol descriptions"
add(G6, "F1", "Table 2 caption (scoring level)", "tex",
    r"\caption{Per-cell AUROC at the $75$th-percentile error threshold, $80{-}100$ pairs per cell.}", None,
    r"\caption{Step-doubling AUROC at the $75$th-percentile error threshold. Oregonator: per-cell (cells pooled over $100$ pairs; a cell is positive if its error exceeds the 75th percentile of its own pair's error map). Euler 2D and Ball 3D: per-trajectory over $80$ pairs (a pair is positive if its mean error exceeds the 75th percentile across pairs).}",
    "Oregonator column: `ablations/baselines/eval_uq_baselines_oregonator.py:85-116` (cell-level); Euler/Ball: `eval_tta_euler_ball3d.py:30-41` (pair-level).")
add(G6, "F2", "Section 5.2, Oregonator \"structural ceiling\"", "tex",
    r"The Oregonator AUROC saturates at",
    r"rather than a failure mode, and Mode~2 still recovers $40$--$68\%$ of the residual RMSE on these cells",
    r"The Oregonator column is scored per cell rather than per trajectory (see caption), which gives lower values ($0.64$--$0.77$); scored per trajectory as for the other environments it is $0.81$--$0.95$ (Table~\ref{tab:cross_seed_auroc}), and Mode~2 recovers $44$--$78\%$ of the residual RMSE on these cells",
    "The ceiling explanation follows from the metric, not the surrogate. 0.81–0.95 = Oregonator range of the new Table 5; 44–78% = OREG_RANGE of the new Table 6 (numbers.md).")
add(G6, "F3", "Table 3 caption (scoring level)", "tex",
    r"``---'' marks combinations not applicable on the indicated environment. Bold marks the highest value per column.}", None,
    r"``---'' marks combinations not applicable on the indicated environment. Bold marks the highest value per column. Scored per cell for step-doubling and random TTA on Oregonator and for the deep ensemble on Oregonator and Euler 2D; per trajectory elsewhere.}",
    "Per-row sources: `uq_baselines.json` (cell), `tta_{euler,ball3d}.json` (pair), `gradient_magnitude_*.json:auroc_pair_grad_mag` (pair), error head and conformal (pair), `ensemble_vs_step_doubling/results/{oregonator,euler}_test.json` (cell), `ball3d_ensemble.json` (pair).")
add(G6, "F4", "Section 5.3, per-environment baselines", "tex",
    r"Per-environment baselines (learned error head, conformal prediction) are competitive on the",
    r"environment they were calibrated for but degrade or become inapplicable elsewhere;",
    r"Per-environment baselines (learned error head, conformal prediction) need a separate training or calibration run on each environment;",
    "Both were run on all three environments (`error_head/train_and_eval{,_euler,_ball3d}.py`, `conformal/eval_conformal_{oregonator,euler,ball3d}.py`), and Table 3 has all nine cells for both.")
add(G6, "F5", "Appendix A.1, self-consistency setup", "tex",
    r"(Oregonator and Euler 2D, identical architecture and optimiser to the main",
    r"runs) using only the self-consistency loss",
    r"(Oregonator and Euler 2D, identical architecture to the main runs, AdamW at learning rate $3{\times}10^{-4}$ with no schedule, $20$ epochs) using only the self-consistency loss on samples with $T \ge 4$",
    "`ablations/self_consistency/train_self_consistency_{oregonator,euler}.py`: `--lr 3e-4`, `--epochs 20`, no scheduler; loss only where `dt > 2·base_dt` (h ≥ 4).")
add(G6, "F6", "Appendix A.1, identity probe", "tex",
    r"$\|f_\theta(s_0, T) - s_0\|_2$ on a held-out split as a direct probe of the", None,
    r"the mean squared difference between $f_\theta(s_0, T)$ and $s_0$ on a held-out split as a direct probe of the",
    "`train_self_consistency_oregonator.py:130` `F.mse_loss(pred, u0_v)` (physical units).")
add(G6, "F7", "Fig. 7 caption, identity probe", "tex",
    r"The trivial-fixed-point probe $\|f(s,T) - s\|$ (grey triangles)", None,
    r"The trivial-fixed-point probe $\mathrm{MSE}(f(s,T), s)$ (grey triangles)",
    "As F6. The figure's own legend says ‖f(s,T) − s‖; not an environment label, so not regenerated.")
add(G6, "F8", "Fig. 1 caption, how Mode 2 speedup is obtained", "tex",
    r"(a) CPU-CPU wall-clock speedup vs horizon for Mode 1 (surrogate alone, solid) and Mode 2 (with trust-aware", None,
    r"(a) CPU-CPU speedup vs horizon for Mode 1 (surrogate alone, measured wall-clock, solid) and Mode 2 (computed from the measured per-call times as $t_{\mathrm{sol}}/(3t_{\mathrm{sur}} + (1{-}q)\,t_{\mathrm{sol}})$, with trust-aware",
    "`build_fig1_v2.py` `m2_speedup` (also used for the effective speedup in panel b). No Mode 2 run was timed end to end.")
add(G6, "F9", "Section 5.5, Mode 2 speedup", "tex",
    r"while retaining a $\sim 3\times$ effective speedup.", None,
    r"while retaining an estimated $\sim 3\times$ effective speedup (computed from per-call timings; Figure~\ref{fig:mode2}).",
    "As F8 / F10.")
add(G6, "F10", "Fig. 6 caption, speedup annotation", "tex",
    r"the effective speedup retained.}", None,
    r"the effective speedup, computed as $1/(q\,t_{\mathrm{sur}}/t_{\mathrm{sol}} + 1 - q)$ from CPU timings at $B{=}1$ (Ball~3D: GPU surrogate at $B{=}32$).}",
    "`build_fig6_mode2.py:35-60`. This formula omits the extra step-doubling passes that Fig. 1 counts.")
add(G6, "F11", "Appendix A.7, q-sweep sample", "tex",
    r"distribution splits to characterise the trade-off.", None,
    r"distribution splits at $h{=}64$ to characterise the trade-off, on a separate sample of $80$ (trajectory, $t_0$) pairs per cell from the seed-0 model, so values differ from Table~\ref{tab:m1_m2_horizons}.",
    "`ablations/mode2_qsweep/results/summary.json` (`method` field: 80 pairs per cell); `eval_qsweep.py` `N_PAIRS = 80`. See QSWEEP_SAMPLE in numbers.md.")
add(G6, "F12", "Checklist, libraries", "chk",
    r"(PyTorch, NumPy, h5py, matplotlib, MuJoCo)", None,
    r"(PyTorch, NumPy, h5py, matplotlib)",
    "Ball 3D is pure NumPy (`ball3d_env.py`); MuJoCo is never imported.")

GX = "G. Additional, found while preparing this list (not in either report)"
add(GX, "X1", "Method 3.2, single-horizon probe", "tex",
    r"and the probe degenerates to noise", None,
    r"and the probe at $T/2$ queries a horizon the model was never trained on",
    "The single-horizon appendix reports validation MSE only; no AUROC was computed for the single-horizon models.")
add(GX, "X2", "Section 5.2, single-horizon comparison", "tex",
    r"shows that a surrogate with comparable prediction error has", None,
    r"shows that a single-horizon surrogate has",
    "Table 4: the single-horizon models are 1.7× and 6.0× worse, not comparable.")
add(GX, "X3", "Appendix A.11, step-doubling cost", "tex",
    r"same hardware takes under a second per cell, two surrogate forward passes at", None,
    r"same hardware takes under a second per cell, three surrogate forward passes at",
    "ê needs f(s,T), f(s,T/2) and f(·,T/2): three passes (Fig. 1 counts 3·t_sur). The Fig. 14 caption (\"step-doubling is two surrogate forward passes\") and the figure's text box say two as well.")

GV = "H. Value corrections from the reports (numbers, not descriptions)"
add(GV, "V1", "Appendix A.1, SC loss decrease", "tex",
    r"$\mathcal{L}_{\mathrm{SC}}$ drops by roughly six orders of magnitude over twenty", None,
    r"$\mathcal{L}_{\mathrm{SC}}$ drops by roughly three orders of magnitude over twenty",
    f"history.json `train_sc_loss`: first / minimum = 10^{orders['oreg']:.1f} (Oregonator), 10^{orders['euler']:.1f} (Euler).")
add(GV, "V2", "Fig. 7 caption, SC loss decrease", "tex",
    r"Training loss $\mathcal{L}_{\mathrm{SC}}$ (coloured circles) drops by six", None,
    r"Training loss $\mathcal{L}_{\mathrm{SC}}$ (coloured circles) drops by three", "As V1.")
add(GV, "V3", "Appendix A.1, epoch-5 level", "tex",
    r"already saturated near $10^{-6}$ with no gradient signal pointing away from the", None,
    r"already reached $10^{-6}$ to $10^{-5}$ with no gradient signal pointing away from the",
    f"Epoch-5 `train_sc_loss`: Oregonator {ep5['oreg']:.1e}, Euler {ep5['euler']:.1e}.")
add(GV, "V4", "Appendix A.1, validation MSE (Euler not flat)", "tex",
    r"epochs, while validation MSE against the reference solver stays flat throughout.", None,
    r"epochs, while validation MSE against the reference solver does not improve (Oregonator) or oscillates (Euler 2D).",
    f"`val_mse_vs_GT` range: Oregonator {valrng['oreg'][0]:.3f}–{valrng['oreg'][1]:.3f}, Euler {valrng['euler'][0]:.2f}–{valrng['euler'][1]:.2f}. The same \"flat\" wording recurs in the next sentences and the Fig. 7 caption (\"stays flat\").")
add(GV, "V5", "Appendix A.4, pure-DAgger ratio", "tex",
    r"Figure~\ref{fig:dagger} reports the result. Pure DAgger is $3{-}9\times$ worse", None,
    r"Figure~\ref{fig:dagger} reports the result. Pure DAgger is $3.6{-}9.5\times$ worse",
    f"Best-val ratios λ=1 / λ=0.1: {', '.join(f'{x:.2f}' for x in r1)} (Oregonator, Euler, Ball).")
add(GV, "V6", "Appendix A.4, lambda=0 gaps", "tex",
    r"($\lambda{=}0$) is competitive on Ball 3D and Euler 2D (within $5\%$ and", None,
    r"($\lambda{=}0$) is competitive on Ball 3D and Euler 2D (within $6\%$ and",
    f"λ=0 vs λ=0.1 best val: Ball +{r0['b']:.1f}%, Euler +{r0['e']:.1f}%, Oregonator +{r0['o']:.1f}%. Also change the next line's \"$31\\%$ of the hybrid\" to \"$26\\%$ of the hybrid\".")
add(GV, "V7", "Appendix A.4, Euler gap", "tex",
    r"$31\%$ of the hybrid, respectively) but loses $26\%$ on Oregonator, where", None,
    r"$26\%$ of the hybrid, respectively) but loses $26\%$ on Oregonator, where", "As V6.")
add(GV, "V8", "Section 5.3, closest competitor", "tex",
    r"The closest competitor is the deep ensemble at mean $0.77$,", None,
    r"The closest competitors are the learned error head (mean $0.78$) and the deep ensemble (mean $0.77$),",
    "Means of the printed Table 3 rows: error head 0.781, ensemble 0.767.")
add(GV, "V9", "Table 7, step-doubling h=32", "tex",
    r"\textbf{Step-doubling (ours)} & $\mathbf{0.95}$ & $\mathbf{0.91}$ & $0.81$ & $0.81$ & $\mathbf{0.83}$ & $\mathbf{0.86}$ & $\mathbf{0.86}$ \\", None,
    r"\textbf{Step-doubling (ours)} & $\mathbf{0.95}$ & $\mathbf{0.91}$ & $0.81$ & $0.81$ & $\mathbf{0.82}$ & $\mathbf{0.86}$ & $\mathbf{0.86}$ \\",
    f"`energy_momentum_ball3d.json` test h=32 `auroc_step_doubling` = {sd32:.4f}.")
add(GV, "V10", "Appendix A.10, energy margin", "tex",
    r"step-doubling, and beats it by more than $0.04$ on only $2$ of $6$", None,
    r"step-doubling, and beats it by $0.03$--$0.04$ on only $2$ of $6$",
    f"Energy − SD at h=8, 16: {marg[0]:+.3f}, {marg[1]:+.3f}.")
add(GV, "V11", "Appendix A.10, momentum", "tex",
    r"horizons ($h{=}8$ and $h{=}16$). Momentum residual is at chance for every", None,
    r"horizons ($h{=}8$ and $h{=}16$). Momentum residual is at chance for five of six",
    f"Momentum AUROC h=64 = {mom[-1]:.3f}; the next line \"horizon ($\\le 0.51$ for five of six horizons)\" then reads as a restatement and can be trimmed.")
add(GV, "V12", "Appendix A.8, beyond-T_max Oregonator range", "tex",
    r"On Oregonator, AUROC stays at $0.81$--$0.97$ across all three splits up to $h{=}128$,", None,
    r"On Oregonator, AUROC stays at $0.84$--$0.97$ across all three splits up to $h{=}128$,",
    "`ablations/beyond_tmax/results/summary.json`, Oregonator h64–h128.")
add(GV, "V13", "Appendix A.8, beyond-T_max Euler range", "tex",
    r"saturating at $0.96$--$1.00$ across all six cells:", None,
    r"saturating at $0.99$--$1.00$ across all six extrapolated cells:",
    "Same file, Euler h80 and h96 (0.96 is the in-ladder h=64 cell).")
add(GV, "V14", "Fig. 13 caption, Euler variance", "tex",
    r"diverge; high variance on Euler 2D reflects a bimodal split between", None,
    r"diverge; high variance on Euler 2D at $k \le 2$ reflects a bimodal split between",
    f"`rollout_h64.json` Euler std_rmse: k=1 {ro_sd['1']:.2f}, k=2 {ro_sd['2']:.2f}, k=4 {ro_sd['4']:.2f}, k=8 {ro_sd['8']:.2f}.")
add(GV, "V15", "Table 12, CPU threads", "tex",
    r"CPU         & AMD Ryzen 9 8940HX (16 cores, $24$ threads) \\", None,
    r"CPU         & AMD Ryzen 9 8940HX (16 cores, $32$ threads) \\",
    f"`os.cpu_count()` = {ncpu} on the benchmark machine.")
add(GV, "V16", "Table 12, OS", "tex",
    r"OS          & Ubuntu $22.04$ \\", None,
    r"OS          & Pop!\_OS $22.04$ (Ubuntu-based) \\",
    f"/etc/os-release: {osname}.")
add(GV, "V17", "Section 5.2, cells above 0.65", "tex",
    r"and $50$ cells above $0.65$.", None,
    r"and $50$ cells at or above $0.65$.",
    "50 cells are strictly above 0.65 on raw values, but two print as 0.65 in Table 2.")
add(GV, "V18", "Appendix B.3, code availability", "tex",
    r"and evaluation protocols will be released on acceptance. The reference", None,
    r"and evaluation protocols are available at \url{https://github.com/Lossfunk/Hybrid-Neural-World-Models}. The reference",
    "The repository is public (checked with `git ls-remote`).")
add(GV, "V19", "Checklist, code item", "chk",
    r"\item[] Justification: An anonymized supplement provides all source code,", None,
    r"\item[] Justification: The public repository \url{https://github.com/Lossfunk/Hybrid-Neural-World-Models} provides all source code,",
    "Also replace \"the supplement contains\" with \"the repository contains\" two lines below, and in the reproducibility item \"anonymized code with trained checkpoints is provided in the supplementary material\" with the URL.")

# ---------------------------------------------------------------- lines whose numbers your new tables change
PTR = [
    ("Section 5.2, failure cell and cross-seed value", r"$h{=}16$, where AUROC drops to $0.44$ (below chance)", "Table 5 now gives 0.51 ± 0.02 (h=16) and 0.49 ± 0.03 (h=32)."),
    ("Section 5.2, cross-seed quote", r"($0.43 \pm 0.02$ across", "Same; 0.43 ± 0.02 was the old h=32 cell."),
    ("Section 5.5, Mode 2 range", r"cuts trajectory-mean RMSE by $43\%{-}52\%$ relative to the surrogate-alone baseline", "FIG6_RANGE: 42–48%."),
    ("Section 5.5, largest cut", r"with the largest cuts ($-89\%$ on Euler", "LARGEST_CUT: Ball 3D test h=2, −79%."),
    ("Limitations, failure cell", r"On Ball~3D under far-OOD restitution and gravity at $h{=}16$, AUROC drops to $0.44$:", "Table 5."),
    ("Cross-seed text, spread", r"seed-to-seed standard deviation of AUROC is $\le 0.04$ at the median, with the", "T5_MEDIAN_SD / T5_MAX_SD / T5_MAX_SD_CELL."),
    ("Cross-seed text, Oregonator structure", r"saturates near $0.7$ for structural reasons", "Per-trajectory Oregonator is 0.81–0.95 in the new Table 5."),
    ("Table 5 caption", r"three independent training seeds. Median seed-to-seed standard deviation is", "T5_MEDIAN_SD, T5_MAX_SD, T5_MAX_SD_CELL; the \"within $1\\sigma$\" sentence."),
    ("Appendix A.6 result", r"cut exceeds the $25\%$ random floor on every cell. Median gaps above random", "PER_ENV_MEDIANS, GAP_MAX, NEAR_FLOOR_CELLS."),
    ("Table 6 caption", r"Reductions exceed the $-25\%$ random-deferral floor on every cell. Largest cuts:", "LARGEST_CUT and the new table."),
    ("Broader impact, failure cell", r"reports one such failure in our benchmarks (Ball 3D under far-OOD restitution and gravity at $h{=}16$, where", "Table 5."),
]

# ---------------------------------------------------------------- verify + render
def locate(src, s):
    n = src.count(s)
    if n != 1:
        raise SystemExit(f"find string occurs {n}x: {s[:90]!r}")
    i = src.index(s)
    return i, src.count("\n", 0, i) + 1


md = ["# Camera-ready text fixes\n",
      "Each fix gives the location, an exact **find** string, and the smallest **replacement** that describes what "
      "was actually run. Where a fix spans several source lines, select from the start of **find (start)** through the "
      "end of **find (end)** and replace the whole span; text outside the two strings is untouched.\n",
      f"Line numbers are for the submitted source (`neurips_2026.tex` as submitted; = preprint line + {OFFSET}, the "
      "body is identical) and for `checklist.tex`. Your Overleaf edits will shift them, so search for the find string. "
      "Every find string was checked to occur exactly once, and every corrected number was recomputed from the result "
      "files by `scripts/make_text_fixes.py`.\n",
      "Skipped as already handled, per your instructions: the Euler perturbation and Table 9 (including the "
      "\"disjoint seed ranges\" sentence in App. B.1 and Euler OOD-near being Sedov-only), the threshold wording, "
      "Ball 3D timing, Fig. 1 numbers, the cross-seed protocol, the \"200 per cell\" wording, and Fig. 3.\n"]
group = None
for g, fid, title, which, start, end, repl, why in F:
    src = TEX if which == "tex" else CHK
    i, l1 = locate(src, start)
    lo = l1 + (OFFSET if which == "tex" else 0)
    loc = f"`{'neurips_2026.tex' if which == 'tex' else 'checklist.tex'}` line {lo}"
    if end:
        j, l2 = locate(src, end)
        assert j >= i, (fid, "end before start")
        loc += f"–{l2 + (OFFSET if which == 'tex' else 0)}"
    if g != group:
        md.append(f"\n## {g}\n"); group = g
    md.append(f"### {fid}. {title}\n")
    md.append(f"- **Where:** {loc}\n")
    md.append(f"- **Find{' (start)' if end else ''}:**\n```latex\n{start}\n```\n")
    if end:
        md.append(f"- **Find (end):**\n```latex\n{end}\n```\n")
    md.append(f"- **Replace with:**\n```latex\n{repl}\n```\n")
    md.append(f"- **What was run:** {why}\n")

md.append("\n## I. Lines whose numbers change with your new Table 5 / Table 6 / Fig. 6 (no replacement given)\n")
md.append("These are not description-vs-code fixes; the replacement values are in `numbers.md`.\n")
for title, s, note in PTR:
    i, l = locate(TEX, s)
    md.append(f"- `neurips_2026.tex` line {l + OFFSET} ({title}): `{s}` → {note}\n")
i, l = locate(CHK, "AUROC drops to $0.44$")
md.append(f"- `checklist.tex` line {l} (limitations item): `AUROC drops to $0.44$` → Table 5.\n")

(OUT / "text_fixes.md").write_text("\n".join(md))
print(f"wrote text_fixes.md: {len(F)} fixes, {len(PTR) + 1} pointers")
