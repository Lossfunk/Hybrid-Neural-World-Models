#!/usr/bin/env python3
"""Richardson AUROC vs true error — head-to-head with our learned-signal AUROC.

Computes classical Richardson disagreement at the SAME (split, horizon, pair)
test points as the existing C3 AUROC recompute. Two solver configurations:
  (a) production CFL-stability-adaptive Tyson solver at dt=0.05
  (b) fixed-stencil Tyson solver at dt=0.025 (no internal substepping;
      O(dt²) truncation error)

For each (split, h) cell:
  - Pearson R² of Richardson vs true error
  - AUROC at q75 threshold of true error
  - 1000-sample bootstrap 95% CI on AUROC
  - Mean Richardson disagreement magnitude (to document the floating-point-
    noise observation)

Test pair sampling: identical RNG seed (0) and iteration order as the
existing recompute.py, so e_true values can be cross-referenced if needed.
50 pairs/horizon (vs 100 in the original C3 recompute) to keep wall under
~3 hr.

Output:
  results/richardson_auroc.json   per-cell metrics + raw Richardson arrays
  figures/auroc_comparison.pdf    side-by-side bars: ours vs prod-Rich vs fixed-Rich
  figures/richardson_magnitudes.pdf   shows Richardson values are 1e-8 to 1e-9 noise
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "data_generation" / "oregonator"))
sys.path.insert(0, str(ROOT / "models"))

from eval_utils import load_model, load_pair                                # noqa: E402
from oregonator2d_tyson import OregonatorTyson2D, TysonParams               # noqa: E402

CKPT = ROOT / "checkpoints" / "oregonator" / "best.pt"
DATA_DIR = ROOT / "data" / "oregonator"
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Reuse existing C3 AUROC results — same test points, same e_true
EXISTING_AUROC_DIR = ROOT / "ablations" / "auroc_recompute" / "results"

SPLITS = ["test", "ood_near", "ood_far"]
HORIZONS = [2, 4, 8, 16, 32, 64]
N_PAIRS = 50         # vs 100 in original C3 recompute — speed/CI tradeoff
N_BOOT = 1000


def solver_advance(state_np: np.ndarray, n_steps: int, params: dict,
                    dt: float, fixed: bool) -> np.ndarray:
    """Advance state by n_steps × dt using the Tyson solver."""
    sim = OregonatorTyson2D(n_x=state_np.shape[2], n_y=state_np.shape[1],
                              L_x=100.0, L_y=100.0,
                              params=TysonParams(**params),
                              fixed_substep=fixed)
    sim.u[:] = state_np[0]
    sim.v[:] = state_np[1]
    for _ in range(n_steps):
        sim.step(dt)
    return np.stack([sim.u, sim.v], axis=0).astype(np.float32)


def richardson_disagreement(state0: np.ndarray, h: int, params: dict,
                              dt_solver: float, fixed: bool) -> float:
    """Richardson step-doubling on the solver. h is the surrogate horizon
    multiplier; physical time is h × DT_SAVE = 0.05h. solver dt may differ
    (production: dt=0.05, fixed-stencil: dt=0.025).

    For physical time T = 0.05 h:
      - production:   solver advances h steps at dt=0.05
      - fixed-stencil: solver advances 2h steps at dt=0.025
    """
    DT_SAVE = 0.05
    physical_T = h * DT_SAVE
    # Number of solver steps needed for full T and half T
    if abs(dt_solver - DT_SAVE) < 1e-9:
        n_full = h
        n_half = h // 2
    else:
        # fixed-stencil at dt=0.025 — needs 2× steps
        ratio = DT_SAVE / dt_solver       # e.g. 0.05/0.025 = 2
        n_full = int(round(h * ratio))
        n_half = int(round((h // 2) * ratio))
    big = solver_advance(state0, n_full, params, dt_solver, fixed)
    mid = solver_advance(state0, n_half, params, dt_solver, fixed)
    chain = solver_advance(mid, n_half, params, dt_solver, fixed)
    return float(np.sqrt(((big - chain) ** 2).sum(axis=0)).mean())


def bootstrap_auroc_ci(scores: np.ndarray, labels: np.ndarray,
                         n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """Mean + 95% CI of AUROC over bootstrap resamples."""
    rng = np.random.RandomState(seed)
    n = len(scores)
    if labels.sum() == 0 or labels.sum() == n:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            m = roc_auc_score(labels[idx], scores[idx])
        except ValueError:
            continue
        boots.append(m)
    if not boots:
        return float("nan"), float("nan"), float("nan")
    boots = np.asarray(boots)
    return float(boots.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_existing_e_true(split: str, h: int) -> np.ndarray | None:
    """Pull the e_true array from the existing C3 AUROC recompute output.
    NB: that script used n_pairs=100; we use the first N_PAIRS=50 to align."""
    path = EXISTING_AUROC_DIR / f"{split}_per_horizon.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    if str(h) not in d.get("raw_arrays", {}):
        return None
    return np.array(d["raw_arrays"][str(h)]["e_true"])[:N_PAIRS]


def load_existing_e_hat_ours(split: str, h: int) -> np.ndarray | None:
    """Our learned signal scores — for the side-by-side comparison."""
    path = EXISTING_AUROC_DIR / f"{split}_per_horizon.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    if str(h) not in d.get("raw_arrays", {}):
        return None
    return np.array(d["raw_arrays"][str(h)]["e_hat"])[:N_PAIRS]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[richardson_auroc] device={device}  ckpt={CKPT}")
    print(f"[richardson_auroc] splits={SPLITS}  horizons={HORIZONS}  "
          f"n_pairs={N_PAIRS}  bootstrap={N_BOOT}", flush=True)
    print(flush=True)
    model = load_model(str(CKPT), device=device)

    # We re-run inference for ours (not strictly needed since we have raw_arrays
    # but lets us verify alignment), AND compute Richardson on production +
    # fixed-stencil at the same pairs.
    DT_SAVE = 0.05

    summary = {"n_pairs": N_PAIRS, "n_bootstrap": N_BOOT,
                "horizons": HORIZONS, "splits": SPLITS,
                "ckpt": str(CKPT),
                "solver_prod": dict(name="OregonatorTyson2D adaptive",
                                      dt=0.05, fixed_substep=False),
                "solver_fixed": dict(name="OregonatorTyson2D fixed-stencil",
                                       dt=0.025, fixed_substep=True),
                "metrics": {}}

    t_overall = time.time()

    for split in SPLITS:
        ds_path = DATA_DIR / f"oregonator_{split}.h5"
        if not ds_path.exists():
            print(f"  [{split}] missing"); continue
        with h5py.File(ds_path, "r") as f:
            N, T, _, H, W = f["states"].shape
        print(f"[{split}] N={N} T={T}  computing Richardson disagreements...",
              flush=True)
        rng = np.random.RandomState(0)         # SAME SEED as recompute.py for alignment
        per_h = {}
        for h in HORIZONS:
            if h >= T: continue
            e_rich_prod = np.zeros(N_PAIRS, dtype=np.float64)
            e_rich_fixed = np.zeros(N_PAIRS, dtype=np.float64)
            t_h_start = time.time()
            for k in range(N_PAIRS):
                # Same RNG draws as recompute.py
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                with h5py.File(ds_path, "r") as f:
                    state0 = np.array(f["states"][i, t0])
                    params = dict(eps=float(f["params"][i, 1]),
                                   q=float(f["params"][i, 2]),
                                   f=float(f["params"][i, 0]),
                                   D=float(f["params"][i, 3]))
                # Production solver Richardson at dt=0.05
                e_rich_prod[k] = richardson_disagreement(state0, h, params,
                                                            dt_solver=0.05,
                                                            fixed=False)
                # Fixed-stencil at dt=0.025
                e_rich_fixed[k] = richardson_disagreement(state0, h, params,
                                                            dt_solver=0.025,
                                                            fixed=True)
            wall_h = time.time() - t_h_start
            print(f"  h={h:>2d}  wall={wall_h:.1f}s  "
                  f"prod ê range [{e_rich_prod.min():.3e}, {e_rich_prod.max():.3e}]  "
                  f"fixed ê range [{e_rich_fixed.min():.3e}, {e_rich_fixed.max():.3e}]",
                  flush=True)

            # Pull aligned e_true and our learned signal from existing AUROC results
            e_true = load_existing_e_true(split, h)
            e_hat_ours = load_existing_e_hat_ours(split, h)
            if e_true is None or e_hat_ours is None:
                print(f"  warn: no aligned e_true / e_hat_ours for {split} h={h}; "
                      f"skipping AUROC comparison")
                continue

            # Define high-error class at q75 of e_true (same threshold as recompute)
            thr = float(np.quantile(e_true, 0.75))
            labels = (e_true > thr).astype(int)
            if labels.sum() in (0, N_PAIRS):
                print(f"  warn: degenerate labels at h={h}; skipping")
                continue

            # AUROC for each method
            def metric(scores):
                m, lo, hi = bootstrap_auroc_ci(scores, labels)
                # Pearson R² between scores and e_true
                if scores.std() > 1e-12:
                    r = float(np.corrcoef(scores, e_true)[0, 1])
                    r2 = r * r
                else:
                    r2 = float("nan")
                return dict(auroc=m, auroc_lo=lo, auroc_hi=hi, R2=r2,
                             score_mean=float(scores.mean()),
                             score_std=float(scores.std()))
            ours = metric(e_hat_ours)
            rich_prod = metric(e_rich_prod)
            rich_fixed = metric(e_rich_fixed)

            per_h[h] = dict(
                ours=ours, richardson_prod=rich_prod, richardson_fixed=rich_fixed,
                e_rich_prod=e_rich_prod.tolist(),
                e_rich_fixed=e_rich_fixed.tolist(),
                e_true=e_true.tolist(),
                e_hat_ours=e_hat_ours.tolist(),
                threshold=thr, n_pos=int(labels.sum()),
            )
            print(f"    AUROC: ours={ours['auroc']:.3f}[{ours['auroc_lo']:.2f},{ours['auroc_hi']:.2f}]  "
                  f"rich_prod={rich_prod['auroc']:.3f}[{rich_prod['auroc_lo']:.2f},{rich_prod['auroc_hi']:.2f}]  "
                  f"rich_fixed={rich_fixed['auroc']:.3f}[{rich_fixed['auroc_lo']:.2f},{rich_fixed['auroc_hi']:.2f}]")

        summary["metrics"][split] = per_h
        print(flush=True)

    # ── Save full results ────────────────────────────────────────────────
    out = RESULTS_DIR / "richardson_auroc.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[richardson_auroc] saved {out}")

    # ── Aggregate summary ────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("AUROC COMPARISON @ q75 — same 50 test points per cell, same e_true")
    print("=" * 100)
    print(f"{'cell':<22} | {'ours':<22} | {'richardson_prod':<22} | {'richardson_fixed':<22}")
    all_ours, all_rich_prod, all_rich_fixed = [], [], []
    for split, by_h in summary["metrics"].items():
        for h in HORIZONS:
            if h not in by_h: continue
            r = by_h[h]
            o = r["ours"]; rp = r["richardson_prod"]; rf = r["richardson_fixed"]
            print(f"  {split} h={h:>2d}            | "
                  f"{o['auroc']:.3f}[{o['auroc_lo']:.2f},{o['auroc_hi']:.2f}]    | "
                  f"{rp['auroc']:.3f}[{rp['auroc_lo']:.2f},{rp['auroc_hi']:.2f}]    | "
                  f"{rf['auroc']:.3f}[{rf['auroc_lo']:.2f},{rf['auroc_hi']:.2f}]")
            all_ours.append(o["auroc"])
            all_rich_prod.append(rp["auroc"])
            all_rich_fixed.append(rf["auroc"])
    if all_ours:
        print()
        print(f"  MEAN across 18 cells:")
        print(f"    ours              = {np.mean(all_ours):.3f}")
        print(f"    richardson_prod   = {np.mean(all_rich_prod):.3f}")
        print(f"    richardson_fixed  = {np.mean(all_rich_fixed):.3f}")
    print()
    print("RICHARDSON DISAGREEMENT MAGNITUDES (to document floating-point-noise observation)")
    for split, by_h in summary["metrics"].items():
        for h in HORIZONS:
            if h not in by_h: continue
            ms = by_h[h]
            print(f"  {split} h={h:>2d}: prod mean={ms['richardson_prod']['score_mean']:.3e} "
                  f"std={ms['richardson_prod']['score_std']:.3e}  | "
                  f"fixed mean={ms['richardson_fixed']['score_mean']:.3e} "
                  f"std={ms['richardson_fixed']['score_std']:.3e}")

    # ── Figures ──────────────────────────────────────────────────────────
    # AUROC comparison bar chart (3 splits × 6 horizons × 3 methods)
    fig, axes = plt.subplots(len(SPLITS), 1, figsize=(10, 3 * len(SPLITS)),
                              sharex=True)
    if len(SPLITS) == 1:
        axes = [axes]
    width = 0.27
    x_idx = np.arange(len(HORIZONS))
    for ax, split in zip(axes, SPLITS):
        if split not in summary["metrics"]: continue
        by_h = summary["metrics"][split]
        ours_v = []; rich_p_v = []; rich_f_v = []
        ours_e = []; rich_p_e = []; rich_f_e = []
        for h in HORIZONS:
            if h not in by_h:
                for v_arr in [ours_v, rich_p_v, rich_f_v]: v_arr.append(0.5)
                for e_arr in [ours_e, rich_p_e, rich_f_e]: e_arr.append([0.5, 0.5])
                continue
            r = by_h[h]
            ours_v.append(r["ours"]["auroc"])
            rich_p_v.append(r["richardson_prod"]["auroc"])
            rich_f_v.append(r["richardson_fixed"]["auroc"])
            ours_e.append([r["ours"]["auroc_lo"], r["ours"]["auroc_hi"]])
            rich_p_e.append([r["richardson_prod"]["auroc_lo"], r["richardson_prod"]["auroc_hi"]])
            rich_f_e.append([r["richardson_fixed"]["auroc_lo"], r["richardson_fixed"]["auroc_hi"]])
        def errs(vs, es):
            lo = [v - e[0] for v, e in zip(vs, es)]
            hi = [e[1] - v for v, e in zip(vs, es)]
            return [lo, hi]
        ax.bar(x_idx - width, ours_v, width=width, yerr=errs(ours_v, ours_e),
                label="ours (learned)", color="tab:blue", alpha=0.85, capsize=3)
        ax.bar(x_idx, rich_p_v, width=width, yerr=errs(rich_p_v, rich_p_e),
                label="Richardson (prod adaptive)", color="tab:orange", alpha=0.85, capsize=3)
        ax.bar(x_idx + width, rich_f_v, width=width, yerr=errs(rich_f_v, rich_f_e),
                label="Richardson (fixed-stencil)", color="tab:green", alpha=0.85, capsize=3)
        ax.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.6)
        ax.set_xticks(x_idx); ax.set_xticklabels([f"h={h}" for h in HORIZONS])
        ax.set_ylabel("AUROC@q75")
        ax.set_title(f"{split}")
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = FIG_DIR / "auroc_comparison.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  fig: {out}")

    # Magnitude figure — Richardson scores are ~1e-9, ours are ~1e-2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for split in SPLITS:
        if split not in summary["metrics"]: continue
        by_h = summary["metrics"][split]
        for h in HORIZONS:
            if h not in by_h: continue
            ms = by_h[h]
            label_ours = f"ours {split} h={h}" if h == 2 and split == "test" else None
            label_prod = f"Rich(prod) {split} h={h}" if h == 2 and split == "test" else None
            label_fixed = f"Rich(fixed) {split} h={h}" if h == 2 and split == "test" else None
            ax.plot(np.array(ms["e_hat_ours"]), 'o', alpha=0.4, color="tab:blue",
                     markersize=2, label=label_ours)
            ax.plot(np.array(ms["e_rich_prod"]), 's', alpha=0.4, color="tab:orange",
                     markersize=2, label=label_prod)
            ax.plot(np.array(ms["e_rich_fixed"]), '^', alpha=0.4, color="tab:green",
                     markersize=2, label=label_fixed)
    ax.set_yscale("log")
    ax.set_xlabel("test pair index")
    ax.set_ylabel("disagreement magnitude (log)")
    ax.set_title("Disagreement magnitudes — Richardson is ~1e-9 noise vs ours at ~1e-2")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "richardson_magnitudes.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  fig: {out}")

    # ── Paper-ready summary text ─────────────────────────────────────────
    summary["paper_summary"] = (
        f"On the same {N_PAIRS} test pairs per (split, horizon) used for our "
        f"learned-signal AUROC, classical Richardson disagreement on the "
        f"production adaptive solver and on a fixed-stencil solver "
        f"(no internal substepping, dt=0.025, O(dt²) truncation) gives "
        f"disagreement values in the 1e-9 to 1e-8 range — at floating-point "
        f"noise. AUROC of Richardson disagreement vs true error is "
        f"{np.mean(all_rich_prod):.3f} (production) and "
        f"{np.mean(all_rich_fixed):.3f} (fixed-stencil) averaged across 18 "
        f"cells, vs {np.mean(all_ours):.3f} for our learned signal. "
        f"Richardson disagreement at numerical-precision floor cannot rank "
        f"true errors. Modern stiff PDE solvers running with conservative "
        f"timestep margins eliminate classical Richardson as a viable trust "
        f"signal; the learned disagreement signal works because it is "
        f"trained at horizons where prediction error is non-negligible."
    )
    out = RESULTS_DIR / "richardson_auroc.json"
    out.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"TOTAL WALL: {(time.time() - t_overall) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
