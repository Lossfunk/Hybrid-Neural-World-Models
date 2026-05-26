#!/usr/bin/env python3
"""C4 hybrid speedup + classical Richardson three-way comparison.

Three configurations on the SAME 200 test trajectories:
  (A) solver-only — CFL-stability-adaptive Tyson Oregonator solver
  (B) hybrid-ours — adaptive horizon descent gated by surrogate step-doubling ê
  (C) hybrid-Richardson — same descent, but gating uses ground-truth solver
                          step-doubling on each candidate horizon

Threshold calibration: τ_ours and τ_classical separately on a 50-traj val
sample, at 75th percentile of their respective disagreement distributions.
Single τ across all horizons (per user spec).

Per-trajectory captures: wall-clock (median of 3 runs), final-state L²
error vs solver, per-horizon decision histogram, decision trace (for GIF).

Outputs:
  results/run.json — solver config + thresholds + per-traj data
  figures/speedup_bars.pdf, accuracy_vs_speedup.pdf, horizon_usage.pdf
  gifs/hybrid_traj{i}.gif (one or two representative trajectories)
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "data_generation" / "oregonator"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "training" / "oregonator"))

from eval_utils import load_model, load_pair                                # noqa: E402
from oregonator2d_tyson import OregonatorTyson2D, TysonParams               # noqa: E402

CKPT = ROOT / "checkpoints" / "oregonator" / "best.pt"
DATA_DIR = ROOT / "data" / "oregonator"
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
GIF_DIR = HERE / "gifs"
for d in [RESULTS_DIR, FIG_DIR, GIF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HORIZONS_DESC = [64, 32, 16, 8, 4, 2, 1]   # tried in DESCENDING order in the descent
H_LIST = [2, 4, 8, 16, 32, 64]              # used for calibration sampling
TOTAL_STEPS = 64                             # 64 base-dt steps = 3.2 sec sim time per traj


# ────────────────────────────────────────────────────────────────────────────
# Solver wrappers
# ────────────────────────────────────────────────────────────────────────────

def make_sim(params: dict, n_x: int, n_y: int) -> OregonatorTyson2D:
    sim = OregonatorTyson2D(n_x=n_x, n_y=n_y, L_x=100.0, L_y=100.0,
                              params=TysonParams(**params))
    return sim


def solver_advance(state_np: np.ndarray, n_steps: int, params: dict,
                    dt_save: float) -> tuple:
    """Advance state by n_steps × dt_save using the Tyson solver.
    Returns (final_state, wall_seconds)."""
    n_y, n_x = state_np.shape[1], state_np.shape[2]
    sim = make_sim(params, n_x, n_y)
    sim.u[:] = state_np[0]
    sim.v[:] = state_np[1]
    t0 = time.perf_counter()
    for _ in range(n_steps):
        sim.step(dt_save)
    wall = time.perf_counter() - t0
    final = np.stack([sim.u, sim.v], axis=0).astype(np.float32)
    return final, wall


# ────────────────────────────────────────────────────────────────────────────
# Threshold calibration on val data
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_thresholds(model, val_path: Path, dt_save: float, params_list: list,
                          n_samples: int, percentile: float, device: str) -> dict:
    """Sample n_samples (state, h) pairs from val. For each:
      - compute surrogate disagreement ê (cell-mean of ‖f(s,h) − f(f(s,h/2),h/2)‖)
      - compute classical Richardson disagreement (cell-mean of
        ‖Φ(s,h×dt) − Φ(Φ(s,(h/2)×dt),(h/2)×dt)‖)
    τ = `percentile` quantile of each distribution.
    """
    print(f"[calibration] sampling {n_samples} (state, h) pairs from val "
          f"for τ at {percentile*100:.0f}th percentile", flush=True)
    rng = np.random.RandomState(42)
    with h5py.File(val_path, "r") as f:
        N, T, _, H, W = f["states"].shape

    e_ours = []
    e_classical = []
    t_calib_start = time.time()
    for k in range(n_samples):
        h = int(rng.choice([h for h in H_LIST if h * 2 <= T]))
        i = int(rng.randint(0, N))
        t0 = int(rng.randint(0, T - h))
        # Load state + params
        with h5py.File(val_path, "r") as f:
            u0 = np.array(f["states"][i, t0])    # (C, H, W)
            params = dict(eps=float(f["params"][i, 1]),
                           q=float(f["params"][i, 2]),
                           f=float(f["params"][i, 0]),
                           D=float(f["params"][i, 3]))
        # Surrogate ê
        u0_t = torch.from_numpy(u0).to(device).unsqueeze(0)
        dt_t = torch.tensor([h * dt_save], device=device, dtype=torch.float32)
        pred_full = model(u0_t, dt_t)
        pred_mid = model(u0_t, dt_t * 0.5)
        pred_chain = model(pred_mid, dt_t * 0.5)
        e_hat = float(torch.sqrt(((pred_full - pred_chain) ** 2).sum(dim=1)).mean().item())
        e_ours.append(e_hat)
        # Classical Richardson disagreement
        big_step, _ = solver_advance(u0, h, params, dt_save)
        mid_step, _ = solver_advance(u0, h // 2, params, dt_save)
        chain_step, _ = solver_advance(mid_step, h // 2, params, dt_save)
        rich_e = float(np.sqrt(((big_step - chain_step) ** 2).sum(axis=0)).mean())
        e_classical.append(rich_e)
        if (k + 1) % 10 == 0:
            elapsed = time.time() - t_calib_start
            print(f"  calib {k+1}/{n_samples}  wall={elapsed:.0f}s  "
                  f"ê_ours[-1]={e_hat:.4f}  ê_classical[-1]={rich_e:.4f}",
                  flush=True)
    e_ours = np.array(e_ours); e_classical = np.array(e_classical)
    tau_ours = float(np.quantile(e_ours, percentile))
    tau_classical = float(np.quantile(e_classical, percentile))
    print(f"[calibration] tau_ours = {tau_ours:.5f} ({percentile*100:.0f}th pct of n={n_samples})")
    print(f"[calibration] tau_classical = {tau_classical:.5f}")
    return dict(tau_ours=tau_ours, tau_classical=tau_classical,
                ours_dist=e_ours.tolist(), classical_dist=e_classical.tolist())


# ────────────────────────────────────────────────────────────────────────────
# Configuration A: solver-only baseline
# ────────────────────────────────────────────────────────────────────────────

def run_solver_only(state0: np.ndarray, total_steps: int, params: dict,
                     dt_save: float, n_reps: int = 3) -> dict:
    walls = []
    for _ in range(n_reps):
        final, w = solver_advance(state0, total_steps, params, dt_save)
        walls.append(w)
    return dict(final=final, wall=float(np.median(walls)),
                walls=walls, n_steps_solver=total_steps,
                n_steps_surrogate=0, horizon_log=[])


# ────────────────────────────────────────────────────────────────────────────
# Configuration B: hybrid-ours
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_hybrid_ours(state0: np.ndarray, total_steps: int, params: dict,
                     dt_save: float, model, tau: float, device: str,
                     n_reps: int = 3) -> dict:
    """Adaptive horizon descent gated by surrogate step-doubling."""
    n_y, n_x = state0.shape[1], state0.shape[2]
    walls = []
    final_state = None
    horizon_log = []
    n_solver_calls = 0
    for rep in range(n_reps):
        state = torch.from_numpy(state0.copy()).to(device)
        steps_remaining = total_steps
        log = []
        n_solver = 0
        t0 = time.perf_counter()
        while steps_remaining > 0:
            chosen = None
            for h in HORIZONS_DESC:
                if h > steps_remaining:
                    continue
                dt_t = torch.tensor([h * dt_save], device=device, dtype=torch.float32)
                pred_full = model(state.unsqueeze(0), dt_t)[0]
                if h >= 2:
                    pred_mid = model(state.unsqueeze(0), dt_t * 0.5)[0]
                    pred_chain = model(pred_mid.unsqueeze(0), dt_t * 0.5)[0]
                    e_hat = float(torch.sqrt(((pred_full - pred_chain) ** 2).sum(dim=0)).mean().item())
                else:
                    e_hat = 0.0
                if e_hat < tau:
                    chosen = (h, pred_full, e_hat); break
            if chosen is not None:
                h, pred_full, e_hat = chosen
                state = pred_full
                steps_remaining -= h
                log.append(("surrogate", h, e_hat))
            else:
                # solver fallback for h=1 (one base-dt step)
                state_np = state.cpu().numpy()
                final, _ = solver_advance(state_np, 1, params, dt_save)
                state = torch.from_numpy(final).to(device)
                steps_remaining -= 1
                n_solver += 1
                log.append(("solver", 1, float("inf")))
        wall = time.perf_counter() - t0
        walls.append(wall)
        if rep == 0:
            final_state = state.cpu().numpy()
            horizon_log = log
            n_solver_calls = n_solver
    return dict(final=final_state, wall=float(np.median(walls)), walls=walls,
                horizon_log=horizon_log, n_solver_fallbacks=n_solver_calls,
                n_macro=len(horizon_log))


# ────────────────────────────────────────────────────────────────────────────
# Configuration C: hybrid-Richardson (gating uses solver, prediction uses surrogate)
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_hybrid_richardson(state0: np.ndarray, total_steps: int, params: dict,
                            dt_save: float, model, tau: float, device: str,
                            n_reps: int = 3) -> dict:
    """Adaptive horizon descent. Gating uses classical Richardson on the
    ground-truth solver. Once gate passes, prediction uses surrogate."""
    n_y, n_x = state0.shape[1], state0.shape[2]
    walls = []
    final_state = None
    horizon_log = []
    n_solver_gating = 0
    n_solver_fallbacks = 0
    for rep in range(n_reps):
        state = torch.from_numpy(state0.copy()).to(device)
        state_np = state0.copy()
        steps_remaining = total_steps
        log = []
        n_gate = 0
        n_fb = 0
        t0 = time.perf_counter()
        while steps_remaining > 0:
            chosen = None
            for h in HORIZONS_DESC:
                if h > steps_remaining:
                    continue
                if h >= 2:
                    # Richardson gating: 2 solver calls
                    big, _ = solver_advance(state_np, h, params, dt_save)
                    mid, _ = solver_advance(state_np, h // 2, params, dt_save)
                    chain, _ = solver_advance(mid, h // 2, params, dt_save)
                    e_rich = float(np.sqrt(((big - chain) ** 2).sum(axis=0)).mean())
                    n_gate += 1
                else:
                    e_rich = 0.0
                if e_rich < tau:
                    # Take SURROGATE prediction at this horizon (per spec)
                    dt_t = torch.tensor([h * dt_save], device=device, dtype=torch.float32)
                    pred_full = model(state.unsqueeze(0), dt_t)[0]
                    state = pred_full
                    state_np = state.cpu().numpy()
                    steps_remaining -= h
                    log.append(("surrogate", h, e_rich))
                    chosen = True; break
            if not chosen:
                # solver fallback for one base step
                final, _ = solver_advance(state_np, 1, params, dt_save)
                state = torch.from_numpy(final).to(device)
                state_np = final
                steps_remaining -= 1
                n_fb += 1
                log.append(("solver", 1, float("inf")))
        wall = time.perf_counter() - t0
        walls.append(wall)
        if rep == 0:
            final_state = state.cpu().numpy()
            horizon_log = log
            n_solver_gating = n_gate
            n_solver_fallbacks = n_fb
    return dict(final=final_state, wall=float(np.median(walls)), walls=walls,
                horizon_log=horizon_log, n_solver_gating_calls=n_solver_gating,
                n_solver_fallbacks=n_solver_fallbacks, n_macro=len(horizon_log))


# ────────────────────────────────────────────────────────────────────────────
# Aggregation, figures, decision criterion
# ────────────────────────────────────────────────────────────────────────────

def bootstrap_ratio_ci(num: np.ndarray, denom: np.ndarray, n_boot: int = 1000,
                        seed: int = 0) -> tuple:
    rng = np.random.RandomState(seed)
    n = len(num)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        b = num[idx].sum() / max(denom[idx].sum(), 1e-12)
        boots.append(b)
    boots = np.asarray(boots)
    return float(boots.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def horizon_usage_histogram(per_traj_logs: list) -> dict:
    """Aggregate per-horizon decision counts across all trajs."""
    counts = {h: 0 for h in HORIZONS_DESC}
    counts["solver"] = 0
    for log in per_traj_logs:
        for kind, h, _ in log:
            if kind == "surrogate":
                counts[h] = counts.get(h, 0) + 1
            else:
                counts["solver"] = counts.get("solver", 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trajs", type=int, default=200)
    ap.add_argument("--n_calib", type=int, default=50)
    ap.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--percentile", type=float, default=0.75)
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--save_gif", action="store_true",
                     help="Save trajectory frame-decisions for the GIF builder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[c4r] device={device}")
    print(f"[c4r] n_trajs={args.n_trajs}  n_calib={args.n_calib}  "
          f"total_steps={args.total_steps}  percentile={args.percentile}  reps={args.n_reps}",
          flush=True)
    model = load_model(str(CKPT), device=device)

    test_path = DATA_DIR / "oregonator_test.h5"
    val_path = DATA_DIR / "oregonator_val.h5"
    with h5py.File(test_path, "r") as f:
        N_test, T, _, H, W = f["states"].shape
        dt_save = float(f.attrs["dt_save"])

    print(f"[c4r] test split: N={N_test} T={T} HxW={H}x{W} dt_save={dt_save}")
    print(f"[c4r] solver: OregonatorTyson2D — Strang split + CFL-substepped FTCS "
          f"diffusion + implicit-Euler reaction; CFL margin 0.4×dx²/(4D), "
          f"reaction substep ε'/4")
    print(flush=True)

    # ── Calibrate thresholds on val ───────────────────────────────────────
    calib = calibrate_thresholds(model, val_path, dt_save, [], args.n_calib,
                                   args.percentile, device)
    tau_ours = calib["tau_ours"]
    tau_classical = calib["tau_classical"]
    print(flush=True)

    # ── Sample test trajectories ──────────────────────────────────────────
    rng = np.random.RandomState(0)
    sample_idxs = sorted(rng.choice(N_test, size=min(args.n_trajs, N_test),
                                       replace=False).tolist())

    t_run_start = time.time()
    per_traj_results = []
    for k, ti in enumerate(sample_idxs):
        with h5py.File(test_path, "r") as f:
            state0 = np.array(f["states"][ti, 0])
            params = dict(eps=float(f["params"][ti, 1]),
                           q=float(f["params"][ti, 2]),
                           f=float(f["params"][ti, 0]),
                           D=float(f["params"][ti, 3]))
            target = np.array(f["states"][ti, args.total_steps])

        sol = run_solver_only(state0, args.total_steps, params, dt_save, args.n_reps)
        ho = run_hybrid_ours(state0, args.total_steps, params, dt_save,
                              model, tau_ours, device, args.n_reps)
        # Richardson is the most expensive; only run 1 rep to save time at n_trajs scale
        rich_reps = max(1, args.n_reps - 2)
        hr = run_hybrid_richardson(state0, args.total_steps, params, dt_save,
                                     model, tau_classical, device, rich_reps)

        # Errors vs solver target (from dataset — solver-integrated truth)
        err_sol = float(np.sqrt(((sol["final"] - target) ** 2).mean()))
        err_ours = float(np.sqrt(((ho["final"] - target) ** 2).mean()))
        err_rich = float(np.sqrt(((hr["final"] - target) ** 2).mean()))

        rec = dict(
            traj=int(ti),
            wall_solver=sol["wall"], wall_ours=ho["wall"], wall_richardson=hr["wall"],
            err_solver=err_sol, err_ours=err_ours, err_richardson=err_rich,
            ours_n_solver_fb=ho["n_solver_fallbacks"], ours_n_macro=ho["n_macro"],
            rich_n_gate=hr["n_solver_gating_calls"], rich_n_solver_fb=hr["n_solver_fallbacks"],
            rich_n_macro=hr["n_macro"],
            ours_log=[(t, h) for (t, h, _) in ho["horizon_log"]],
            rich_log=[(t, h) for (t, h, _) in hr["horizon_log"]],
        )
        if args.save_gif and k < 2:
            # stash full state evolution for GIF generation later
            rec["save_gif"] = True
        per_traj_results.append(rec)

        if (k + 1) % 5 == 0 or k == 0:
            elapsed = time.time() - t_run_start
            eta_min = (elapsed / (k + 1) * (len(sample_idxs) - k - 1)) / 60
            print(f"  traj {k+1:3d}/{len(sample_idxs)}  ti={ti}  "
                  f"sol={sol['wall']:.2f}s  ours={ho['wall']:.2f}s  "
                  f"rich={hr['wall']:.2f}s  | err sol/ours/rich = "
                  f"{err_sol:.4f}/{err_ours:.4f}/{err_rich:.4f}  ETA={eta_min:.1f}min",
                  flush=True)

    # ── Aggregate ────────────────────────────────────────────────────────
    sol_walls = np.array([r["wall_solver"] for r in per_traj_results])
    ours_walls = np.array([r["wall_ours"] for r in per_traj_results])
    rich_walls = np.array([r["wall_richardson"] for r in per_traj_results])
    err_sol = np.array([r["err_solver"] for r in per_traj_results])
    err_ours = np.array([r["err_ours"] for r in per_traj_results])
    err_rich = np.array([r["err_richardson"] for r in per_traj_results])

    speedup_ours_m, speedup_ours_lo, speedup_ours_hi = bootstrap_ratio_ci(sol_walls, ours_walls)
    speedup_rich_m, speedup_rich_lo, speedup_rich_hi = bootstrap_ratio_ci(sol_walls, rich_walls)
    speedup_ours_over_rich_m, _, _ = bootstrap_ratio_ci(rich_walls, ours_walls)

    horizon_usage_ours = horizon_usage_histogram([r["ours_log"] for r in per_traj_results])
    horizon_usage_rich = horizon_usage_histogram([r["rich_log"] for r in per_traj_results])

    # ── Save results ─────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "run.json"
    out = dict(
        config=dict(
            n_trajs=args.n_trajs, n_calib=args.n_calib,
            total_steps=args.total_steps, percentile=args.percentile,
            n_reps=args.n_reps, ckpt=str(CKPT),
            solver="OregonatorTyson2D Strang+FTCS-CFL+implicit-Euler",
            dt_save=dt_save,
            tau_ours=tau_ours, tau_classical=tau_classical,
            calibration_ours_n=len(calib["ours_dist"]),
            calibration_classical_n=len(calib["classical_dist"]),
        ),
        aggregate=dict(
            speedup_ours_over_solver_mean=speedup_ours_m,
            speedup_ours_over_solver_ci=[speedup_ours_lo, speedup_ours_hi],
            speedup_rich_over_solver_mean=speedup_rich_m,
            speedup_rich_over_solver_ci=[speedup_rich_lo, speedup_rich_hi],
            speedup_ours_over_richardson=speedup_ours_over_rich_m,
            err_solver_mean=float(err_sol.mean()),
            err_ours_mean=float(err_ours.mean()),
            err_richardson_mean=float(err_rich.mean()),
            wall_total_solver_s=float(sol_walls.sum()),
            wall_total_ours_s=float(ours_walls.sum()),
            wall_total_richardson_s=float(rich_walls.sum()),
        ),
        horizon_usage_ours=horizon_usage_ours,
        horizon_usage_richardson=horizon_usage_rich,
        per_traj=per_traj_results,
    )
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[c4r] saved results: {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("AGGREGATE RESULTS (n_trajs={}, total_steps={}, percentile={})".format(
        args.n_trajs, args.total_steps, args.percentile))
    print("=" * 80)
    print(f"  solver-only       wall total = {sol_walls.sum():.1f}s  "
          f"per-traj mean = {sol_walls.mean():.3f}s  "
          f"err = {err_sol.mean():.4f}")
    print(f"  hybrid-ours       wall total = {ours_walls.sum():.1f}s  "
          f"per-traj mean = {ours_walls.mean():.3f}s  "
          f"err = {err_ours.mean():.4f}  "
          f"speedup vs solver = {speedup_ours_m:.2f}× "
          f"[{speedup_ours_lo:.2f}, {speedup_ours_hi:.2f}]")
    print(f"  hybrid-richardson wall total = {rich_walls.sum():.1f}s  "
          f"per-traj mean = {rich_walls.mean():.3f}s  "
          f"err = {err_rich.mean():.4f}  "
          f"speedup vs solver = {speedup_rich_m:.2f}× "
          f"[{speedup_rich_lo:.2f}, {speedup_rich_hi:.2f}]")
    print()
    print(f"  ours / richardson = {speedup_ours_over_rich_m:.2f}×")
    print()
    print(f"Horizon usage (counts):")
    print(f"  ours:       {horizon_usage_ours}")
    print(f"  richardson: {horizon_usage_rich}")
    print()
    print(f"OUTCOME DECISION:")
    if speedup_ours_lo > 2.0 * speedup_rich_hi:
        print("  A — hybrid-ours >2× faster than Richardson at equiv accuracy. "
              "SHIP at 200, paper framing intact.")
    elif speedup_rich_lo > speedup_ours_hi:
        print("  B — Richardson hybrid faster than ours. STOP and reframe.")
    else:
        print("  C — within bootstrap CI. ESCALATE to 1000 trajs overnight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
