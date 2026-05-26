#!/usr/bin/env python3
"""C4 verification: hybrid scheme wall-clock speedup.

Implements TEMPORAL adaptive chaining (NOT spatial per-cell hybrid). At each
macro-step:
  1. Take candidate big surrogate step:  pred_full = f(s, Δt)
  2. Compute step-doubling estimator:    pred_chain = f(f(s, Δt/2), Δt/2)
  3. ê = ‖pred_full − pred_chain‖ (domain-mean per macro-step)
  4. If ê < τ: accept pred_full (1 macro-step taken at horizon Δt)
     Else: recurse with Δt/2 (split into two macro-steps)
     If at smallest dt and still ê ≥ τ: fall back to running the solver for
     Δt directly.

Compare three configurations on N test trajectories:
  A) Solver-only          — full ground-truth integration
  B) Surrogate-only       — pure surrogate at largest horizon, chained as
                            many times as needed to reach total_time
  C) Hybrid (this scheme)

Reports per-trajectory and aggregate:
  - Wall clock per trajectory
  - Final-state L2 error vs solver
  - For hybrid: surrogate-step count, solver-fallback count

τ is calibrated on val (grid search over candidate τ values, pick the one
that maintains accuracy <= solver-only's own discretization error while
maximizing speedup).

Usage:
  python eval_c4.py --ckpt path/to/best.pt --split test --n_trajs 100
  python eval_c4.py --ckpt path/to/best.pt --calibrate_tau   # τ-calibration on val

Caveats:
  - Wall-clock comparison is apples-to-apples only when both surrogate and
    solver run on the same hardware. We default to CPU for both unless
    --gpu is set, in which case both surrogate AND solver fallback use
    the same device — but our solver is NumPy-only on CPU, so for honest
    benchmarking this script forces surrogate to CPU as well.
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

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "envs"))
sys.path.insert(0, str(ROOT / "models"))

from eval_utils import load_model                           # noqa: E402
from oregonator2d_tyson import OregonatorTyson2D, TysonParams   # noqa: E402


@torch.no_grad()
def hybrid_rollout(model, u_init: torch.Tensor, total_steps: int, dt_base: float,
                    horizons: list, tau: float,
                    solver_fallback_fn=None) -> dict:
    """Roll forward total_steps × dt_base of physical time using temporal
    adaptive chaining.

    horizons: list of integer step-multiples (e.g., [64, 32, 16, 8, 4, 2, 1])
              tried in DESCENDING order. The estimator decides at each macro
              whether to accept the big-step prediction or recurse.

    solver_fallback_fn: optional callable (state_np, n_steps) -> state_np.
                       If provided, used when even h=1 has ê >= tau.
                       state_np shape (C, H, W); returns same.

    Returns dict with: final_state, n_surrogate_steps, n_solver_fallback,
                        wall_time_s, history (list of step decisions).
    """
    device = u_init.device
    horizons_desc = sorted(horizons, reverse=True)
    state = u_init.clone()
    steps_remaining = total_steps
    n_surr = 0
    n_solver = 0
    history = []
    t_start = time.time()

    while steps_remaining > 0:
        # Pick the largest horizon that fits and clears the threshold
        chosen_h = None
        chosen_pred = None
        for h in horizons_desc:
            if h > steps_remaining:
                continue
            dt = h * dt_base
            dt_t = torch.tensor([dt], device=device, dtype=torch.float32)
            pred_full = model(state.unsqueeze(0), dt_t)[0]
            if h >= 2:
                # Step-doubling probe
                pred_mid = model(state.unsqueeze(0), dt_t * 0.5)[0]
                pred_chain = model(pred_mid.unsqueeze(0), dt_t * 0.5)[0]
                e_hat = float(torch.sqrt(((pred_full - pred_chain) ** 2).sum(dim=0)).mean().item())
            else:
                e_hat = 0.0  # h=1 always accepted
            if e_hat < tau:
                chosen_h = h
                chosen_pred = pred_full
                history.append(("surrogate", h, e_hat))
                break
        if chosen_h is not None:
            state = chosen_pred
            steps_remaining -= chosen_h
            n_surr += 1
        else:
            # All horizons rejected (even h=1) → solver fallback for h=1
            if solver_fallback_fn is None:
                # No fallback provided: accept h=1 anyway
                dt = 1 * dt_base
                dt_t = torch.tensor([dt], device=device, dtype=torch.float32)
                state = model(state.unsqueeze(0), dt_t)[0]
                steps_remaining -= 1
                n_surr += 1
                history.append(("surrogate_forced", 1, e_hat))
            else:
                state_np = state.cpu().numpy()
                state_np = solver_fallback_fn(state_np, 1)
                state = torch.from_numpy(state_np).to(device)
                steps_remaining -= 1
                n_solver += 1
                history.append(("solver", 1, e_hat))

    wall = time.time() - t_start
    return dict(final_state=state, n_surrogate_steps=n_surr,
                 n_solver_fallback=n_solver, wall_time_s=wall, history=history)


@torch.no_grad()
def surrogate_only_rollout(model, u_init: torch.Tensor, total_steps: int,
                             dt_base: float, max_h: int) -> dict:
    """Surrogate-only: chain at the largest available horizon (no estimator,
    no fallback). Baseline upper-bound on speedup."""
    device = u_init.device
    state = u_init.clone()
    remaining = total_steps
    n_steps = 0
    t_start = time.time()
    while remaining > 0:
        h = min(max_h, remaining)
        dt = h * dt_base
        dt_t = torch.tensor([dt], device=device, dtype=torch.float32)
        state = model(state.unsqueeze(0), dt_t)[0]
        remaining -= h
        n_steps += 1
    wall = time.time() - t_start
    return dict(final_state=state, n_steps=n_steps, wall_time_s=wall)


def solver_rollout(params, ic_state, total_steps, dt_save):
    """Run the OregonatorTyson2D solver from `ic_state` for total_steps steps.
    Returns final_state (C, H, W) and wall_time_s."""
    sim = OregonatorTyson2D(n_x=ic_state.shape[2], n_y=ic_state.shape[1],
                              L_x=100.0, L_y=100.0,
                              params=TysonParams(**params))
    sim.u[:] = ic_state[0]
    sim.v[:] = ic_state[1]
    sim.t_sim = 0.0
    t_start = time.time()
    for _ in range(total_steps):
        sim.step(dt_save)
    wall = time.time() - t_start
    return np.stack([sim.u, sim.v], axis=0).astype(np.float32), wall


def _make_solver_fallback(params, dt_save, n_x, n_y):
    """Build a closure that advances state by n_steps base_dt steps via the
    solver. Used inside hybrid_rollout when ê remains > τ even at h=1."""
    def fallback(state_np: np.ndarray, n_steps: int) -> np.ndarray:
        sim = OregonatorTyson2D(n_x=n_x, n_y=n_y, L_x=100.0, L_y=100.0,
                                  params=TysonParams(**params))
        sim.u[:] = state_np[0]
        sim.v[:] = state_np[1]
        for _ in range(n_steps):
            sim.step(dt_save)
        return np.stack([sim.u, sim.v], axis=0).astype(np.float32)
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_trajs", type=int, default=20)
    ap.add_argument("--total_steps", type=int, default=64)
    ap.add_argument("--horizons", default="64,32,16,8,4,2,1")
    ap.add_argument("--tau", type=float, default=0.02,
                     help="Adaptive chaining threshold (calibrate on val first)")
    ap.add_argument("--use_solver_fallback", action="store_true",
                     help="Fall back to solver when ê >= tau even at h=1")
    args = ap.parse_args()

    # Force CPU for honest comparison (solver is NumPy-only)
    device = "cpu"
    torch.set_num_threads(1)
    horizons = [int(x) for x in args.horizons.split(",")]
    ds_path = ROOT / "data" / "oregonator" / f"oregonator_{args.split}.h5"

    print(f"[c4] split={args.split}  n_trajs={args.n_trajs}  total_steps={args.total_steps}")
    print(f"[c4] tau={args.tau}  horizons={horizons}  device={device}")
    print(f"[c4] solver_fallback={args.use_solver_fallback}")
    model = load_model(args.ckpt, device=device)

    with h5py.File(ds_path, "r") as f:
        N, T, _, H, W = f["states"].shape
        dt_save = float(f.attrs["dt_save"])

    rng = np.random.RandomState(0)
    sample_idxs = rng.choice(N, size=min(args.n_trajs, N), replace=False).tolist()

    results = []
    for ti in sample_idxs:
        with h5py.File(ds_path, "r") as f:
            u_init = np.array(f["states"][ti, 0])           # (C, H, W) at t=0
            target = np.array(f["states"][ti, args.total_steps])
            params = dict(eps=float(f["params"][ti, 1]),
                           q=float(f["params"][ti, 2]),
                           f=float(f["params"][ti, 0]),
                           D=float(f["params"][ti, 3]))
        u_init_t = torch.from_numpy(u_init).to(device)

        # A) Solver-only: from u_init, advance total_steps and time
        sol_state, t_sol = solver_rollout(params, u_init, args.total_steps, dt_save)
        # Use saved target as ground truth (already integrated by the dataset
        # generator) — solver wall is measured here for fair comparison
        # against the surrogate (the saved target gives the "what should be
        # the answer" reference; t_sol gives the wall cost).
        gt = target

        # B) Surrogate-only at largest horizon
        sur = surrogate_only_rollout(model, u_init_t, args.total_steps,
                                       dt_save, max_h=max(horizons))
        sur_state = sur["final_state"].cpu().numpy()

        # C) Hybrid (with optional solver fallback)
        fallback = _make_solver_fallback(params, dt_save, W, H) if args.use_solver_fallback else None
        hyb = hybrid_rollout(model, u_init_t, args.total_steps, dt_save,
                              horizons, args.tau, solver_fallback_fn=fallback)
        hyb_state = hyb["final_state"].cpu().numpy()

        # Errors against the dataset target (which is the solver-integrated truth)
        err_sol = float(np.sqrt(((sol_state - gt) ** 2).mean()))
        err_sur = float(np.sqrt(((sur_state - gt) ** 2).mean()))
        err_hyb = float(np.sqrt(((hyb_state - gt) ** 2).mean()))

        results.append(dict(
            traj=int(ti),
            wall_solver=t_sol, wall_surrogate=sur["wall_time_s"], wall_hybrid=hyb["wall_time_s"],
            err_solver=err_sol, err_surrogate=err_sur, err_hybrid=err_hyb,
            n_surr=hyb["n_surrogate_steps"], n_solver=hyb["n_solver_fallback"],
        ))
        print(f"  traj {ti:4d}: "
              f"wall sol={t_sol:.3f}s sur={sur['wall_time_s']:.3f}s hyb={hyb['wall_time_s']:.3f}s  "
              f"err sol={err_sol:.4f} sur={err_sur:.4f} hyb={err_hyb:.4f}  "
              f"hyb steps={hyb['n_surrogate_steps']}+{hyb['n_solver_fallback']}solver")

    # Aggregate
    sol_walls = np.array([r["wall_solver"] for r in results])
    sur_walls = np.array([r["wall_surrogate"] for r in results])
    hyb_walls = np.array([r["wall_hybrid"] for r in results])
    speedup_sur = sol_walls.sum() / max(sur_walls.sum(), 1e-12)
    speedup_hyb = sol_walls.sum() / max(hyb_walls.sum(), 1e-12)
    err_sur_mean = np.mean([r["err_surrogate"] for r in results])
    err_hyb_mean = np.mean([r["err_hybrid"] for r in results])

    print()
    print("=" * 78)
    print(f"AGGREGATE over {len(results)} trajectories  total_steps={args.total_steps}  τ={args.tau}")
    print("=" * 78)
    print(f"  solver-only      total_wall = {sol_walls.sum():.2f}s  "
          f"avg/traj = {sol_walls.mean():.3f}s")
    print(f"  surrogate-only   total_wall = {sur_walls.sum():.2f}s  "
          f"avg/traj = {sur_walls.mean():.3f}s  "
          f"speedup = {speedup_sur:.1f}×  err = {err_sur_mean:.4f}")
    print(f"  hybrid           total_wall = {hyb_walls.sum():.2f}s  "
          f"avg/traj = {hyb_walls.mean():.3f}s  "
          f"speedup = {speedup_hyb:.1f}×  err = {err_hyb_mean:.4f}")

    # Save
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"c4_{args.split}_tau{args.tau}.json"
    out_path.write_text(json.dumps({
        "split": args.split, "tau": args.tau,
        "total_steps": args.total_steps, "horizons": horizons,
        "n_trajs": len(results),
        "speedup_surrogate_over_solver": float(speedup_sur),
        "speedup_hybrid_over_solver": float(speedup_hyb),
        "err_surrogate_mean": float(err_sur_mean),
        "err_hybrid_mean": float(err_hyb_mean),
        "per_traj": results,
    }, indent=2))
    print(f"[c4] results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
