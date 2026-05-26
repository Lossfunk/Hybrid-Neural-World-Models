#!/usr/bin/env python3
"""Mode evaluation framework — pure surrogate, spatial blend, and
surrogate+solver hybrid.

Three inference modes implemented for the same target rollout:

  Mode 1 — pure surrogate, single forward pass at horizon dt. FiLM-conditioned
           on continuous dt, supports arbitrary target horizons including
           non-trained values inside and outside the training range.
  Mode 2 — spatial blend: predict at h_big globally + h_small chain
           globally, per-cell take the prediction with lower local ê.
  Mode 3 — surrogate + solver hybrid: predict surrogate at h_big globally,
           where ê-map > τ run the solver locally with surrogate-value halo
           (halo BC = linear interp from s to f_surr). No-crop version of
           patched solver.

Note: a "Mode 2 adaptive chaining by step-doubling agreement" was tested
and found broken — the heuristic picks self-consistent but inaccurate
decompositions. FiLM extrapolation in Mode 1 already handles arbitrary T,
so chained decomposition is not needed.

Every mode is timed end-to-end (CPU) and evaluated against the dataset
ground truth (which is the production solver's integrated trajectory).

Outputs a unified JSON with rows: (split, mode, target_dt, traj) → wall,
n_calls, RMSE.

Usage:
  python eval_modes.py --ckpt path/best.pt --split test --n_trajs 10
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

from eval_utils import load_model, predict, step_doubling_estimator   # noqa: E402
from oregonator2d_tyson import OregonatorTyson2D, TysonParams         # noqa: E402

DT_BASE = 0.05
TRAINED_ATOMS = [1, 2, 4, 8, 16, 32, 64]                # in units of dt_base


# ─────────────────────────────────────────────────────────────────────────
#  Mode 1 — pure surrogate single pass
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mode1_pure_surrogate(model, s_init: torch.Tensor, target_dt: float
                          ) -> dict:
    """Single forward pass at target_dt (continuous, FiLM-conditioned)."""
    if s_init.dim() == 3:
        s_init = s_init.unsqueeze(0)
    t0 = time.time()
    pred = predict(model, s_init, target_dt)[0]      # (C, H, W)
    wall = time.time() - t0
    return {
        "method": "mode1",
        "final_state": pred.detach().cpu().numpy(),
        "wall_time_s": wall,
        "n_surrogate_calls": 1,
        "n_solver_calls": 0,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Mode 2 — adaptive chaining
# ─────────────────────────────────────────────────────────────────────────
#  Mode 2 — spatial blend (was Mode 3 in earlier internal numbering)
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mode2_spatial_blend(model, s_init: torch.Tensor, target_dt: float,
                         h_small_atom: int = 16,
                         dt_base: float = DT_BASE) -> dict:
    """Predict at h_big = target_dt and at h_small = h_small_atom × dt_base
    chain (whose total = target_dt). Compute ê-map at h_big. Per-cell:
      - if ê(x,y) < median(ê): take the h_big prediction at that cell
      - else: take the h_small chain prediction at that cell

    h_small_atom must divide the target_dt cleanly, else falls back.
    """
    if s_init.dim() == 3:
        s_init = s_init.unsqueeze(0)
    target_steps = int(round(target_dt / dt_base))
    # Pick the largest atom that divides target_steps AND is strictly less
    # than target_steps (so the chain has ≥ 2 legs and the blend is non-trivial).
    if h_small_atom >= target_steps or target_steps % h_small_atom != 0:
        h_small_atom = next(
            (a for a in [32, 16, 8, 4, 2, 1] if a < target_steps and target_steps % a == 0),
            1,
        )

    t0 = time.time()
    # h_big single pass
    pred_big = predict(model, s_init, target_dt)         # (1, C, H, W)
    # h_small chain: target_steps / h_small_atom legs at dt_small
    n_legs = target_steps // h_small_atom
    dt_small = h_small_atom * dt_base
    s_chain = s_init.clone()
    for _ in range(n_legs):
        s_chain = predict(model, s_chain, dt_small)
    pred_small = s_chain                                   # (1, C, H, W)
    # ê-map at h_big (one extra step-doubling probe)
    e_hat_map, _ = step_doubling_estimator(model, s_init, target_dt)
    # Per-cell mask: high-ê gets the chain prediction
    thr = e_hat_map.median().item()
    mask = (e_hat_map > thr).unsqueeze(1)                  # (1, 1, H, W)
    blended = torch.where(mask, pred_small, pred_big)[0]   # (C, H, W)
    wall = time.time() - t0
    n_calls = 1 + n_legs + 2     # h_big + n_legs + step-doubling pair
    return {
        "method": "mode2",
        "final_state": blended.detach().cpu().numpy(),
        "wall_time_s": wall,
        "n_surrogate_calls": n_calls,
        "n_solver_calls": 0,
        "h_small_atom": h_small_atom,
        "n_legs_small": n_legs,
        "e_hat_threshold": thr,
        "high_e_fraction": float(mask.float().mean().item()),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Mode 3 — surrogate + solver spatial hybrid (no-crop simple version)
# ─────────────────────────────────────────────────────────────────────────
def _dilate_mask(mask: np.ndarray, n: int = 1) -> np.ndarray:
    """Binary dilation by n cells (simple iterative shift)."""
    out = mask.copy()
    for _ in range(n):
        out = (out
                | np.roll(out, 1, axis=0) | np.roll(out, -1, axis=0)
                | np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1))
    return out


def _patched_diffusion_step(u: np.ndarray, dt: float, dx: float, D: float,
                              halo_start: np.ndarray, halo_end: np.ndarray,
                              halo_mask: np.ndarray,
                              fixed_substep: bool = False) -> np.ndarray:
    """Diffusion step on active cells (~halo_mask), with halo cells held at
    a linearly-interpolated value: halo(t') = halo_start + (t'/dt)·(halo_end
    − halo_start). This is Option B from the design — the halo is
    time-consistent with the active region.
    """
    if D <= 0.0:
        return u
    if fixed_substep:
        n_sub = 1
    else:
        dt_max = 0.4 * dx * dx / (4.0 * D)
        n_sub = max(1, int(np.ceil(dt / dt_max)))
    dt_sub = dt / n_sub
    for k in range(n_sub):
        # halo at the midpoint of this substep (most accurate explicit choice)
        frac = (k + 0.5) / n_sub
        halo_now = halo_start + frac * (halo_end - halo_start)
        u_for_lapl = np.where(halo_mask, halo_now, u)
        lapl = (np.roll(u_for_lapl, 1, axis=0) + np.roll(u_for_lapl, -1, axis=0)
                + np.roll(u_for_lapl, 1, axis=1) + np.roll(u_for_lapl, -1, axis=1)
                - 4.0 * u_for_lapl) / (dx * dx)
        u = np.where(halo_mask, u, u + dt_sub * D * lapl)
    return u


def _reaction_step_local(u: np.ndarray, v: np.ndarray, dt: float,
                          eps: float, q: float, f: float,
                          active_mask: np.ndarray) -> tuple:
    """Implicit-Euler reaction step applied only to active cells.

    Mirrors the structure of OregonatorTyson2D._reaction_step but
    vectorised over only the active region. Halo cells unchanged.
    """
    if active_mask.sum() == 0:
        return u, v
    # Newton iteration on F(u_n+1) = u_n+1 − u_n − dt·R(u_n+1) = 0
    # Closed-form 2×2 Jacobian of R = [(1/ε)(u(1−u) − f·v·(u−q)/(u+q)), u−v]
    flat_active = active_mask.ravel()
    u_n = u.copy()
    v_n = v.copy()
    u_curr = u_n.ravel()[flat_active].copy()
    v_curr = v_n.ravel()[flat_active].copy()
    u_old = u_curr.copy()
    v_old = v_curr.copy()
    eps_safe = max(eps, 1e-12)
    for _ in range(8):
        # Guard against Newton excursions that leave the physical manifold
        # (mirrors OregonatorTyson2D._reaction_step's per-iter clipping).
        np.clip(u_curr, 0.0, 1.5, out=u_curr)
        np.clip(v_curr, 0.0, 2.0, out=v_curr)
        # R(u, v)
        denom = u_curr + q
        denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        ru = (1.0 / eps_safe) * (u_curr * (1.0 - u_curr)
                                  - f * v_curr * (u_curr - q) / denom)
        rv = u_curr - v_curr
        # F = u_curr − u_old − dt·R
        F1 = u_curr - u_old - dt * ru
        F2 = v_curr - v_old - dt * rv
        # Jacobian dR/d(u, v)
        dru_du = (1.0 / eps_safe) * (
            (1.0 - 2.0 * u_curr)
            - f * v_curr * 2.0 * q / denom ** 2
        )
        dru_dv = -(1.0 / eps_safe) * f * (u_curr - q) / denom
        drv_du = np.ones_like(u_curr)
        drv_dv = -np.ones_like(u_curr)
        # dF/du, dF/dv
        J11 = 1.0 - dt * dru_du
        J12 = -dt * dru_dv
        J21 = -dt * drv_du
        J22 = 1.0 - dt * drv_dv
        det = J11 * J22 - J12 * J21
        det = np.where(np.abs(det) < 1e-15, 1e-15, det)
        du = (-F1 * J22 + F2 * J12) / det
        dv = (-F2 * J11 + F1 * J21) / det
        u_curr = u_curr + du
        v_curr = v_curr + dv
        if np.max(np.abs(du)) + np.max(np.abs(dv)) < 1e-10:
            break
    np.clip(u_curr, 0.0, 1.5, out=u_curr)
    np.clip(v_curr, 0.0, 2.0, out=v_curr)
    u_flat = u_n.ravel().copy()
    v_flat = v_n.ravel().copy()
    u_flat[flat_active] = u_curr
    v_flat[flat_active] = v_curr
    return u_flat.reshape(u.shape), v_flat.reshape(v.shape)


def _solver_patched_step(s_init_np: np.ndarray, halo_end_np: np.ndarray,
                          halo_mask: np.ndarray, dt: float, params: dict,
                          dx: float = 100.0 / 256,
                          dt_save: float = DT_BASE) -> np.ndarray:
    """Run patched Strang split on the patch (active = ~halo_mask) for
    physical time `dt`, sub-stepping at dt_save granularity to match the
    production solver's tested regime.

    Halo follows linear interpolation from s_init (at t=0) to halo_end
    (at t=dt). At each dt_save substep, halo is the interpolated value at
    the START of that substep (used as the diffusion stencil's Dirichlet
    BC during the substep).

    s_init_np / halo_end_np: (C=2, H, W); halo_mask: (H, W) bool.
    Returns (C, H, W); halo cells in output match halo_end_np.
    """
    u = s_init_np[0].astype(np.float64).copy()
    v = s_init_np[1].astype(np.float64).copy()
    u_start = s_init_np[0].astype(np.float64)
    u_end = halo_end_np[0].astype(np.float64)
    active_mask = ~halo_mask
    n_sub = max(1, int(round(dt / dt_save)))
    dt_sub = dt / n_sub
    for k in range(n_sub):
        # halo at the START and END of this substep — pass both to the
        # diffusion routine (it midpoint-interpolates internally).
        frac_start = k / n_sub
        frac_end = (k + 1) / n_sub
        u_halo_start = u_start + frac_start * (u_end - u_start)
        u_halo_end = u_start + frac_end * (u_end - u_start)
        # Strang: react(dt/2) → diffuse(dt) → react(dt/2)
        u, v = _reaction_step_local(
            u, v, dt_sub * 0.5,
            params["eps"], params["q"], params["f"], active_mask,
        )
        u = _patched_diffusion_step(
            u, dt_sub, dx, params["D"],
            halo_start=u_halo_start, halo_end=u_halo_end,
            halo_mask=halo_mask,
        )
        u, v = _reaction_step_local(
            u, v, dt_sub * 0.5,
            params["eps"], params["q"], params["f"], active_mask,
        )
    out = np.stack([u, v], axis=0).astype(np.float32)
    # Halo cells in the output: surrogate prediction (we chose not to refine).
    out[:, halo_mask] = halo_end_np[:, halo_mask].astype(np.float32)
    return out


@torch.no_grad()
def mode3_solver_hybrid(model, s_init: torch.Tensor, target_dt: float,
                          params: dict, tau_local: float | None = None,
                          dt_base: float = DT_BASE,
                          dilate_n: int = 1) -> dict:
    """Surrogate at target_dt, then solver patches where ê-map lights up.

    tau_local: per-cell threshold on ê. If None, defaults to median(ê-map).
    dilate_n: halo width (cells) added around active mask. 1 = stencil radius.
    """
    if s_init.dim() == 3:
        s_init = s_init.unsqueeze(0)

    t0 = time.time()
    e_hat_map, pred_full = step_doubling_estimator(model, s_init, target_dt)
    e_hat_np = e_hat_map[0].detach().cpu().numpy()
    pred_full_np = pred_full[0].detach().cpu().numpy()
    s_init_np = s_init[0].detach().cpu().numpy()

    if tau_local is None:
        tau_local = float(np.quantile(e_hat_np, 0.75))
    high_e_mask = e_hat_np > tau_local
    # Active (patch interior) = high-ê cells; halo = NOT in active region
    # but neighbors of active cells (so solver stencil reads valid values).
    # Simpler representation: active_mask is the CELLS we update; everything
    # else is halo. For halo BC we use the surrogate's prediction (Option B
    # would interpolate from s_init to pred_full; for the simple version we
    # use pred_full constant).
    active_mask = _dilate_mask(high_e_mask, dilate_n)
    halo_mask = ~active_mask

    # Run patched solver on active region; halo linearly interpolates from
    # s_init (at t=0) to pred_full (at t=dt) — Option B.
    out = _solver_patched_step(
        s_init_np=s_init_np,
        halo_end_np=pred_full_np,
        halo_mask=halo_mask,
        dt=target_dt,
        params=params,
    )
    wall = time.time() - t0
    return {
        "method": "mode3",
        "final_state": out,
        "wall_time_s": wall,
        "n_surrogate_calls": 3,   # 1 full + 2 step-doubling
        "n_solver_calls": 1,       # one solver step on the patch
        "tau_local": tau_local,
        "active_fraction": float(active_mask.mean()),
        "high_e_fraction": float(high_e_mask.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Mode 3X — trajectory-level mode selector (Option X)
#
#  The "spatial patching" Mode 3 has a fundamental halo-BC drift problem at
#  long T (linear-interp halo can't track non-linear evolution; errors leak
#  into the active region from the halo). Option X side-steps this by
#  switching at the trajectory level instead of per-cell:
#    1. Compute ê-map and pred_full (3 surrogate calls).
#    2. If ê.mean() < tau_global → trust the surrogate; return pred_full.
#    3. Else → run the FULL solver from s_init for time target_dt.
#  Strict superset of Mode 1 in accuracy (the only thing it does extra is
#  fall back to the exact solver when the trust signal is high).
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mode3x_traj_select(model, s_init: torch.Tensor, target_dt: float,
                         params: dict, tau_global: float,
                         dt_base: float = DT_BASE) -> dict:
    """Trajectory-level: surrogate when global ê is low, full solver otherwise.

    tau_global: threshold on ê.mean() (mean over all cells × channels). When
        the mean predicted error is below this, we accept the surrogate
        prediction. When above, we run the production solver for accuracy.
    """
    if s_init.dim() == 3:
        s_init = s_init.unsqueeze(0)

    t0 = time.time()
    e_hat_map, pred_full = step_doubling_estimator(model, s_init, target_dt)
    e_hat_mean = float(e_hat_map.mean().item())
    used_solver = e_hat_mean >= tau_global

    if not used_solver:
        out = pred_full[0].detach().cpu().numpy()
        n_solver_calls = 0
    else:
        s_init_np = s_init[0].detach().cpu().numpy()
        sim = OregonatorTyson2D(
            n_x=s_init_np.shape[2], n_y=s_init_np.shape[1],
            L_x=100.0, L_y=100.0,
            params=TysonParams(**params),
        )
        sim.u[:] = s_init_np[0]
        sim.v[:] = s_init_np[1]
        sim.t_sim = 0.0
        n_solver_calls = int(round(target_dt / dt_base))
        for _ in range(n_solver_calls):
            sim.step(dt_base)
        out = sim.saved_state().astype(np.float32)
    wall = time.time() - t0

    return {
        "method": "mode3x",
        "final_state": out,
        "wall_time_s": wall,
        "n_surrogate_calls": 3,
        "n_solver_calls": n_solver_calls,
        "tau_global": tau_global,
        "e_hat_mean": e_hat_mean,
        "used_solver": used_solver,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Solver baseline (full grid) for comparison
# ─────────────────────────────────────────────────────────────────────────
def solver_full(s_init_np: np.ndarray, target_dt: float, params: dict,
                  dt_save: float = DT_BASE) -> dict:
    """Run the production solver from s_init for target_dt physical time.
    Returns final_state + wall_time_s. Uses dt_save-step granularity to
    match the dataset's integration step pattern."""
    sim = OregonatorTyson2D(
        n_x=s_init_np.shape[2], n_y=s_init_np.shape[1],
        L_x=100.0, L_y=100.0,
        params=TysonParams(**params),
    )
    sim.u[:] = s_init_np[0]
    sim.v[:] = s_init_np[1]
    sim.t_sim = 0.0
    n_steps = int(round(target_dt / dt_save))
    t0 = time.time()
    for _ in range(n_steps):
        sim.step(dt_save)
    wall = time.time() - t0
    out = sim.saved_state().astype(np.float32)
    return {
        "method": "solver_full",
        "final_state": out,
        "wall_time_s": wall,
        "n_surrogate_calls": 0,
        "n_solver_calls": n_steps,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def run_one_traj(model, ds_path: str, traj_idx: int, target_dt: float,
                   target_steps: int, mode_names: list, device: str = "cpu"
                   ) -> list:
    """For a single (trajectory, target_dt) pair, run all requested modes
    starting from frame 0, against ground-truth target_steps frames forward."""
    with h5py.File(ds_path, "r") as f:
        s_init = np.array(f["states"][traj_idx, 0], dtype=np.float32)
        target = np.array(f["states"][traj_idx, target_steps], dtype=np.float32)
        params = dict(
            f=float(f["params"][traj_idx, 0]),
            eps=float(f["params"][traj_idx, 1]),
            q=float(f["params"][traj_idx, 2]),
            D=float(f["params"][traj_idx, 3]),
        )
    s_init_t = torch.from_numpy(s_init).to(device)

    out_rows = []
    for mode_name in mode_names:
        if mode_name == "mode1":
            r = mode1_pure_surrogate(model, s_init_t, target_dt)
        elif mode_name == "mode2":
            r = mode2_spatial_blend(model, s_init_t, target_dt)
        elif mode_name == "mode3":
            r = mode3_solver_hybrid(model, s_init_t, target_dt, params)
        elif mode_name == "solver":
            r = solver_full(s_init, target_dt, params)
        else:
            raise ValueError(f"unknown mode {mode_name}")
        r["rmse_vs_gt"] = rmse(r["final_state"], target)
        r["traj_idx"] = traj_idx
        r["target_dt"] = target_dt
        r["target_steps"] = target_steps
        # Drop the (large) state arrays from the returned record before
        # saving — keep them only if explicit caller wants visualizations
        r.pop("final_state", None)
        out_rows.append(r)
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test",
                     help="test | ood_near | ood_far")
    ap.add_argument("--n_trajs", type=int, default=10)
    ap.add_argument("--target_steps_list", default="32,64,128",
                     help="comma-separated h values (× dt_save = target_dt)")
    ap.add_argument("--modes", default="solver,mode1,mode2,mode3")
    ap.add_argument("--device", default="cpu",
                     help="cpu (fair vs solver) or cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)

    ds_path = ROOT / "data" / "oregonator" / f"oregonator_{args.split}.h5"
    target_steps_list = [int(x) for x in args.target_steps_list.split(",")]
    mode_names = args.modes.split(",")

    print(f"[modes] split={args.split}  n_trajs={args.n_trajs}  "
          f"targets={target_steps_list}  modes={mode_names}  "
          f"device={args.device}", flush=True)

    model = load_model(args.ckpt, device=args.device)

    with h5py.File(ds_path, "r") as f:
        N = f["states"].shape[0]
    rng = np.random.RandomState(args.seed)
    sample_idxs = rng.choice(N, size=min(args.n_trajs, N), replace=False).tolist()
    print(f"[modes] traj indices: {sample_idxs}", flush=True)

    rows = []
    for ti in sample_idxs:
        for h_steps in target_steps_list:
            target_dt = h_steps * DT_BASE
            print(f"  traj {ti:4d}  target_steps={h_steps}  "
                  f"target_dt={target_dt:.2f}", flush=True)
            traj_rows = run_one_traj(model, str(ds_path), ti, target_dt,
                                       h_steps, mode_names, args.device)
            for r in traj_rows:
                msg = (f"    {r['method']:15s}  "
                        f"wall={r['wall_time_s']:6.3f}s  "
                        f"n_surr={r['n_surrogate_calls']:>3}  "
                        f"n_solver={r['n_solver_calls']:>3}  "
                        f"RMSE={r['rmse_vs_gt']:.4f}")
                print(msg, flush=True)
            rows.extend(traj_rows)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"modes_{args.split}.json"
    out_path.write_text(json.dumps(
        {"split": args.split, "n_trajs": len(sample_idxs),
         "target_steps_list": target_steps_list,
         "modes": mode_names, "rows": rows}, indent=2))
    print(f"\n[modes] results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
