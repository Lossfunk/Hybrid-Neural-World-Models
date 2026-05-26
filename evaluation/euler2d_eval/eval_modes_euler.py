#!/usr/bin/env python3
"""Mode 1 + Mode 2 (trust-aware fallback) for Euler v2.

Two modes:
  Mode 1: pure surrogate, single FiLM forward pass.
  Mode 2: compute step-doubling ê.mean. If below tau_global → return surrogate.
          If above → run the production HLL solver (Euler v2 fixed-dt).

Fall-back solver: euler2d_v2.Env._step_fixed_dt loop, used by the data
generator. Identical to the GT integrator.

Usage:
  python eval_modes_euler.py --ckpt path/best.pt --split test --n_trajs 8
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
NEURIPS = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "data_generation" / "euler2d"))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))
sys.path.insert(0, str(ROOT / "data_generation" / "euler2d"))

from eval_utils_euler import (load_model, predict, step_doubling_estimator,    # noqa: E402
                                  GAMMA)
import euler2d_v2 as env_mod                                # noqa: E402
from data_utils_2d import Euler2DDataset                    # noqa: E402
from config import load_config                              # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mode1_pure_surrogate(model, s_init: torch.Tensor, target_dt: float) -> dict:
    if s_init.dim() == 3:
        s_init = s_init.unsqueeze(0)
    t0 = time.time()
    pred = predict(model, s_init, target_dt)[0]
    wall = time.time() - t0
    return {
        "method": "mode1",
        "final_state": pred.detach().cpu().numpy(),
        "wall_time_s": wall,
        "n_surrogate_calls": 1,
        "n_solver_calls": 0,
    }


# ─────────────────────────────────────────────────────────────────────────
def solver_full_euler(s_init_np: np.ndarray, target_dt: float,
                       env_cfg: dict, base_dt: float = 0.002) -> dict:
    """Run the production HLL solver from s_init for target_dt physical time.
    Uses fixed-dt sub-stepping at base_dt granularity (matches GT generator).
    Returns final_state in the same (4, H, W) shape."""
    p = env_cfg["solver"]["params"]
    nx, ny = p["grid"][0], p["grid"][1]
    domain = tuple(p["domain"])
    gamma = float(p["gamma"])
    cfl = float(p["cfl"])

    # Convert (4, H, W) -> (H, W, 4) for euler2d_v2 internal layout
    q = np.transpose(s_init_np, (1, 2, 0)).astype(np.float64)
    dx = (domain[1] - domain[0]) / nx
    dy = (domain[3] - domain[2]) / ny

    # Build a thin wrapper Env for _step_fixed_dt
    env_obj = env_mod.Env({
        "env": {
            "solver": {"params": p},
            "base_dt": base_dt,
            "event_detector": {"params": {"threshold": 0.05}},
            "trajectory_length": int(round(target_dt / base_dt)),
        }
    })
    env_obj._q = q
    env_obj._dx = dx
    env_obj._dy = dy

    n_steps = int(round(target_dt / base_dt))
    t0 = time.time()
    env_obj._step_fixed_dt(target_dt)
    wall = time.time() - t0
    out = np.transpose(env_obj._q, (2, 0, 1)).astype(np.float32)
    return {
        "method": "solver_full",
        "final_state": out,
        "wall_time_s": wall,
        "n_surrogate_calls": 0,
        "n_solver_calls": n_steps,
    }


# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mode2_temporal_hybrid(model, s_init: torch.Tensor, target_dt: float,
                            env_cfg: dict, trained_max_dt: float = 0.128,
                            base_dt: float = 0.002) -> dict:
    """Temporal handoff: surrogate within trained regime, solver beyond.

    If target_dt <= trained_max_dt: pure Mode 1 (single FiLM call).
    Else:
        1. pred_intermediate = surrogate(s_init, trained_max_dt)   1 surrogate call
        2. final = solver(pred_intermediate, target_dt - trained_max_dt) on HLL
        3. return final.

    Built on the user's idea: avoid Mode 1's catastrophic extrapolation past
    h=64 by handing off to the solver at the trained boundary. Cost vs solver:
    saves the trained-regime portion of solver compute.
    """
    if s_init.dim() == 3:
        s_init = s_init.unsqueeze(0)
    t0 = time.time()
    if target_dt <= trained_max_dt + 1e-9:
        # No extrapolation: use Mode 1 directly
        pred = predict(model, s_init, target_dt)[0].detach().cpu().numpy()
        n_solver_calls = 0
        used_solver = False
    else:
        # Step 1: surrogate up to trained_max_dt
        pred_intermediate = predict(model, s_init, trained_max_dt)[0].detach().cpu().numpy()

        # Clamp unphysical values before handing to the HLL solver. The
        # surrogate can produce slightly negative density or pressure that
        # destabilises the solver. Clamping to physical bounds.
        rho = np.maximum(pred_intermediate[0], 1e-3)
        rhou = pred_intermediate[1]; rhov = pred_intermediate[2]
        E = pred_intermediate[3]
        # Ensure pressure is non-negative: E >= kinetic + min_internal
        gamma = float(env_cfg["solver"]["params"]["gamma"])
        kinetic = 0.5 * (rhou * rhou + rhov * rhov) / rho
        p_min = 1e-3
        E = np.maximum(E, kinetic + p_min / (gamma - 1.0))
        pred_intermediate = np.stack([rho, rhou, rhov, E], axis=0).astype(np.float32)

        # Step 2: solver for the remaining (target_dt - trained_max_dt)
        remaining_dt = target_dt - trained_max_dt
        sv = solver_full_euler(pred_intermediate, remaining_dt, env_cfg, base_dt=base_dt)
        pred = sv["final_state"]
        n_solver_calls = sv["n_solver_calls"]
        used_solver = True
    wall = time.time() - t0
    return {
        "method": "mode2_temporal",
        "final_state": pred,
        "wall_time_s": wall,
        "n_surrogate_calls": 1,
        "n_solver_calls": n_solver_calls,
        "trained_max_dt": trained_max_dt,
        "used_solver": used_solver,
    }


@torch.no_grad()
def mode2_trust_aware(model, s_init: torch.Tensor, target_dt: float,
                       env_cfg: dict, tau_global: float,
                       base_dt: float = 0.002) -> dict:
    """Trust-aware fallback. ê.mean threshold decision."""
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
        sv = solver_full_euler(s_init_np, target_dt, env_cfg, base_dt=base_dt)
        out = sv["final_state"]
        n_solver_calls = sv["n_solver_calls"]
    wall = time.time() - t0
    return {
        "method": "mode2",
        "final_state": out,
        "wall_time_s": wall,
        "n_surrogate_calls": 3,
        "n_solver_calls": n_solver_calls,
        "tau_global": tau_global,
        "e_hat_mean": e_hat_mean,
        "used_solver": used_solver,
    }


# ─────────────────────────────────────────────────────────────────────────
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def calibrate_tau(model, ds: Euler2DDataset, traj_idxs: list,
                    horizons_steps: list, device: str, quantile: float = 0.5,
                    t0_seed: int = 12345):
    """Calibrate τ_global using random t0 (mid-trajectory, not the IC).

    Sampling from t0=0 only is wrong because Euler ICs are piecewise-constant
    (Schulz-Rinne quadrants, Sedov delta), the hardest case for the model.
    Random t0 samples evolved states which is the realistic use case.
    """
    print(f"[calib] q={quantile}, n_trajs={len(traj_idxs)}", flush=True)
    e_means = {h: [] for h in horizons_steps}
    rng = np.random.RandomState(t0_seed)
    for ti in traj_idxs:
        for h in horizons_steps:
            t0 = int(rng.randint(0, ds.T - h))
            s = ds.frame(ti, t0).to(device).unsqueeze(0)
            e_hat, _ = step_doubling_estimator(model, s, h * ds.dt)
            e_means[h].append(float(e_hat.mean().item()))
    out = {h: float(np.quantile(np.array(e_means[h]), quantile)) for h in horizons_steps}
    for h, v in out.items():
        print(f"  h={h}  τ={v:.4f}  range=[{min(e_means[h]):.4f}, {max(e_means[h]):.4f}]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_trajs", type=int, default=8)
    ap.add_argument("--target_steps_list", default="2,4,8,16,32,64")
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--device", default="cpu",
                     help="cpu (fair vs solver) or cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "euler2d"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)

    ds_path = Path(args.data_dir) / f"euler2d_v2_{args.split}.h5"
    horizons_steps = [int(x) for x in args.target_steps_list.split(",")]
    print(f"[modes] split={args.split}  n_trajs={args.n_trajs}  "
          f"horizons={horizons_steps}", flush=True)

    model = load_model(args.ckpt, device=args.device)
    ds = Euler2DDataset(str(ds_path))
    rng = np.random.RandomState(args.seed)
    traj_idxs = rng.choice(ds.N, size=min(args.n_trajs, ds.N), replace=False).tolist()
    print(f"[modes] traj indices: {traj_idxs}", flush=True)

    print("\n=== Stage 1: calibrate τ_global per horizon ===", flush=True)
    tau_per_h = calibrate_tau(model, ds, traj_idxs, horizons_steps,
                                device=args.device, quantile=args.quantile)

    env_cfg = load_config("euler2d_v2")["env"]
    base_dt = float(ds.dt)

    print("\n=== Stage 2: eval Mode 1, Mode 2, solver ===", flush=True)
    rows = []
    # Sample a random t0 per (traj, horizon) — t0=0 would always pick the
    # piecewise-constant IC, the hardest case. Match validate() methodology.
    rng_t0 = np.random.RandomState(args.seed + 9999)
    for ti in traj_idxs:
        for h_steps in horizons_steps:
            target_dt = h_steps * base_dt
            t0_eval = int(rng_t0.randint(0, ds.T - h_steps))
            s_init = ds.frame(ti, t0_eval).to(args.device)
            s_init_np = s_init.cpu().numpy()
            target_np = ds.frame(ti, t0_eval + h_steps).cpu().numpy()

            # Mode 1
            r1 = mode1_pure_surrogate(model, s_init, target_dt)
            r1["rmse_vs_gt"] = rmse(r1["final_state"], target_np)
            r1["traj_idx"] = ti; r1["target_steps"] = h_steps; r1["target_dt"] = target_dt
            r1.pop("final_state", None); rows.append(r1)

            # Mode 2
            r2 = mode2_trust_aware(model, s_init, target_dt, env_cfg,
                                     tau_global=tau_per_h[h_steps], base_dt=base_dt)
            r2["rmse_vs_gt"] = rmse(r2["final_state"], target_np)
            r2["traj_idx"] = ti; r2["target_steps"] = h_steps; r2["target_dt"] = target_dt
            r2.pop("final_state", None); rows.append(r2)

            # Solver
            sv = solver_full_euler(s_init_np, target_dt, env_cfg, base_dt=base_dt)
            sv["rmse_vs_gt"] = rmse(sv["final_state"], target_np)
            sv["traj_idx"] = ti; sv["target_steps"] = h_steps; sv["target_dt"] = target_dt
            sv.pop("final_state", None); rows.append(sv)
            for r in (r1, r2, sv): r["t0"] = t0_eval

            print(f"  traj {ti:3d}  h={h_steps:3d}  t0={t0_eval:3d}  "
                  f"M1 rmse={r1['rmse_vs_gt']:.4f} wall={r1['wall_time_s']:.2f}s | "
                  f"M2 rmse={r2['rmse_vs_gt']:.4f} wall={r2['wall_time_s']:.2f}s solver={r2['used_solver']} | "
                  f"SV rmse={sv['rmse_vs_gt']:.4f} wall={sv['wall_time_s']:.2f}s",
                  flush=True)

    out_path = Path(args.out) if args.out else (
        ROOT / "evaluation" / "euler2d_eval" / "results" /
        f"euler_modes_{args.split}_q{int(args.quantile*100)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split, "n_trajs": len(traj_idxs),
        "traj_idxs": traj_idxs, "horizons_steps": horizons_steps,
        "quantile": args.quantile, "tau_per_h": tau_per_h,
        "rows": rows,
    }, indent=2))
    print(f"\n[modes] wrote {out_path}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
