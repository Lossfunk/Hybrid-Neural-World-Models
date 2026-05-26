#!/usr/bin/env python3
"""Evaluate Mode 3X (Option X — trajectory-level mode selector).

Two-stage eval:
  Stage 1 (calibration): for each (traj, horizon), compute ê.mean(). Use the
      per-horizon median ê.mean() across the eval set as tau_global. This
      means roughly half the trajectories will trigger the solver fallback.
  Stage 2 (eval): run mode1, mode3x, solver_full per traj/horizon. Compute
      RMSE and wall time.

Output: results/modes_pareto_x.json
"""
from __future__ import annotations

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

from eval_utils import load_model, predict, step_doubling_estimator        # noqa: E402
from eval_modes import (mode1_pure_surrogate, mode3x_traj_select,            # noqa: E402
                          solver_full, rmse, DT_BASE)


def calibrate_thresholds(model, ds_path: str, traj_idxs: list,
                          horizons_steps: list, device: str = "cpu",
                          quantile: float = 0.5) -> dict:
    """Compute per-horizon tau_global = quantile of ê.mean() over the
    given trajectories. Default quantile=0.5 → half the trajs will trigger
    the solver branch."""
    print(f"[calib] quantile={quantile}  trajs={len(traj_idxs)}", flush=True)
    e_means = {h: [] for h in horizons_steps}
    with h5py.File(ds_path, "r") as f:
        for ti in traj_idxs:
            s_init = np.array(f["states"][ti, 0], dtype=np.float32)
            s_init_t = torch.from_numpy(s_init).to(device)
            for h_steps in horizons_steps:
                target_dt = h_steps * DT_BASE
                e_hat_map, _ = step_doubling_estimator(
                    model, s_init_t.unsqueeze(0), target_dt)
                e_means[h_steps].append(float(e_hat_map.mean().item()))
    tau = {}
    for h_steps in horizons_steps:
        vals = np.array(e_means[h_steps])
        tau[h_steps] = float(np.quantile(vals, quantile))
        print(f"  h={h_steps}  ê.mean range=[{vals.min():.4f}, {vals.max():.4f}]  "
              f"τ(q{int(quantile*100)})={tau[h_steps]:.4f}", flush=True)
    return tau


def run_eval(model, ds_path: str, traj_idxs: list, horizons_steps: list,
              tau_per_h: dict, device: str = "cpu") -> list:
    rows = []
    with h5py.File(ds_path, "r") as f:
        for ti in traj_idxs:
            s_init = np.array(f["states"][ti, 0], dtype=np.float32)
            params = dict(
                f=float(f["params"][ti, 0]),
                eps=float(f["params"][ti, 1]),
                q=float(f["params"][ti, 2]),
                D=float(f["params"][ti, 3]),
            )
            s_init_t = torch.from_numpy(s_init).to(device)
            for h_steps in horizons_steps:
                target_dt = h_steps * DT_BASE
                target = np.array(f["states"][ti, h_steps], dtype=np.float32)
                tau = tau_per_h[h_steps]

                # Mode 1
                r = mode1_pure_surrogate(model, s_init_t, target_dt)
                r["rmse_vs_gt"] = rmse(r["final_state"], target)
                r["traj_idx"] = ti; r["target_dt"] = target_dt
                r["target_steps"] = h_steps
                r.pop("final_state", None)
                rows.append(r)

                # Mode 3X
                r = mode3x_traj_select(model, s_init_t, target_dt, params,
                                        tau_global=tau)
                r["rmse_vs_gt"] = rmse(r["final_state"], target)
                r["traj_idx"] = ti; r["target_dt"] = target_dt
                r["target_steps"] = h_steps
                r.pop("final_state", None)
                rows.append(r)

                # Solver full
                r = solver_full(s_init, target_dt, params)
                r["rmse_vs_gt"] = rmse(r["final_state"], target)
                r["traj_idx"] = ti; r["target_dt"] = target_dt
                r["target_steps"] = h_steps
                r.pop("final_state", None)
                rows.append(r)

                m1 = rows[-3]; m3x = rows[-2]; sv = rows[-1]
                print(f"  traj {ti:3d}  h={h_steps:3d}  "
                      f"M1 rmse={m1['rmse_vs_gt']:.4f} wall={m1['wall_time_s']:.2f}s | "
                      f"M3X rmse={m3x['rmse_vs_gt']:.4f} wall={m3x['wall_time_s']:.2f}s "
                      f"used_solver={m3x['used_solver']} | "
                      f"SV rmse={sv['rmse_vs_gt']:.4f} wall={sv['wall_time_s']:.2f}s",
                      flush=True)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        ROOT / "checkpoints" / "shortcut_oregonator_v3" / "seed0" / "best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_trajs", type=int, default=8)
    ap.add_argument("--target_steps_list", default="4,8,16,32,64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantile", type=float, default=0.5,
                     help="τ_global = quantile-th percentile of ê.mean() "
                          "across calibration trajectories. q=0.5 → half "
                          "the trajs trigger solver.")
    ap.add_argument("--out", default=str(
        ROOT / "results" / "modes_pareto_x.json"))
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)

    ds_path = ROOT / "data" / "oregonator" / f"oregonator_{args.split}.h5"
    horizons_steps = [int(x) for x in args.target_steps_list.split(",")]

    model = load_model(args.ckpt, device=args.device)
    with h5py.File(ds_path, "r") as f:
        N = f["states"].shape[0]
    rng = np.random.RandomState(args.seed)
    traj_idxs = rng.choice(N, size=min(args.n_trajs, N),
                             replace=False).tolist()
    print(f"[mode3x] split={args.split}  n_trajs={len(traj_idxs)}  "
          f"horizons={horizons_steps}", flush=True)
    print(f"[mode3x] traj indices: {traj_idxs}", flush=True)

    print("\n=== Stage 1: calibrate τ_global per horizon ===", flush=True)
    tau_per_h = calibrate_thresholds(
        model, str(ds_path), traj_idxs, horizons_steps,
        device=args.device, quantile=args.quantile,
    )

    print("\n=== Stage 2: eval mode1 vs mode3x vs solver ===", flush=True)
    rows = run_eval(model, str(ds_path), traj_idxs, horizons_steps,
                      tau_per_h, device=args.device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split,
        "n_trajs": len(traj_idxs),
        "traj_idxs": traj_idxs,
        "horizons_steps": horizons_steps,
        "quantile": args.quantile,
        "tau_per_h": tau_per_h,
        "rows": rows,
    }, indent=2))
    print(f"\n[mode3x] wrote {out_path}")


if __name__ == "__main__":
    main()
