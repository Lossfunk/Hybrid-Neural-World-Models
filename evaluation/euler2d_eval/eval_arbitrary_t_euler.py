#!/usr/bin/env python3
"""Mode 1 vs solver across trained + extrapolated horizons for Euler.

Sweeps target horizon h at trained anchors {1, 2, 4, 8, 16, 32, 64} steps,
non-trained interpolated {3, 11, 19, 39}, and extrapolated beyond h=64
{80, 90, 99}.

For each h: run Mode 1 + production HLL solver on the same n_trajs
trajectories using random t0. Report wall times and RMSE.

Output: results/euler_modes_arbitrary_t.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
NEURIPS = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "data_generation" / "euler2d"))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))
sys.path.insert(0, str(ROOT / "data_generation" / "euler2d"))

from eval_utils_euler import load_model, predict     # noqa: E402
from data_utils_2d import Euler2DDataset             # noqa: E402
from config import load_config                       # noqa: E402
from eval_modes_euler import (mode1_pure_surrogate,   # noqa: E402
                                  solver_full_euler, rmse)


HORIZONS_TRAINED = [1, 2, 4, 8, 16, 32, 64]
HORIZONS_INTERP  = [3, 11, 19, 39]              # between trained anchors
HORIZONS_EXTRAP  = [80, 90, 99]                  # beyond trained max h=64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_trajs", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                     help="cpu (fair vs solver) or cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "euler2d"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)

    horizons = sorted(set(HORIZONS_TRAINED + HORIZONS_INTERP + HORIZONS_EXTRAP))
    ds_path = Path(args.data_dir) / f"euler2d_v2_{args.split}.h5"
    print(f"[arb_t] split={args.split}  horizons={horizons}", flush=True)

    model = load_model(args.ckpt, device=args.device)
    ds = Euler2DDataset(str(ds_path))
    rng = np.random.RandomState(args.seed)
    traj_idxs = rng.choice(ds.N, size=min(args.n_trajs, ds.N),
                             replace=False).tolist()
    print(f"[arb_t] traj indices: {traj_idxs}", flush=True)

    env_cfg = load_config("euler2d_v2")["env"]
    base_dt = float(ds.dt)

    # Random t0 per (traj, h) — same as eval_modes_euler v2
    rng_t0 = np.random.RandomState(args.seed + 9999)

    # Warm up the model on GPU/CPU before timing
    s_warm = ds.frame(traj_idxs[0], 0).to(args.device).unsqueeze(0)
    with torch.no_grad():
        _ = predict(model, s_warm, base_dt * 4)
    print("[arb_t] model warmed", flush=True)

    rows = []
    for ti in traj_idxs:
        for h in horizons:
            if h >= ds.T:
                continue
            target_dt = h * base_dt
            t0 = int(rng_t0.randint(0, ds.T - h))
            s_init = ds.frame(ti, t0).to(args.device)
            s_init_np = s_init.cpu().numpy()
            target_np = ds.frame(ti, t0 + h).cpu().numpy()

            # Mode 1
            r1 = mode1_pure_surrogate(model, s_init, target_dt)
            r1["rmse_vs_gt"] = rmse(r1["final_state"], target_np)
            r1["traj_idx"] = ti; r1["target_steps"] = h; r1["target_dt"] = target_dt; r1["t0"] = t0
            r1.pop("final_state", None); rows.append(r1)

            # Solver
            sv = solver_full_euler(s_init_np, target_dt, env_cfg, base_dt=base_dt)
            sv["rmse_vs_gt"] = rmse(sv["final_state"], target_np)
            sv["traj_idx"] = ti; sv["target_steps"] = h; sv["target_dt"] = target_dt; sv["t0"] = t0
            sv.pop("final_state", None); rows.append(sv)

            print(f"  traj {ti:3d}  h={h:3d}  t0={t0:3d}  "
                  f"M1 rmse={r1['rmse_vs_gt']:.4f} wall={r1['wall_time_s']:.3f}s | "
                  f"SV rmse={sv['rmse_vs_gt']:.4f} wall={sv['wall_time_s']:.3f}s | "
                  f"speedup={sv['wall_time_s']/r1['wall_time_s']:.2f}x",
                  flush=True)

    out_path = Path(args.out) if args.out else (
        ROOT / "evaluation" / "euler2d_eval" / "results" /
        f"euler_modes_arbitrary_t_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split, "n_trajs": len(traj_idxs),
        "traj_idxs": traj_idxs,
        "horizons_trained": HORIZONS_TRAINED,
        "horizons_interp": HORIZONS_INTERP,
        "horizons_extrap": HORIZONS_EXTRAP,
        "rows": rows,
    }, indent=2))
    print(f"\n[arb_t] wrote {out_path}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
