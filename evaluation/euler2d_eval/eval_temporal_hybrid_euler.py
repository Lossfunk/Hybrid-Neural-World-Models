#!/usr/bin/env python3
"""Compare Mode 1 vs Mode 2-temporal-hybrid vs solver across horizons.

Mode 2-temporal-hybrid: surrogate from t=0 to t=trained_max (h=64, dt=0.128s),
then solver from there for the remainder. Designed for extrapolation
horizons where Mode 1 alone fails.
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
                                  mode2_temporal_hybrid,
                                  solver_full_euler, rmse)


HORIZONS = [4, 8, 16, 32, 64, 80, 90, 99]    # mix of trained + extrap
TRAINED_MAX_H = 64                            # max horizon model was trained on
TRAINED_MAX_DT = TRAINED_MAX_H * 0.002         # 0.128s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_trajs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "euler2d"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)

    ds_path = Path(args.data_dir) / f"euler2d_v2_{args.split}.h5"
    print(f"[temp_hybrid] split={args.split}  trained_max_dt={TRAINED_MAX_DT}",
          flush=True)

    model = load_model(args.ckpt, device=args.device)
    ds = Euler2DDataset(str(ds_path))
    rng = np.random.RandomState(args.seed)
    traj_idxs = rng.choice(ds.N, size=min(args.n_trajs, ds.N),
                             replace=False).tolist()
    print(f"[temp_hybrid] traj indices: {traj_idxs}", flush=True)

    env_cfg = load_config("euler2d_v2")["env"]
    base_dt = float(ds.dt)

    # warm
    s_warm = ds.frame(traj_idxs[0], 0).to(args.device).unsqueeze(0)
    with torch.no_grad():
        _ = predict(model, s_warm, base_dt * 4)

    rng_t0 = np.random.RandomState(args.seed + 9999)
    rows = []
    for ti in traj_idxs:
        for h in HORIZONS:
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
            r1["traj_idx"]=ti; r1["target_steps"]=h; r1["target_dt"]=target_dt; r1["t0"]=t0
            r1.pop("final_state", None); rows.append(r1)

            # Mode 2 temporal hybrid
            r2 = mode2_temporal_hybrid(model, s_init, target_dt, env_cfg,
                                          trained_max_dt=TRAINED_MAX_DT,
                                          base_dt=base_dt)
            r2["rmse_vs_gt"] = rmse(r2["final_state"], target_np)
            r2["traj_idx"]=ti; r2["target_steps"]=h; r2["target_dt"]=target_dt; r2["t0"]=t0
            r2.pop("final_state", None); rows.append(r2)

            # Solver
            sv = solver_full_euler(s_init_np, target_dt, env_cfg, base_dt=base_dt)
            sv["rmse_vs_gt"] = rmse(sv["final_state"], target_np)
            sv["traj_idx"]=ti; sv["target_steps"]=h; sv["target_dt"]=target_dt; sv["t0"]=t0
            sv.pop("final_state", None); rows.append(sv)

            print(f"  traj {ti:3d}  h={h:3d}  t0={t0:3d}  "
                  f"M1 rmse={r1['rmse_vs_gt']:.4f} wall={r1['wall_time_s']:.3f}s | "
                  f"M2temp rmse={r2['rmse_vs_gt']:.4f} wall={r2['wall_time_s']:.3f}s "
                  f"used_solver={r2['used_solver']} | "
                  f"SV rmse={sv['rmse_vs_gt']:.4f} wall={sv['wall_time_s']:.3f}s",
                  flush=True)

    out_path = Path(args.out) if args.out else (
        ROOT / "evaluation" / "euler2d_eval" / "results" /
        f"euler_temporal_hybrid_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split, "n_trajs": len(traj_idxs),
        "trained_max_h": TRAINED_MAX_H,
        "trained_max_dt": TRAINED_MAX_DT,
        "rows": rows,
    }, indent=2))
    print(f"\n[temp_hybrid] wrote {out_path}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
