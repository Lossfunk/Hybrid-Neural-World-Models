#!/usr/bin/env python3
"""Speedup at single trajectory (B=1) vs horizon, all 3 envs.

For each (env, horizon h ∈ {2, 4, 8, 16, 32, 64}) measure:
  - Solver wall time on a single trajectory
  - Surrogate wall time at B=1
  - Speedup = solver / surrogate
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
ROOT = HERE.parent.parent.parent
RESULTS = HERE / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HORIZONS = [2, 4, 8, 16, 32, 64]
N_REPEATS = 3
N_WARMUP = 2


def bench_oreg():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "data_generation" / "oregonator"))
    from eval_utils import load_model
    from oregonator2d_tyson import OregonatorTyson2D, TysonParams

    DT_BASE = 0.05
    ckpt = ROOT / "checkpoints" / "oregonator" / "best.pt"
    data = ROOT / "data" / "oregonator" / "oregonator_test.h5"
    model = load_model(str(ckpt), device=DEVICE)

    with h5py.File(data, "r") as f:
        s_init = np.array(f["states"][0, 0], dtype=np.float32)
        params = dict(
            f=float(f["params"][0, 0]), eps=float(f["params"][0, 1]),
            q=float(f["params"][0, 2]), D=float(f["params"][0, 3]),
        )
    s_t = torch.from_numpy(s_init).unsqueeze(0).to(DEVICE)

    out = {}
    for h in HORIZONS:
        target_dt = h * DT_BASE
        # surrogate
        dt = torch.tensor([target_dt], dtype=torch.float32, device=DEVICE)
        for _ in range(N_WARMUP): _ = model(s_t, dt)
        if DEVICE == "cuda": torch.cuda.synchronize()
        sur_t = []
        for _ in range(N_REPEATS):
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            _ = model(s_t, dt)
            if DEVICE == "cuda": torch.cuda.synchronize()
            sur_t.append(time.time() - t0)
        sur_ms = float(np.median(sur_t)) * 1000

        # solver
        sim = OregonatorTyson2D(n_x=256, n_y=256, L_x=100., L_y=100.,
                                params=TysonParams(**params))
        sim.u[:] = s_init[0]; sim.v[:] = s_init[1]; sim.t_sim = 0
        for _ in range(2): sim.step(DT_BASE)  # warmup
        solv_t = []
        for _ in range(N_REPEATS):
            sim.u[:] = s_init[0]; sim.v[:] = s_init[1]; sim.t_sim = 0
            t0 = time.time()
            for _ in range(h): sim.step(DT_BASE)
            solv_t.append(time.time() - t0)
        solv_ms = float(np.median(solv_t)) * 1000
        out[h] = {"surrogate_ms": sur_ms, "solver_ms": solv_ms,
                   "speedup": solv_ms / sur_ms}
        print(f"  oreg h={h:3d}  sur={sur_ms:7.2f}ms  solv={solv_ms:8.1f}ms  "
              f"speedup={out[h]['speedup']:.0f}x", flush=True)
    return out


def bench_euler():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "data_generation" / "euler2d"))
    from eval_utils_euler import load_model
    from data_utils_2d import Euler2DDataset
    import euler2d_v2 as env_mod

    BASE_DT = 0.002
    ckpt = ROOT / "checkpoints" / "euler2d" / "best.pt"
    data = ROOT / "data" / "euler2d" / "euler2d_v2_test.h5"
    model = load_model(str(ckpt), device=DEVICE)
    ds = Euler2DDataset(str(data))
    s_t = ds.frame(0, 0).to(DEVICE).unsqueeze(0)
    s_init_np = s_t[0].cpu().numpy()

    out = {}
    for h in HORIZONS:
        target_dt = h * BASE_DT
        dt = torch.tensor([target_dt], dtype=torch.float32, device=DEVICE)
        for _ in range(N_WARMUP): _ = model(s_t, dt)
        if DEVICE == "cuda": torch.cuda.synchronize()
        sur_t = []
        for _ in range(N_REPEATS):
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            _ = model(s_t, dt)
            if DEVICE == "cuda": torch.cuda.synchronize()
            sur_t.append(time.time() - t0)
        sur_ms = float(np.median(sur_t)) * 1000

        # solver
        env_cfg_p = {"grid": [128, 128], "domain": [0., 1., 0., 1.],
                       "gamma": 1.4, "cfl": 0.4}
        domain = tuple(env_cfg_p["domain"])
        nx, ny = env_cfg_p["grid"]
        dx = (domain[1] - domain[0]) / nx
        dy = (domain[3] - domain[2]) / ny
        solv_t = []
        for _ in range(N_REPEATS + 1):
            env_obj = env_mod.Env({"env": {"solver": {"params": env_cfg_p},
                                              "base_dt": BASE_DT,
                                              "event_detector": {"params": {"threshold": 0.05}},
                                              "trajectory_length": h}})
            env_obj._q = np.transpose(s_init_np, (1, 2, 0)).astype(np.float64)
            env_obj._dx = dx; env_obj._dy = dy
            t0 = time.time()
            env_obj._step_fixed_dt(target_dt)
            solv_t.append(time.time() - t0)
        solv_ms = float(np.median(solv_t[1:])) * 1000
        out[h] = {"surrogate_ms": sur_ms, "solver_ms": solv_ms,
                   "speedup": solv_ms / sur_ms}
        print(f"  euler h={h:3d}  sur={sur_ms:7.2f}ms  solv={solv_ms:8.1f}ms  "
              f"speedup={out[h]['speedup']:.0f}x", flush=True)
    return out


def bench_ball():
    sys.path.insert(0, str(ROOT / "training" / "ball3d"))
    sys.path.insert(0, str(ROOT / "data_generation" / "ball3d"))
    from shortcut_ball3d import ShortcutBall3D
    from ball3d_env import Ball3DEnv

    DT_BASE = 0.01
    ckpt = ROOT / "checkpoints" / "ball3d" / "best.pt"
    data = ROOT / "data" / "ball3d" / "ball3d_test.h5"
    ck = torch.load(str(ckpt), map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    model = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                              emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                              ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    with h5py.File(data, "r") as f:
        s_init = np.array(f["states"][0, 0], dtype=np.float32)
        rest = float(f.attrs.get("restitution", 1.0))
        grav = float(f.attrs.get("gravity_z", 0.0))
    s_t = torch.from_numpy(s_init).unsqueeze(0).to(DEVICE)

    out = {}
    for h in HORIZONS:
        target_dt = h * DT_BASE
        dt = torch.tensor([target_dt], dtype=torch.float32, device=DEVICE)
        for _ in range(N_WARMUP): _ = model(s_t, dt)
        if DEVICE == "cuda": torch.cuda.synchronize()
        sur_t = []
        for _ in range(50):
            if DEVICE == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            _ = model(s_t, dt)
            if DEVICE == "cuda": torch.cuda.synchronize()
            sur_t.append(time.time() - t0)
        sur_ms = float(np.median(sur_t)) * 1000

        solv_t = []
        for _ in range(50):
            env = Ball3DEnv()
            env._state = s_init.astype(np.float64).copy()
            env._restitution = rest
            env._gravity_vec = np.array([0., 0., grav])
            t0 = time.time()
            env.step(target_dt)
            solv_t.append(time.time() - t0)
        solv_ms = float(np.median(solv_t)) * 1000
        out[h] = {"surrogate_ms": sur_ms, "solver_ms": solv_ms,
                   "speedup": solv_ms / sur_ms}
        print(f"  ball  h={h:3d}  sur={sur_ms:7.4f}ms  solv={solv_ms:8.4f}ms  "
              f"speedup={out[h]['speedup']:.2f}x", flush=True)
    return out


def main():
    print(f"[horizon_sweep] device={DEVICE}", flush=True)
    print("\n=== Oregonator ===", flush=True)
    oreg = bench_oreg()
    print("\n=== Euler ===", flush=True)
    euler = bench_euler()
    print("\n=== Ball 3D ===", flush=True)
    ball = bench_ball()
    out = {"oregonator": oreg, "euler": euler, "ball3d": ball}
    out_path = RESULTS / "horizon_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
