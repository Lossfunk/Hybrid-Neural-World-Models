#!/usr/bin/env python3
"""CPU-only honest benchmark.

Surrogate moved to CPU; solver already on CPU. Reports:
  - Surrogate ms/pair at B in {1, 4, 16, 32}
  - Solver ms/traj
  - Speedup at each B = (B * solver_ms) / surrogate_batch_total_ms
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
DEVICE = "cpu"
torch.set_num_threads(8)  # cap so we don't hog cores

H = 64
BATCH_SIZES = [1, 4, 16, 32]
N_REPEATS = 3
N_WARMUP = 1


def time_surrogate(model, states, B, target_dt):
    s = states[:B]
    dt = torch.full((B,), target_dt, dtype=torch.float32, device=DEVICE)
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(s, dt)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.time()
        with torch.no_grad():
            _ = model(s, dt)
        times.append(time.time() - t0)
    return float(np.median(times)) * 1000  # ms total batch


def bench_oreg():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "data_generation" / "oregonator"))
    from eval_utils import load_model
    from oregonator2d_tyson import OregonatorTyson2D, TysonParams

    DT_BASE = 0.05
    target_dt = H * DT_BASE
    ckpt = ROOT / "checkpoints" / "oregonator" / "best.pt"
    data = ROOT / "data" / "oregonator" / "oregonator_test.h5"
    model = load_model(str(ckpt), device=DEVICE)
    print(f"[oreg cpu] model loaded", flush=True)

    with h5py.File(data, "r") as f:
        states = np.array(f["states"][:max(BATCH_SIZES), 0], dtype=np.float32)
        params = dict(f=float(f["params"][0, 0]), eps=float(f["params"][0, 1]),
                       q=float(f["params"][0, 2]), D=float(f["params"][0, 3]))
    states_t = torch.from_numpy(states).to(DEVICE)

    # solver
    sim = OregonatorTyson2D(n_x=256, n_y=256, L_x=100., L_y=100.,
                              params=TysonParams(**params))
    sim.u[:] = states[0, 0]; sim.v[:] = states[0, 1]; sim.t_sim = 0
    for _ in range(2): sim.step(DT_BASE)
    solv_t = []
    for _ in range(2):
        sim.u[:] = states[0, 0]; sim.v[:] = states[0, 1]; sim.t_sim = 0
        t0 = time.time()
        for _ in range(H): sim.step(DT_BASE)
        solv_t.append(time.time() - t0)
    solver_ms = float(np.median(solv_t)) * 1000

    out = {"solver_ms_per_traj": solver_ms,
           "surrogate_batch_ms": {}, "speedup_by_batch": {}}
    print(f"  oreg solver: {solver_ms:.1f} ms/traj", flush=True)
    for B in BATCH_SIZES:
        if B > states_t.shape[0]: break
        ms = time_surrogate(model, states_t, B, target_dt)
        out["surrogate_batch_ms"][B] = ms
        out["speedup_by_batch"][B] = (B * solver_ms) / ms
        print(f"  oreg B={B:3d}  surrogate={ms:.1f}ms total  ms/pair={ms/B:.2f}  "
              f"speedup={(B*solver_ms)/ms:.1f}x", flush=True)
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
    target_dt = H * BASE_DT
    ckpt = ROOT / "checkpoints" / "euler2d" / "best.pt"
    data = ROOT / "data" / "euler2d" / "euler2d_v2_test.h5"
    model = load_model(str(ckpt), device=DEVICE)
    ds = Euler2DDataset(str(data))
    states_t = torch.stack([ds.frame(i, 0) for i in range(min(max(BATCH_SIZES), ds.N))])
    s_init_np = states_t[0].numpy()

    # solver
    p = {"grid": [128, 128], "domain": [0., 1., 0., 1.], "gamma": 1.4, "cfl": 0.4}
    domain = tuple(p["domain"]); nx, ny = p["grid"]
    dx, dy = (domain[1]-domain[0])/nx, (domain[3]-domain[2])/ny
    solv_t = []
    for _ in range(3):
        env_obj = env_mod.Env({"env": {"solver": {"params": p}, "base_dt": BASE_DT,
                                         "event_detector": {"params": {"threshold": 0.05}},
                                         "trajectory_length": H}})
        env_obj._q = np.transpose(s_init_np, (1, 2, 0)).astype(np.float64)
        env_obj._dx = dx; env_obj._dy = dy
        t0 = time.time()
        env_obj._step_fixed_dt(target_dt)
        solv_t.append(time.time() - t0)
    solver_ms = float(np.median(solv_t)) * 1000

    out = {"solver_ms_per_traj": solver_ms,
           "surrogate_batch_ms": {}, "speedup_by_batch": {}}
    print(f"  euler solver: {solver_ms:.1f} ms/traj", flush=True)
    for B in BATCH_SIZES:
        if B > states_t.shape[0]: break
        ms = time_surrogate(model, states_t, B, target_dt)
        out["surrogate_batch_ms"][B] = ms
        out["speedup_by_batch"][B] = (B * solver_ms) / ms
        print(f"  euler B={B:3d}  surrogate={ms:.1f}ms total  ms/pair={ms/B:.2f}  "
              f"speedup={(B*solver_ms)/ms:.1f}x", flush=True)
    return out


def bench_ball():
    sys.path.insert(0, str(ROOT / "training" / "ball3d"))
    sys.path.insert(0, str(ROOT / "data_generation" / "ball3d"))
    from shortcut_ball3d import ShortcutBall3D
    from ball3d_env import Ball3DEnv

    DT_BASE = 0.01
    target_dt = H * DT_BASE
    ckpt = ROOT / "checkpoints" / "ball3d" / "best.pt"
    data = ROOT / "data" / "ball3d" / "ball3d_test.h5"
    ck = torch.load(str(ckpt), map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    model = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                              emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                              ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    with h5py.File(data, "r") as f:
        states = np.array(f["states"][:max(BATCH_SIZES), 0], dtype=np.float32)
        rest = float(f.attrs.get("restitution", 1.0))
        grav = float(f.attrs.get("gravity_z", 0.0))
    states_t = torch.from_numpy(states).to(DEVICE)

    solv_t = []
    for _ in range(20):
        env = Ball3DEnv()
        env._state = states[0].astype(np.float64).copy()
        env._restitution = rest
        env._gravity_vec = np.array([0., 0., grav])
        t0 = time.time()
        env.step(target_dt)
        solv_t.append(time.time() - t0)
    solver_ms = float(np.median(solv_t)) * 1000

    out = {"solver_ms_per_traj": solver_ms,
           "surrogate_batch_ms": {}, "speedup_by_batch": {}}
    print(f"  ball solver: {solver_ms:.4f} ms/traj", flush=True)
    for B in BATCH_SIZES:
        if B > states_t.shape[0]: break
        ms = time_surrogate(model, states_t, B, target_dt)
        out["surrogate_batch_ms"][B] = ms
        out["speedup_by_batch"][B] = (B * solver_ms) / ms
        print(f"  ball B={B:3d}  surrogate={ms:.3f}ms total  ms/pair={ms/B:.4f}  "
              f"speedup={(B*solver_ms)/ms:.2f}x", flush=True)
    return out


def main():
    print(f"[cpu_bench] threads={torch.get_num_threads()}", flush=True)
    print("\n=== Oregonator (CPU-CPU) ===", flush=True)
    oreg = bench_oreg()
    print("\n=== Euler (CPU-CPU) ===", flush=True)
    euler = bench_euler()
    print("\n=== Ball 3D (CPU-CPU) ===", flush=True)
    ball = bench_ball()
    out = {"oregonator": oreg, "euler": euler, "ball3d": ball,
           "horizon": H, "device": "cpu", "torch_threads": torch.get_num_threads()}
    out_path = RESULTS / "cpu_h64.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
