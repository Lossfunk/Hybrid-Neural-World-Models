#!/usr/bin/env python3
"""Surrogate-vs-solver batch benchmark — Euler 2D."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(NEURIPS))
sys.path.insert(0, str(ROOT / "data_generation" / "euler2d"))

from eval_utils_euler import load_model
from data_utils_2d import Euler2DDataset
import euler2d_v2 as env_mod

CKPT = (ROOT / "checkpoints" / "euler2d" / "best.pt")
DATA = ROOT / "data" / "euler2d" / "euler2d_v2_test.h5"
RESULTS = HERE / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

H = 64
BASE_DT = 0.002
TARGET_DT = H * BASE_DT
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
N_REPEATS = 5
N_WARMUP = 3


@torch.no_grad()
def time_surrogate_batch(model, states_t: torch.Tensor, B: int):
    dt = torch.full((B,), TARGET_DT, dtype=torch.float32, device=DEVICE)
    s = states_t[:B]
    for _ in range(N_WARMUP):
        _ = model(s, dt)
    if DEVICE == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(N_REPEATS):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        _ = model(s, dt)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append(time.time() - t0)
    return float(np.median(times)) * 1000 / B


def time_solver(s_init_np: np.ndarray):
    """Single trajectory at h=64."""
    env_cfg = {
        "solver": {"params": {"grid": [128, 128], "domain": [0, 1, 0, 1],
                                "gamma": 1.4, "cfl": 0.4}},
    }
    p = env_cfg["solver"]["params"]
    domain = tuple(p["domain"])
    nx, ny = p["grid"]
    dx = (domain[1] - domain[0]) / nx
    dy = (domain[3] - domain[2]) / ny

    times = []
    for _ in range(N_REPEATS + 1):
        env_obj = env_mod.Env({
            "env": {
                "solver": {"params": p},
                "base_dt": BASE_DT,
                "event_detector": {"params": {"threshold": 0.05}},
                "trajectory_length": H,
            }
        })
        env_obj._q = np.transpose(s_init_np, (1, 2, 0)).astype(np.float64)
        env_obj._dx = dx
        env_obj._dy = dy
        t0 = time.time()
        env_obj._step_fixed_dt(TARGET_DT)
        times.append(time.time() - t0)
    return float(np.median(times[1:])) * 1000


def main():
    print(f"[euler_bench] device={DEVICE}", flush=True)
    model = load_model(str(CKPT), device=DEVICE)
    print(f"[euler_bench] model loaded", flush=True)

    ds = Euler2DDataset(str(DATA))
    states_t = torch.stack([ds.frame(i, 0) for i in range(min(64, ds.N))]).to(DEVICE)
    print(f"[euler_bench] loaded states {states_t.shape}", flush=True)

    s_init_np = states_t[0].cpu().numpy()
    print(f"\n[euler_bench] timing solver ...", flush=True)
    solver_ms = time_solver(s_init_np)
    print(f"  solver: {solver_ms:.1f} ms/traj", flush=True)

    print(f"\n[euler_bench] timing surrogate ...", flush=True)
    sur_ms = {}
    for B in BATCH_SIZES:
        if B > states_t.shape[0]: break
        ms = time_surrogate_batch(model, states_t, B)
        sur_ms[B] = ms
        print(f"  B={B:3d}: {ms:.3f} ms/pair  speedup={solver_ms/ms:.0f}x", flush=True)

    out = {
        "env": "euler", "horizon": H, "target_dt": TARGET_DT, "device": DEVICE,
        "n_repeats": N_REPEATS,
        "solver_ms_per_traj": solver_ms,
        "surrogate_ms_per_pair_by_batch": sur_ms,
        "speedup_by_batch": {B: solver_ms/m for B, m in sur_ms.items()},
    }
    out_path = RESULTS / "euler_h64.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
