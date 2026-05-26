#!/usr/bin/env python3
"""Surrogate-vs-solver batch benchmark — Oregonator.

Measures:
  - Surrogate ms/pair at batch B in {1, 4, 16, 32, 64} on GPU (h=64)
  - Solver ms/pair (single trajectory, CPU; solver is sequential anyway)
  - Speedup vs solver at each batch size

Output: ablations/batch_bench/results/oregonator_h64.json
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
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "data_generation" / "oregonator"))

from eval_utils import load_model, predict
from oregonator2d_tyson import OregonatorTyson2D, TysonParams

CKPT = ROOT / "checkpoints" / "oregonator" / "best.pt"
DATA = ROOT / "data" / "oregonator" / "oregonator_test.h5"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

H = 64
DT_BASE = 0.05
TARGET_DT = H * DT_BASE
BATCH_SIZES = [1, 2, 4, 8, 16]
N_REPEATS = 5
N_WARMUP = 3


@torch.no_grad()
def time_surrogate_batch(model, states_t: torch.Tensor, B: int):
    dt = torch.full((B,), TARGET_DT, dtype=torch.float32, device=DEVICE)
    s = states_t[:B]
    # warmup
    for _ in range(N_WARMUP):
        _ = predict(model, s, dt)
    if DEVICE == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(N_REPEATS):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        _ = predict(model, s, dt)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append(time.time() - t0)
    median_total_s = float(np.median(times))
    return median_total_s * 1000 / B  # ms/pair


def time_solver(states_np: np.ndarray, params_list: list):
    """Time a single trajectory at h=64. Multiple trajectories run sequentially."""
    s_init = states_np[0]
    p = params_list[0]
    sim = OregonatorTyson2D(
        n_x=s_init.shape[2], n_y=s_init.shape[1],
        L_x=100.0, L_y=100.0,
        params=TysonParams(**p),
    )
    # warmup
    sim.u[:] = s_init[0]; sim.v[:] = s_init[1]; sim.t_sim = 0
    for _ in range(2):
        sim.step(DT_BASE)
    times = []
    for _ in range(N_REPEATS):
        sim.u[:] = s_init[0]; sim.v[:] = s_init[1]; sim.t_sim = 0
        t0 = time.time()
        for _ in range(H):
            sim.step(DT_BASE)
        times.append(time.time() - t0)
    return float(np.median(times)) * 1000  # ms/traj


def main():
    print(f"[oreg_bench] device={DEVICE}", flush=True)
    model = load_model(str(CKPT), device=DEVICE)
    print(f"[oreg_bench] model loaded", flush=True)

    # Load 64 trajectories (initial frames) + their params for the solver
    with h5py.File(DATA, "r") as f:
        states = np.array(f["states"][:64, 0], dtype=np.float32)
        params = []
        for i in range(8):  # only need 1 for solver, sample a few
            params.append(dict(
                f=float(f["params"][i, 0]),
                eps=float(f["params"][i, 1]),
                q=float(f["params"][i, 2]),
                D=float(f["params"][i, 3]),
            ))
    states_t = torch.from_numpy(states).to(DEVICE)
    print(f"[oreg_bench] loaded states {states_t.shape}", flush=True)

    # Time solver (single traj, h=64)
    print(f"\n[oreg_bench] timing solver ...", flush=True)
    solver_ms_per_traj = time_solver(states[:1], params[:1])
    print(f"  solver: {solver_ms_per_traj:.1f} ms/traj", flush=True)

    # Time surrogate at each batch size
    print(f"\n[oreg_bench] timing surrogate (batch sizes {BATCH_SIZES}) ...", flush=True)
    sur_ms_per_pair = {}
    for B in BATCH_SIZES:
        if B > states_t.shape[0]:
            print(f"  B={B}: skipped (need more trajs)"); continue
        ms = time_surrogate_batch(model, states_t, B)
        sur_ms_per_pair[B] = ms
        print(f"  B={B:3d}: {ms:.3f} ms/pair  speedup={solver_ms_per_traj/ms:.0f}x",
              flush=True)

    out = {
        "env": "oregonator",
        "horizon": H,
        "target_dt": TARGET_DT,
        "device": DEVICE,
        "n_repeats": N_REPEATS,
        "solver_ms_per_traj": solver_ms_per_traj,
        "surrogate_ms_per_pair_by_batch": sur_ms_per_pair,
        "speedup_by_batch": {B: solver_ms_per_traj / ms
                              for B, ms in sur_ms_per_pair.items()},
    }
    out_path = RESULTS / "oregonator_h64.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
