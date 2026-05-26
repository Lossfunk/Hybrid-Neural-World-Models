#!/usr/bin/env python3
"""Surrogate-vs-solver batch benchmark — Ball 3D."""
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
sys.path.insert(0, str(ROOT / "training" / "ball3d"))
sys.path.insert(0, str(ROOT / "data_generation" / "ball3d"))

from shortcut_ball3d import ShortcutBall3D
from ball3d_env import Ball3DEnv

CKPT = ROOT / "checkpoints" / "ball3d" / "best.pt"
DATA = ROOT / "data" / "ball3d" / "ball3d_test.h5"
RESULTS = HERE / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

H = 64
DT_BASE = 0.01
TARGET_DT = H * DT_BASE
BATCH_SIZES = [1, 4, 16, 32, 64, 128, 256]
N_REPEATS = 50
N_WARMUP = 5


def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = ShortcutBall3D(
        state_dim=9, hidden_dim=cfg["hidden_dim"], emb_dim=cfg["emb_dim"],
        n_blocks=cfg["n_blocks"],
        ch_mean=ckpt["ch_mean"], ch_std=ckpt["ch_std"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


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


def time_solver(s_init_np: np.ndarray, restitution: float, gravity: float):
    times = []
    for _ in range(N_REPEATS + 5):
        env = Ball3DEnv()
        env._state = s_init_np.astype(np.float64).copy()
        env._restitution = float(restitution)
        env._gravity_vec = np.array([0.0, 0.0, float(gravity)])
        t0 = time.time()
        env.step(TARGET_DT)
        times.append(time.time() - t0)
    return float(np.median(times[5:])) * 1000


def main():
    print(f"[ball_bench] device={DEVICE}", flush=True)
    model = load_model(str(CKPT))
    print(f"[ball_bench] model loaded", flush=True)

    with h5py.File(DATA, "r") as f:
        states = np.array(f["states"][:256, 0], dtype=np.float32)
        rest = float(f.attrs.get("restitution", 1.0))
        grav = float(f.attrs.get("gravity_z", 0.0))
    states_t = torch.from_numpy(states).to(DEVICE)
    print(f"[ball_bench] loaded states {states_t.shape}  rest={rest}  g={grav}", flush=True)

    print(f"\n[ball_bench] timing solver ...", flush=True)
    solver_ms = time_solver(states[0], rest, grav)
    print(f"  solver: {solver_ms:.4f} ms/traj", flush=True)

    print(f"\n[ball_bench] timing surrogate ...", flush=True)
    sur_ms = {}
    for B in BATCH_SIZES:
        if B > states_t.shape[0]: break
        ms = time_surrogate_batch(model, states_t, B)
        sur_ms[B] = ms
        print(f"  B={B:4d}: {ms:.4f} ms/pair  speedup={solver_ms/ms:.2f}x", flush=True)

    out = {
        "env": "ball3d", "horizon": H, "target_dt": TARGET_DT, "device": DEVICE,
        "n_repeats": N_REPEATS,
        "solver_ms_per_traj": solver_ms,
        "surrogate_ms_per_pair_by_batch": sur_ms,
        "speedup_by_batch": {B: solver_ms/m for B, m in sur_ms.items()},
    }
    out_path = RESULTS / "ball3d_h64.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
