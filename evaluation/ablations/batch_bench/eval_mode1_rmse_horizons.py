#!/usr/bin/env python3
"""Compute Mode 1 RMSE at multiple horizons for Oregonator and Euler.

For each (env, h in {2, 4, 8, 16, 32, 64}), run the surrogate at horizon h
on N test trajectories and compute mean RMSE against ground truth. This
gives the data needed to plot Mode 1 as a curve in (RMSE, speedup) space.

Output: ablations/batch_bench/results/m1_rmse_horizons.json
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
RESULTS.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HORIZONS = [2, 4, 8, 16, 32, 64]
N_TRAJ = 80


def bench_oreg():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils import load_model, predict
    DT_BASE = 0.05
    ckpt = ROOT / "checkpoints" / "oregonator" / "best.pt"
    data = ROOT / "data" / "oregonator" / "oregonator_test.h5"
    model = load_model(str(ckpt), device=DEVICE)

    rng = np.random.RandomState(7)
    out = {}
    with h5py.File(data, "r") as f:
        N, T = f["states"].shape[:2]
        for h in HORIZONS:
            rmses = []
            for _ in range(N_TRAJ):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s = torch.from_numpy(np.array(f["states"][i, t0])).unsqueeze(0).to(DEVICE)
                gt = torch.from_numpy(np.array(f["states"][i, t0 + h])).unsqueeze(0).to(DEVICE)
                dt = torch.tensor([h * DT_BASE], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    pred = predict(model, s, dt)
                rmse = float(torch.sqrt(((pred - gt) ** 2).mean()).item())
                rmses.append(rmse)
            out[h] = float(np.mean(rmses))
            print(f"  oreg h={h:2d}  mean RMSE = {out[h]:.4f}", flush=True)
    return out


def bench_euler():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils_euler import load_model
    from data_utils_2d import Euler2DDataset
    BASE_DT = 0.002
    ckpt = ROOT / "checkpoints" / "euler2d" / "best.pt"
    data = ROOT / "data" / "euler2d" / "euler2d_v2_test.h5"
    model = load_model(str(ckpt), device=DEVICE)
    ds = Euler2DDataset(str(data))

    rng = np.random.RandomState(7)
    out = {}
    for h in HORIZONS:
        rmses = []
        for _ in range(N_TRAJ):
            i = int(rng.randint(0, ds.N))
            t0 = int(rng.randint(0, ds.T - h))
            u0 = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
            ut = ds.frame(i, t0 + h).to(DEVICE).unsqueeze(0)
            dt = torch.tensor([h * BASE_DT], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                pred = model(u0, dt)
            rmse = float(torch.sqrt(((pred - ut) ** 2).mean()).item())
            rmses.append(rmse)
        out[h] = float(np.mean(rmses))
        print(f"  euler h={h:2d}  mean RMSE = {out[h]:.4f}", flush=True)
    return out


def main():
    print(f"[m1 rmse] device={DEVICE}", flush=True)
    print("\n=== Oregonator ===", flush=True)
    oreg = bench_oreg()
    print("\n=== Euler ===", flush=True)
    euler = bench_euler()
    out = {"oregonator": oreg, "euler": euler, "n_trajs": N_TRAJ,
           "horizons": HORIZONS}
    out_path = RESULTS / "m1_rmse_horizons.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
