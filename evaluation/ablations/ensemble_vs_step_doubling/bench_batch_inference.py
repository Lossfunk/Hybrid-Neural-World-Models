#!/usr/bin/env python3
"""Batch-inference speed: step-doubling (1 model, 3 sequential passes) vs
K=3 ensemble (3 models, parallel-resident). Measures wall time per pair
amortized at batch size B for both signals on Euler and Oregonator.

Output: results/batch_speed.json + a paper-quality figure.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_REP = 10
WARM = 3


def bench_step_doubling(model, states, dt, n_rep=N_REP):
    B = states.shape[0]
    dt_t = torch.full((B,), dt, dtype=torch.float32, device=DEVICE)
    half_t = dt_t * 0.5
    for _ in range(WARM):
        with torch.no_grad():
            pred_full = model(states, dt_t)
            pred_mid = model(states, half_t)
            _ = model(pred_mid, half_t)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        with torch.no_grad():
            pred_full = model(states, dt_t)
            pred_mid = model(states, half_t)
            _ = model(pred_mid, half_t)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_rep / B * 1000


def bench_ensemble_resident(models, states, dt, n_rep=N_REP):
    """Sequential passes on K different models, all resident in memory.
    On single GPU this is the realistic scenario (you can't do 3 forward
    passes truly in parallel on one device without multi-stream tricks)."""
    B = states.shape[0]
    dt_t = torch.full((B,), dt, dtype=torch.float32, device=DEVICE)
    for _ in range(WARM):
        with torch.no_grad():
            for m in models:
                _ = m(states, dt_t)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        with torch.no_grad():
            for m in models:
                _ = m(states, dt_t)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_rep / B * 1000


def memory_resident(models, states, dt):
    """Measure peak GPU memory when keeping K models + B batched activations."""
    if DEVICE != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    B = states.shape[0]
    dt_t = torch.full((B,), dt, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for m in models:
            _ = m(states, dt_t)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9        # GB


def run_euler():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils_euler import load_model
    from data_utils_2d import Euler2DDataset

    ckpt_base = ROOT / "checkpoints" / "euler2d"
    models = [load_model(str(ckpt_base / f"seed{s}" / "best.pt"), device=DEVICE)
              for s in [0, 1, 2]]

    ds = Euler2DDataset(str(ROOT / "data" / "euler2d" / "euler2d_v2_test.h5"))
    s0 = ds.frame(2, 5).to(DEVICE)
    base_dt = ds.dt

    out = {}
    for B in [1, 4, 16, 32, 64]:
        states = s0.unsqueeze(0).repeat(B, 1, 1, 1)
        for h in [2, 8, 32, 64]:
            try:
                dt_target = h * base_dt
                sd_ms = bench_step_doubling(models[0], states, dt_target)
                en_ms = bench_ensemble_resident(models, states, dt_target)
                gc.collect(); torch.cuda.empty_cache()
                out_key = f"B{B}_h{h}"
                out[out_key] = {"sd_ms_per_pair": sd_ms,
                                "ensemble_ms_per_pair": en_ms,
                                "speedup_sd_over_ens": en_ms / sd_ms,
                                "B": B, "h": h}
                print(f"  Euler  B={B:2d}  h={h:3d}: SD={sd_ms:.3f} ms/pair  "
                        f"ENS={en_ms:.3f} ms/pair  ratio={en_ms/sd_ms:.2f}x",
                        flush=True)
            except RuntimeError as e:
                print(f"  Euler  B={B:2d}  h={h:3d}: OOM ({str(e)[:60]})", flush=True)
                out[f"B{B}_h{h}"] = {"oom": True, "B": B, "h": h}

    # Memory snapshot
    states = s0.unsqueeze(0).repeat(64, 1, 1, 1)
    mem_resident_3 = memory_resident(models, states, 64 * base_dt)
    mem_resident_1 = memory_resident([models[0]], states, 64 * base_dt)
    out["memory_gb"] = {"sd_1_model": mem_resident_1,
                          "ensemble_3_models": mem_resident_3}
    print(f"  Euler memory at B=64, h=64:  SD={mem_resident_1:.2f} GB  "
            f"ENS={mem_resident_3:.2f} GB", flush=True)
    ds.close()
    return out


def run_oregonator():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "training" / "oregonator"))
    from eval_utils import load_model
    ckpt_base = ROOT / "checkpoints" / "oregonator"
    models = [load_model(str(ckpt_base / f"seed{s}" / "best.pt"), device=DEVICE)
              for s in [0, 1, 2]]
    f = h5py.File(ROOT / "data" / "oregonator" /
                    "oregonator_test.h5", "r")
    s0 = torch.from_numpy(f["states"][2, 20]).float().to(DEVICE)
    base_dt = float(f.attrs["dt_save"])
    f.close()

    out = {}
    for B in [1, 4, 8, 16, 32, 64]:
        states = s0.unsqueeze(0).repeat(B, 1, 1, 1)
        for h in [2, 8, 32, 64]:
            try:
                dt_target = h * base_dt
                sd_ms = bench_step_doubling(models[0], states, dt_target)
                en_ms = bench_ensemble_resident(models, states, dt_target)
                gc.collect(); torch.cuda.empty_cache()
                out[f"B{B}_h{h}"] = {"sd_ms_per_pair": sd_ms,
                                       "ensemble_ms_per_pair": en_ms,
                                       "speedup_sd_over_ens": en_ms / sd_ms,
                                       "B": B, "h": h}
                print(f"  Oreg   B={B:2d}  h={h:3d}: SD={sd_ms:.3f} ms/pair  "
                        f"ENS={en_ms:.3f} ms/pair  ratio={en_ms/sd_ms:.2f}x",
                        flush=True)
            except RuntimeError as e:
                print(f"  Oreg   B={B:2d}  h={h:3d}: OOM ({str(e)[:60]})", flush=True)
                out[f"B{B}_h{h}"] = {"oom": True, "B": B, "h": h}

    states = s0.unsqueeze(0).repeat(32, 1, 1, 1)
    try:
        mem_resident_3 = memory_resident(models, states, 64 * base_dt)
        mem_resident_1 = memory_resident([models[0]], states, 64 * base_dt)
    except RuntimeError:
        mem_resident_3 = mem_resident_1 = float("nan")
    out["memory_gb"] = {"sd_1_model": mem_resident_1,
                          "ensemble_3_models": mem_resident_3}
    print(f"  Oreg memory at B=32, h=64:  SD={mem_resident_1:.2f} GB  "
            f"ENS={mem_resident_3:.2f} GB", flush=True)
    return out


def main():
    res = {}
    print("=== Euler batch-inference benchmark ===", flush=True)
    res["euler"] = run_euler()
    print("\n=== Oregonator batch-inference benchmark ===", flush=True)
    res["oregonator"] = run_oregonator()

    out_path = HERE / "results" / "batch_speed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
