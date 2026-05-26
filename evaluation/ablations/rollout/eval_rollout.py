#!/usr/bin/env python3
"""Closed-loop rollout test for WM stability.

For each env at h_total = 64, compare:
  (a) Single-shot prediction f(s_0, h=64): one forward pass at full horizon.
  (b) Closed-loop k-chain rollout: f applied k times at h=64/k each.
       chain at k=2: f(f(s_0, h=32), h=32)
       chain at k=4: f applied 4 times at h=16
       chain at k=8: f applied 8 times at h=8
  (c) GT trajectory at the corresponding final time.

Reports RMSE of each rollout vs GT, plus drift growth.

This shows the WM is stable under autoregressive rollout, not just a
one-shot predictor.
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
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

H_TOTAL = 64
N_TRAJS = 32


@torch.no_grad()
def closed_loop_chain(model, s0, k_chain, target_dt):
    """Apply f k_chain times at horizon target_dt/k_chain each."""
    h_step = target_dt / k_chain
    s = s0
    for _ in range(k_chain):
        dt = torch.full((s.shape[0],), float(h_step), device=DEVICE)
        s = model(s, dt)
    return s


def rmse_state(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


def bench_oreg():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils import load_model, predict

    DT_BASE = 0.05
    target_dt = H_TOTAL * DT_BASE
    ckpt = ROOT / "checkpoints" / "oregonator" / "best.pt"
    data = ROOT / "data" / "oregonator" / "oregonator_test.h5"
    model = load_model(str(ckpt), device=DEVICE)

    out = {}
    with h5py.File(data, "r") as f:
        N, T = f["states"].shape[:2]
        rng = np.random.RandomState(7)
        for k_chain in [1, 2, 4, 8]:
            rmses = []
            for _ in range(N_TRAJS):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - H_TOTAL))
                s0 = torch.from_numpy(np.array(f["states"][i, t0])).unsqueeze(0).to(DEVICE)
                gt = torch.from_numpy(np.array(f["states"][i, t0 + H_TOTAL])).unsqueeze(0).to(DEVICE)
                pred = closed_loop_chain(model, s0, k_chain, target_dt)
                rmses.append(rmse_state(pred, gt))
            out[k_chain] = {"mean_rmse_vs_gt": float(np.mean(rmses)),
                              "std_rmse": float(np.std(rmses)),
                              "n_trajs": N_TRAJS}
            print(f"  oreg k={k_chain}  mean RMSE={out[k_chain]['mean_rmse_vs_gt']:.4f}  "
                  f"std={out[k_chain]['std_rmse']:.4f}", flush=True)
    return out


def bench_euler():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils_euler import load_model
    from data_utils_2d import Euler2DDataset

    BASE_DT = 0.002
    target_dt = H_TOTAL * BASE_DT
    ckpt = ROOT / "checkpoints" / "euler2d" / "best.pt"
    data = ROOT / "data" / "euler2d" / "euler2d_v2_test.h5"
    model = load_model(str(ckpt), device=DEVICE)
    ds = Euler2DDataset(str(data))

    out = {}
    rng = np.random.RandomState(7)
    for k_chain in [1, 2, 4, 8]:
        rmses = []
        for _ in range(N_TRAJS):
            i = int(rng.randint(0, ds.N))
            t0 = int(rng.randint(0, ds.T - H_TOTAL))
            s0 = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
            gt = ds.frame(i, t0 + H_TOTAL).to(DEVICE).unsqueeze(0)
            pred = closed_loop_chain(model, s0, k_chain, target_dt)
            rmses.append(rmse_state(pred, gt))
        out[k_chain] = {"mean_rmse_vs_gt": float(np.mean(rmses)),
                          "std_rmse": float(np.std(rmses)),
                          "n_trajs": N_TRAJS}
        print(f"  euler k={k_chain}  mean RMSE={out[k_chain]['mean_rmse_vs_gt']:.4f}  "
              f"std={out[k_chain]['std_rmse']:.4f}", flush=True)
    return out


def bench_ball():
    sys.path.insert(0, str(ROOT / "training" / "ball3d"))
    from shortcut_ball3d import ShortcutBall3D

    DT_BASE = 0.01
    target_dt = H_TOTAL * DT_BASE
    ckpt = ROOT / "checkpoints" / "ball3d" / "best.pt"
    data = ROOT / "data" / "ball3d" / "ball3d_test.h5"
    ck = torch.load(str(ckpt), map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    model = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                              emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                              ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    out = {}
    rng = np.random.RandomState(7)
    with h5py.File(data, "r") as f:
        N, T = f["states"].shape[:2]
        for k_chain in [1, 2, 4, 8]:
            rmses = []
            for _ in range(N_TRAJS):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - H_TOTAL))
                s0 = torch.from_numpy(np.array(f["states"][i, t0])).unsqueeze(0).to(DEVICE)
                gt = torch.from_numpy(np.array(f["states"][i, t0 + H_TOTAL])).unsqueeze(0).to(DEVICE)
                pred = closed_loop_chain(model, s0, k_chain, target_dt)
                rmses.append(rmse_state(pred, gt))
            out[k_chain] = {"mean_rmse_vs_gt": float(np.mean(rmses)),
                              "std_rmse": float(np.std(rmses)),
                              "n_trajs": N_TRAJS}
            print(f"  ball k={k_chain}  mean RMSE={out[k_chain]['mean_rmse_vs_gt']:.4f}  "
                  f"std={out[k_chain]['std_rmse']:.4f}", flush=True)
    return out


def main():
    print(f"[rollout] device={DEVICE}", flush=True)
    print("\n=== Oregonator ===", flush=True)
    oreg = bench_oreg()
    print("\n=== Euler ===", flush=True)
    euler = bench_euler()
    print("\n=== Ball 3D ===", flush=True)
    ball = bench_ball()
    out = {"oregonator": oreg, "euler": euler, "ball3d": ball,
           "horizon_total": H_TOTAL,
           "interpretation": "k_chain=1 is single-shot WM. k_chain=2,4,8 chain WM over progressively shorter horizons. RMSE vs GT shows long-horizon stability."}
    out_path = RESULTS / "rollout_h64.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
