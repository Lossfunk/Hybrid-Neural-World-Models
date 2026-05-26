#!/usr/bin/env python3
"""Mode 2 gate-threshold sensitivity sweep.

For each (env, split): collect (sd_score, true_rmse) per pair at h=64 and
h=32 (the long-horizon regimes where Mode 2 matters). Then sweep gate
thresholds q ∈ {0.5, 0.6, 0.75, 0.85, 0.9} and report the resulting
Mode 2 RMSE (average over untriggered pairs, scaled by untriggered fraction;
triggered pairs assumed to fall back to a perfect solver).

This defends our reported q=0.75 default against reviewer asks about
sensitivity.

Output: ablations/mode2_qsweep/results/{env}_{split}.json
"""
from __future__ import annotations

import argparse
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
HORIZONS = [32, 64]
QS = [0.5, 0.6, 0.75, 0.85, 0.9]
N_PAIRS = 80


@torch.no_grad()
def predict_pde(model, state, dt):
    if state.dim() == 3: state = state.unsqueeze(0)
    B = state.shape[0]
    if isinstance(dt, (float, int)):
        dt = torch.full((B,), float(dt), device=state.device)
    return model(state, dt)


@torch.no_grad()
def step_doubling(model, state, dt):
    pf = predict_pde(model, state, dt)
    pm = predict_pde(model, state, dt * 0.5)
    pc = predict_pde(model, pm, dt * 0.5)
    e = torch.sqrt(((pf - pc) ** 2).sum(dim=1) + 1e-12)
    return e, pf


def mode2_at_q(sd_score, m1_rmse, q: float):
    """Trust-aware Mode 2 RMSE at threshold q.
    Triggered (top 1-q fraction): assumed perfect solver (RMSE=0).
    Untriggered: keep Mode 1 RMSE."""
    sd = np.array(sd_score); m1 = np.array(m1_rmse)
    thr = float(np.quantile(sd, q))
    trig = sd > thr
    untrig = ~trig
    untrig_m1 = float(m1[untrig].mean()) if untrig.sum() > 0 else float("nan")
    m2 = untrig_m1 * (untrig.sum() / len(m1))
    return {
        "q": q,
        "threshold": thr,
        "trigger_rate": float(trig.mean()),
        "untriggered_m1_rmse": untrig_m1,
        "mode2_rmse": m2,
        "captured_high_m1_mean": float(m1[trig].mean()) if trig.sum() > 0 else float("nan"),
    }


def run_oregonator(split: str):
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils import load_model, true_error  # noqa: E402

    ckpt = ROOT / "checkpoints" / "oregonator" / "best.pt"
    model = load_model(str(ckpt), device=DEVICE)
    ds_path = ROOT / "data" / "oregonator" / f"oregonator_{split}.h5"
    DT_BASE = 0.05
    rng = np.random.RandomState(7)
    out = {}
    with h5py.File(ds_path, "r") as f:
        N, T = f["states"].shape[:2]
        for h in HORIZONS:
            sd_scores, m1_rmses = [], []
            t0_ = time.time()
            for _ in range(N_PAIRS):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s = torch.from_numpy(np.array(f["states"][i, t0])
                                       ).unsqueeze(0).to(DEVICE)
                tgt = torch.from_numpy(np.array(f["states"][i, t0 + h])
                                         ).unsqueeze(0).to(DEVICE)
                dt = torch.tensor([h * DT_BASE], dtype=torch.float32, device=DEVICE)
                e_map, pred = step_doubling(model, s, dt)
                rmse = float(torch.sqrt(((pred - tgt) ** 2).mean()).cpu().numpy())
                sd_scores.append(float(e_map.mean().cpu().numpy()))
                m1_rmses.append(rmse)
            out[h] = {
                "n_pairs": N_PAIRS,
                "elapsed_s": time.time() - t0_,
                "m1_rmse_mean": float(np.mean(m1_rmses)),
                "qsweep": {q: mode2_at_q(sd_scores, m1_rmses, q) for q in QS},
            }
            print(f"  oreg {split} h={h:2d}  M1={out[h]['m1_rmse_mean']:.4f}  "
                  f"q05/q75/q90 M2 = "
                  f"{out[h]['qsweep'][0.5]['mode2_rmse']:.4f}/"
                  f"{out[h]['qsweep'][0.75]['mode2_rmse']:.4f}/"
                  f"{out[h]['qsweep'][0.9]['mode2_rmse']:.4f}  "
                  f"({out[h]['elapsed_s']:.1f}s)", flush=True)
    out_path = HERE / "results" / f"oregonator_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))


def run_euler(split: str):
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils_euler import load_model  # noqa: E402
    from data_utils_2d import Euler2DDataset  # noqa: E402

    ckpt = (ROOT / "checkpoints" / "euler2d" / "best.pt")
    model = load_model(str(ckpt), device=DEVICE)
    ds_path = ROOT / "data" / "euler2d" / f"euler2d_v2_{split}.h5"
    ds = Euler2DDataset(str(ds_path))
    rng = np.random.RandomState(7)
    out = {}
    for h in HORIZONS:
        if h >= ds.T: continue
        sd_scores, m1_rmses = [], []
        dt_target = h * ds.dt
        t0_ = time.time()
        for _ in range(N_PAIRS):
            i = int(rng.randint(0, ds.N))
            t0 = int(rng.randint(0, ds.T - h))
            u0 = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
            ut = ds.frame(i, t0 + h).to(DEVICE).unsqueeze(0)
            e_map, pred = step_doubling(model, u0, dt_target)
            rmse = float(torch.sqrt(((pred - ut) ** 2).mean()).cpu().numpy())
            sd_scores.append(float(e_map.mean().cpu().numpy()))
            m1_rmses.append(rmse)
        out[h] = {
            "n_pairs": N_PAIRS,
            "elapsed_s": time.time() - t0_,
            "m1_rmse_mean": float(np.mean(m1_rmses)),
            "qsweep": {q: mode2_at_q(sd_scores, m1_rmses, q) for q in QS},
        }
        print(f"  euler {split} h={h:2d}  M1={out[h]['m1_rmse_mean']:.4f}  "
              f"q05/q75/q90 M2 = "
              f"{out[h]['qsweep'][0.5]['mode2_rmse']:.4f}/"
              f"{out[h]['qsweep'][0.75]['mode2_rmse']:.4f}/"
              f"{out[h]['qsweep'][0.9]['mode2_rmse']:.4f}  "
              f"({out[h]['elapsed_s']:.1f}s)", flush=True)
    out_path = HERE / "results" / f"euler_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="oregonator,euler")
    ap.add_argument("--splits", default="test,ood_near,ood_far")
    args = ap.parse_args()
    for env in args.envs.split(","):
        for split in args.splits.split(","):
            print(f"\n=== {env} {split} ===", flush=True)
            if env == "oregonator": run_oregonator(split)
            elif env == "euler":     run_euler(split)


if __name__ == "__main__":
    main()
