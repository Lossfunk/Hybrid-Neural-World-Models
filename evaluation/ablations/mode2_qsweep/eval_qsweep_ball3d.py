#!/usr/bin/env python3
"""Mode 2 q-sweep for Ball 3D."""
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = [32, 64]
QS = [0.5, 0.6, 0.75, 0.85, 0.9]
N_PAIRS = 80
DT_BASE = 0.01

CKPT = ROOT / "checkpoints" / "ball3d" / "best.pt"


def mode2_at_q(sd_score, m1_rmse, q):
    sd = np.array(sd_score); m1 = np.array(m1_rmse)
    thr = float(np.quantile(sd, q))
    trig = sd > thr
    untrig = ~trig
    untrig_m1 = float(m1[untrig].mean()) if untrig.sum() > 0 else float("nan")
    m2 = untrig_m1 * (untrig.sum() / len(m1))
    return {"q": q, "threshold": thr, "trigger_rate": float(trig.mean()),
             "mode2_rmse": m2}


def run(split):
    ck = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    model = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                              emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                              ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    data = ROOT / "data" / "ball3d" / f"ball3d_{split}.h5"
    rng = np.random.RandomState(7)
    out = {}
    with h5py.File(data, "r") as f:
        states = np.array(f["states"], dtype=np.float32)  # (N, T, 9)
        N, T = states.shape[:2]
        for h in HORIZONS:
            sd_scores, m1_rmses = [], []
            for _ in range(N_PAIRS):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s = torch.from_numpy(states[i, t0]).unsqueeze(0).to(DEVICE)
                gt = torch.from_numpy(states[i, t0 + h]).unsqueeze(0).to(DEVICE)
                target_dt = h * DT_BASE
                dt_t = torch.full((1,), target_dt, dtype=torch.float32, device=DEVICE)
                half_t = dt_t * 0.5
                with torch.no_grad():
                    pf = model(s, dt_t)
                    pm = model(s, half_t)
                    pc = model(pm, half_t)
                e = float(torch.sqrt(((pf - pc) ** 2).sum() + 1e-12).item())
                rmse = float(torch.sqrt(((pf - gt) ** 2).mean()).item())
                sd_scores.append(e); m1_rmses.append(rmse)
            out[h] = {"n_pairs": N_PAIRS, "m1_rmse_mean": float(np.mean(m1_rmses)),
                       "qsweep": {q: mode2_at_q(sd_scores, m1_rmses, q) for q in QS}}
            qs = out[h]["qsweep"]
            print(f"  ball3d {split} h={h}  M1={out[h]['m1_rmse_mean']:.4f}  "
                  f"q05/q75/q09 M2 = {qs[0.5]['mode2_rmse']:.4f}/"
                  f"{qs[0.75]['mode2_rmse']:.4f}/{qs[0.9]['mode2_rmse']:.4f}",
                  flush=True)

    out_path = HERE / "results" / f"ball3d_{split}.json"
    out_path.write_text(json.dumps(out, indent=2))


def main():
    for split in ["test", "ood_near", "ood_far"]:
        print(f"\n=== ball3d {split} ===", flush=True)
        run(split)


if __name__ == "__main__":
    main()
