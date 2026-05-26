#!/usr/bin/env python3
"""Random test-time augmentation (TTA) UQ baseline on Euler 2D and Ball 3D.

For each (env, split, horizon), perturb the input with K=4 small Gaussian
noises and compute the per-trajectory standard deviation of predictions.
Compare AUROC to step-doubling.

Outputs:
  ablations/baselines/results/tta_euler.json
  ablations/baselines/results/tta_ball3d.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HORIZONS = [2, 4, 8, 16, 32, 64]
SPLITS = ["test", "ood_near", "ood_far"]
N_PAIRS_PER_H = 80
K_TTA = 4
SIGMA_REL = 0.01


def auroc_q75(scores, te):
    sc = np.array(scores); tv = np.array(te)
    if len(sc) < 4: return float("nan")
    thr = float(np.quantile(tv, 0.75))
    lbl = (tv > thr).astype(int)
    if lbl.sum() == 0 or lbl.sum() == len(lbl): return float("nan")
    return float(roc_auc_score(lbl, sc))


def run_euler():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils_euler import (load_model, predict, step_doubling_estimator,
                                       true_error)
    from data_utils_2d import Euler2DDataset

    ckpt = (ROOT / "checkpoints" / "euler2d" / "best.pt")
    model = load_model(str(ckpt), device=DEVICE)
    DATA = ROOT / "data" / "euler2d"

    out = {}
    for split in SPLITS:
        print(f"\n=== Euler {split} ===", flush=True)
        ds = Euler2DDataset(str(DATA / f"euler2d_v2_{split}.h5"))
        rng = np.random.RandomState(0)
        ch_std = model.ch_std.view(1, -1, 1, 1).to(DEVICE)
        out[split] = {}
        for h in HORIZONS:
            if h >= ds.T: continue
            target_dt = h * ds.dt
            sd_scores, tta_scores, te_vals = [], [], []
            for _ in range(N_PAIRS_PER_H):
                i = int(rng.randint(0, ds.N))
                t0 = int(rng.randint(0, ds.T - h))
                u0 = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
                ut = ds.frame(i, t0 + h).to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    sd_map, pf = step_doubling_estimator(model, u0, target_dt)
                    te_map = true_error(pf, ut)
                    preds = []
                    for k in range(K_TTA):
                        eps = torch.randn_like(u0) * (ch_std * SIGMA_REL)
                        preds.append(predict(model, u0 + eps, target_dt))
                    stack = torch.stack(preds, dim=0)
                    std_per_cell = stack.std(dim=0)
                    tta_map = torch.sqrt((std_per_cell ** 2).sum(dim=1) + 1e-12)
                sd_scores.append(float(sd_map.mean().cpu().numpy()))
                tta_scores.append(float(tta_map.mean().cpu().numpy()))
                te_vals.append(float(te_map.mean().cpu().numpy()))
            out[split][h] = {
                "auroc_sd": auroc_q75(sd_scores, te_vals),
                "auroc_tta": auroc_q75(tta_scores, te_vals),
            }
            print(f"  euler {split} h={h:2d}  SD={out[split][h]['auroc_sd']:.4f}  "
                  f"TTA={out[split][h]['auroc_tta']:.4f}", flush=True)

    out_path = HERE / "results" / "tta_euler.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


def run_ball3d():
    sys.path.insert(0, str(ROOT / "training" / "ball3d"))
    from shortcut_ball3d import ShortcutBall3D

    DT_BASE = 0.01
    ckpt = ROOT / "checkpoints" / "ball3d" / "best.pt"
    ck = torch.load(str(ckpt), map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    model = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                              emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                              ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    DATA_DIR = ROOT / "data" / "ball3d"
    ch_std = model.ch_std.view(1, -1).to(DEVICE)

    out = {}
    for split in SPLITS:
        print(f"\n=== Ball3D {split} ===", flush=True)
        with h5py.File(DATA_DIR / f"ball3d_{split}.h5", "r") as f:
            N, T = f["states"].shape[:2]
            states = np.array(f["states"], dtype=np.float32)
        rng = np.random.RandomState(0)
        out[split] = {}
        for h in HORIZONS:
            if h >= T: continue
            target_dt = h * DT_BASE
            sd_scores, tta_scores, te_vals = [], [], []
            for _ in range(N_PAIRS_PER_H):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s = torch.from_numpy(states[i, t0]).unsqueeze(0).to(DEVICE)
                gt = torch.from_numpy(states[i, t0 + h]).unsqueeze(0).to(DEVICE)
                dt_t = torch.full((1,), target_dt, dtype=torch.float32, device=DEVICE)
                half_t = dt_t * 0.5
                with torch.no_grad():
                    pf = model(s, dt_t)
                    pm = model(s, half_t)
                    pc = model(pm, half_t)
                    sd_e = float(torch.sqrt(((pf - pc) ** 2).sum() + 1e-12).item())
                    preds = []
                    for k in range(K_TTA):
                        eps = torch.randn_like(s) * (ch_std * SIGMA_REL)
                        preds.append(model(s + eps, dt_t))
                    stack = torch.stack(preds, dim=0)
                    std = stack.std(dim=0, unbiased=False)
                    tta_e = float(torch.sqrt((std ** 2).sum() + 1e-12).item())
                    te = float(torch.sqrt(((pf - gt) ** 2).mean()).item())
                sd_scores.append(sd_e); tta_scores.append(tta_e); te_vals.append(te)
            out[split][h] = {
                "auroc_sd": auroc_q75(sd_scores, te_vals),
                "auroc_tta": auroc_q75(tta_scores, te_vals),
            }
            print(f"  ball3d {split} h={h:2d}  SD={out[split][h]['auroc_sd']:.4f}  "
                  f"TTA={out[split][h]['auroc_tta']:.4f}", flush=True)

    out_path = HERE / "results" / "tta_ball3d.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="both", choices=["euler", "ball3d", "both"])
    args = ap.parse_args()
    if args.env in ("euler", "both"):
        run_euler()
    if args.env in ("ball3d", "both"):
        run_ball3d()


if __name__ == "__main__":
    main()
