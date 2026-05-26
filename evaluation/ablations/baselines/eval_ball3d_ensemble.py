#!/usr/bin/env python3
"""Ensemble K=3 disagreement vs step-doubling on Ball 3D.

For each (split, horizon), use the 3 trained Ball3D seeds, compute the
per-trajectory disagreement (std of predictions across seeds, norm across
state dims) and compare against step-doubling AUROC.

Outputs: ablations/baselines/results/ball3d_ensemble.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "training" / "ball3d"))

from shortcut_ball3d import ShortcutBall3D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT_BASE = 0.01
HORIZONS = [2, 4, 8, 16, 32, 64]
SPLITS = ["test", "ood_near", "ood_far"]
N_PAIRS_PER_H = 80
SEEDS = [0, 1, 2]


def load_model(seed):
    ck = torch.load(str(ROOT / "checkpoints" / "ball3d" / f"seed{seed}" / "best.pt"),
                     map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    m = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                          emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                          ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


@torch.no_grad()
def step_doubling(model, s, dt):
    if s.dim() == 1: s = s.unsqueeze(0)
    B = s.shape[0]
    dt_t = torch.full((B,), float(dt), device=s.device, dtype=torch.float32)
    half_t = dt_t * 0.5
    pf = model(s, dt_t)
    pm = model(s, half_t)
    pc = model(pm, half_t)
    e = torch.sqrt(((pf - pc) ** 2).sum(dim=1) + 1e-12)
    return e, pf


@torch.no_grad()
def ensemble_disagreement(models, s, dt):
    if s.dim() == 1: s = s.unsqueeze(0)
    B = s.shape[0]
    dt_t = torch.full((B,), float(dt), device=s.device, dtype=torch.float32)
    preds = torch.stack([m(s, dt_t) for m in models], dim=0)  # (K, B, 9)
    std = preds.std(dim=0, unbiased=False)                    # (B, 9)
    return torch.sqrt((std ** 2).sum(dim=1) + 1e-12)          # (B,)


def auroc_at_q75(scores, true_errs):
    sc = np.array(scores); te = np.array(true_errs)
    if len(sc) < 4: return float("nan")
    thr = float(np.quantile(te, 0.75))
    lbl = (te > thr).astype(int)
    if lbl.sum() == 0 or lbl.sum() == len(lbl): return float("nan")
    return float(roc_auc_score(lbl, sc))


def eval_split(models, split):
    data = ROOT / "data" / "ball3d" / f"ball3d_{split}.h5"
    rng = np.random.RandomState(0)
    out = {}
    with h5py.File(data, "r") as f:
        N, T = f["states"].shape[:2]
        for h in HORIZONS:
            sd_scores, en_scores, te_vals = [], [], []
            for _ in range(N_PAIRS_PER_H):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s = torch.from_numpy(np.array(f["states"][i, t0])).to(DEVICE)
                gt = torch.from_numpy(np.array(f["states"][i, t0 + h])).to(DEVICE)
                target_dt = h * DT_BASE
                with torch.no_grad():
                    sd_e, pf = step_doubling(models[0], s, target_dt)
                    en_e = ensemble_disagreement(models, s, target_dt)
                    te = torch.sqrt(((pf - gt) ** 2).mean()).item()
                sd_scores.append(float(sd_e[0].item()))
                en_scores.append(float(en_e[0].item()))
                te_vals.append(float(te))
            out[h] = {
                "auroc_sd": auroc_at_q75(sd_scores, te_vals),
                "auroc_ensemble": auroc_at_q75(en_scores, te_vals),
                "n_pairs": len(sd_scores),
            }
            print(f"  ball3d {split} h={h:2d}  SD={out[h]['auroc_sd']:.4f}  "
                  f"ENS={out[h]['auroc_ensemble']:.4f}", flush=True)
    return out


def main():
    print(f"[ball3d_ensemble] device={DEVICE}", flush=True)
    print(f"[ball3d_ensemble] loading 3 seeds ...", flush=True)
    models = [load_model(s) for s in SEEDS]
    print(f"[ball3d_ensemble] all loaded", flush=True)

    out = {}
    for split in SPLITS:
        print(f"\n=== {split} ===", flush=True)
        out[split] = eval_split(models, split)

    out_path = HERE / "results" / "ball3d_ensemble.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
