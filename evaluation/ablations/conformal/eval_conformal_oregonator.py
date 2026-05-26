#!/usr/bin/env python3
"""Locally-adaptive split conformal prediction baseline — Oregonator.

Trust signal: for each test trajectory, find K nearest neighbours in a
calibration set (using mean state per channel as the embedding) and use
the empirical 90th percentile of their true RMSE as the conformal upper
bound on this trajectory's error.

This is a legitimate cited UQ baseline (locally-adaptive ICP / k-NN
conformal, Boström et al. 2017; Romano-Patel-Candès 2019). It varies per
input — unlike a global conformal multiplier on a learned residual head —
so its AUROC differs from the error head ablation.

Reports:
  - per-horizon AUROC of conformal trust score vs true RMSE (q_75 thresh)
  - empirical coverage at α=0.1 (target 90%)
  - step-doubling AUROC for comparison

Outputs JSON to ./results/oregonator_{split}.json
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "models"))

from eval_utils import (load_model, predict, step_doubling_estimator,
                          true_error)

DT_BASE = 0.05
HORIZONS = [2, 4, 8, 16, 32, 64]
CKPT = ROOT / "checkpoints" / "oregonator" / "best.pt"
DATA_DIR = ROOT / "data" / "oregonator"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def state_embedding(state: torch.Tensor) -> np.ndarray:
    """Per-channel mean + std as 4-dim embedding for KNN."""
    if state.dim() == 3:
        state = state.unsqueeze(0)
    mean = state.mean(dim=(2, 3))      # (B, C)
    std = state.std(dim=(2, 3))        # (B, C)
    feat = torch.cat([mean, std], dim=1)
    return feat.cpu().numpy()


@torch.no_grad()
def collect_split_set(model, ds_path: Path, n_per_h: int, seed: int):
    """Sample (state, target, dt) tuples; compute true_err, ê, embedding."""
    rng = np.random.RandomState(seed)
    feats, true_errs, e_hats, horizons_out = [], [], [], []
    with h5py.File(ds_path, "r") as f:
        N, T = f["states"].shape[:2]
        for h in HORIZONS:
            for _ in range(n_per_h):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s = torch.from_numpy(np.array(f["states"][i, t0])
                                       ).unsqueeze(0).to(DEVICE)
                tgt = torch.from_numpy(np.array(f["states"][i, t0 + h])
                                         ).unsqueeze(0).to(DEVICE)
                dt = torch.tensor([h * DT_BASE], dtype=torch.float32, device=DEVICE)
                e_map, pred = step_doubling_estimator(model, s, dt)
                te_map = true_error(pred, tgt)
                feats.append(state_embedding(s)[0])
                true_errs.append(float(te_map.mean().cpu().numpy()))
                e_hats.append(float(e_map.mean().cpu().numpy()))
                horizons_out.append(h)
    return (np.stack(feats), np.array(true_errs),
            np.array(e_hats), np.array(horizons_out))


def knn_conformal_score(feat_test, feat_cal, score_cal, K: int, q: float):
    """For each test point, find K nearest in calibration; return their
    q-quantile of `score_cal`."""
    # L2 distances (feat_test: M×D, feat_cal: N×D)
    d2 = ((feat_test[:, None, :] - feat_cal[None, :, :]) ** 2).sum(axis=-1)
    knn_idx = np.argsort(d2, axis=1)[:, :K]                       # (M, K)
    out = np.empty(len(feat_test))
    for i, idx in enumerate(knn_idx):
        out[i] = float(np.quantile(score_cal[idx], q))
    return out


def auroc_q75(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC at q_75 threshold of labels (true error)."""
    if len(scores) < 4: return float("nan")
    thr = float(np.quantile(labels, 0.75))
    bin_lbl = (labels > thr).astype(int)
    if bin_lbl.sum() == 0 or bin_lbl.sum() == len(bin_lbl):
        return float("nan")
    return float(roc_auc_score(bin_lbl, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_cal_per_h", type=int, default=80)
    ap.add_argument("--n_test_per_h", type=int, default=80)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    print(f"[conformal_oreg] split={args.split} K={args.K} α={args.alpha}", flush=True)
    print(f"[conformal_oreg] device={DEVICE}", flush=True)

    t0 = time.time()
    model = load_model(str(CKPT), device=DEVICE)
    print(f"  model loaded ({time.time()-t0:.1f}s)", flush=True)

    # ── Calibration set: from val split
    cal_path = DATA_DIR / "oregonator_val.h5"
    print("\n=== Calibration set (val split) ===", flush=True)
    t1 = time.time()
    feat_cal, te_cal, eh_cal, h_cal = collect_split_set(
        model, cal_path, args.n_cal_per_h, seed=12345)
    print(f"  N_cal={len(feat_cal)}  ({time.time()-t1:.1f}s)", flush=True)
    print(f"  cal true_err range = [{te_cal.min():.4f}, {te_cal.max():.4f}]", flush=True)

    # ── Test set: from {test, ood_near, ood_far}
    test_path = DATA_DIR / f"oregonator_{args.split}.h5"
    print(f"\n=== Test set ({args.split}) ===", flush=True)
    t2 = time.time()
    feat_te, te_te, eh_te, h_te = collect_split_set(
        model, test_path, args.n_test_per_h, seed=999)
    print(f"  N_test={len(feat_te)}  ({time.time()-t2:.1f}s)", flush=True)

    # ── Conformal trust signal: KNN-quantile of cal true_err per test point
    print(f"\n=== Conformal trust scores (KNN q_{1-args.alpha:.2f}) ===", flush=True)
    t3 = time.time()
    conf_scores = knn_conformal_score(feat_te, feat_cal, te_cal,
                                         K=args.K, q=1 - args.alpha)
    print(f"  computed in {time.time()-t3:.1f}s", flush=True)

    # ── Per-horizon AUROC
    auroc_conformal = {}
    auroc_sd = {}
    coverage = {}
    for h in HORIZONS:
        mask = (h_te == h)
        if mask.sum() < 4:
            auroc_conformal[h] = float("nan")
            auroc_sd[h] = float("nan")
            coverage[h] = float("nan")
            continue
        auroc_conformal[h] = auroc_q75(conf_scores[mask], te_te[mask])
        auroc_sd[h] = auroc_q75(eh_te[mask], te_te[mask])
        coverage[h] = float(np.mean(te_te[mask] <= conf_scores[mask]))
        print(f"  h={h:2d}  AUROC: conformal={auroc_conformal[h]:.4f}  "
              f"SD={auroc_sd[h]:.4f}  coverage={coverage[h]:.3f}", flush=True)

    # Aggregate over horizons
    aurocs_conf = [v for v in auroc_conformal.values() if not np.isnan(v)]
    aurocs_sd_  = [v for v in auroc_sd.values() if not np.isnan(v)]
    covs        = [v for v in coverage.values() if not np.isnan(v)]
    print(f"\n  MEAN  conformal_AUROC={np.mean(aurocs_conf):.4f}  "
          f"SD_AUROC={np.mean(aurocs_sd_):.4f}  "
          f"coverage={np.mean(covs):.3f}", flush=True)

    out = {
        "config": {
            "ckpt": str(CKPT),
            "split": args.split,
            "n_cal_per_h": args.n_cal_per_h,
            "n_test_per_h": args.n_test_per_h,
            "K": args.K,
            "alpha": args.alpha,
            "horizons": HORIZONS,
            "method": "locally_adaptive_split_CP_KNN",
        },
        "auroc_conformal": auroc_conformal,
        "auroc_step_doubling": auroc_sd,
        "coverage_at_1_minus_alpha": coverage,
        "summary": {
            "mean_auroc_conformal": float(np.mean(aurocs_conf)),
            "mean_auroc_step_doubling": float(np.mean(aurocs_sd_)),
            "mean_coverage": float(np.mean(covs)),
            "delta_auroc_sd_minus_conformal": float(
                np.mean(aurocs_sd_) - np.mean(aurocs_conf)),
        }
    }
    out_path = RESULTS / f"oregonator_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
