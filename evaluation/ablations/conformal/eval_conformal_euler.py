#!/usr/bin/env python3
"""Locally-adaptive split conformal prediction baseline — Euler 2D.

Mirror of eval_conformal_oregonator.py for the Euler shortcut surrogate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))
sys.path.insert(0, str(ROOT / "models"))

from eval_utils_euler import (load_model, predict, step_doubling_estimator,
                                 true_error)
from data_utils_2d import Euler2DDataset

HORIZONS = [2, 4, 8, 16, 32, 64]
CKPT = (ROOT / "checkpoints" / "euler2d" / "best.pt")
DATA_DIR = ROOT / "data" / "euler2d"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def state_embedding(state: torch.Tensor) -> np.ndarray:
    if state.dim() == 3:
        state = state.unsqueeze(0)
    mean = state.mean(dim=(2, 3))
    std = state.std(dim=(2, 3))
    feat = torch.cat([mean, std], dim=1)
    return feat.cpu().numpy()


@torch.no_grad()
def collect_split_set(model, ds: Euler2DDataset, n_per_h: int, seed: int):
    rng = np.random.RandomState(seed)
    feats, true_errs, e_hats, horizons_out = [], [], [], []
    for h in HORIZONS:
        if h >= ds.T: continue
        dt_target = h * ds.dt
        for _ in range(n_per_h):
            i = int(rng.randint(0, ds.N))
            t0 = int(rng.randint(0, ds.T - h))
            u0 = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
            ut = ds.frame(i, t0 + h).to(DEVICE).unsqueeze(0)
            e_map, pred = step_doubling_estimator(model, u0, dt_target)
            te_map = true_error(pred, ut)
            feats.append(state_embedding(u0)[0])
            true_errs.append(float(te_map.mean().cpu().numpy()))
            e_hats.append(float(e_map.mean().cpu().numpy()))
            horizons_out.append(h)
    return (np.stack(feats), np.array(true_errs),
            np.array(e_hats), np.array(horizons_out))


def knn_conformal_score(feat_test, feat_cal, score_cal, K: int, q: float):
    d2 = ((feat_test[:, None, :] - feat_cal[None, :, :]) ** 2).sum(axis=-1)
    knn_idx = np.argsort(d2, axis=1)[:, :K]
    out = np.empty(len(feat_test))
    for i, idx in enumerate(knn_idx):
        out[i] = float(np.quantile(score_cal[idx], q))
    return out


def auroc_q75(scores: np.ndarray, labels: np.ndarray) -> float:
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

    print(f"[conformal_euler] split={args.split} K={args.K} α={args.alpha}", flush=True)

    t0 = time.time()
    model = load_model(str(CKPT), device=DEVICE)
    print(f"  model loaded ({time.time()-t0:.1f}s)", flush=True)

    cal_path = DATA_DIR / "euler2d_v2_val.h5"
    cal_ds = Euler2DDataset(str(cal_path))
    print(f"\n=== Calibration set (val, N={cal_ds.N} T={cal_ds.T}) ===", flush=True)
    t1 = time.time()
    feat_cal, te_cal, eh_cal, h_cal = collect_split_set(
        model, cal_ds, args.n_cal_per_h, seed=12345)
    print(f"  N_cal={len(feat_cal)}  ({time.time()-t1:.1f}s)", flush=True)

    test_path = DATA_DIR / f"euler2d_v2_{args.split}.h5"
    test_ds = Euler2DDataset(str(test_path))
    print(f"\n=== Test set ({args.split}, N={test_ds.N} T={test_ds.T}) ===", flush=True)
    t2 = time.time()
    feat_te, te_te, eh_te, h_te = collect_split_set(
        model, test_ds, args.n_test_per_h, seed=999)
    print(f"  N_test={len(feat_te)}  ({time.time()-t2:.1f}s)", flush=True)

    print(f"\n=== Conformal trust scores (KNN q_{1-args.alpha:.2f}) ===", flush=True)
    conf_scores = knn_conformal_score(feat_te, feat_cal, te_cal,
                                         K=args.K, q=1 - args.alpha)

    auroc_conformal, auroc_sd, coverage = {}, {}, {}
    for h in HORIZONS:
        mask = (h_te == h)
        if mask.sum() < 4:
            auroc_conformal[h] = auroc_sd[h] = coverage[h] = float("nan")
            continue
        auroc_conformal[h] = auroc_q75(conf_scores[mask], te_te[mask])
        auroc_sd[h]        = auroc_q75(eh_te[mask], te_te[mask])
        coverage[h]        = float(np.mean(te_te[mask] <= conf_scores[mask]))
        print(f"  h={h:2d}  AUROC: conformal={auroc_conformal[h]:.4f}  "
              f"SD={auroc_sd[h]:.4f}  coverage={coverage[h]:.3f}", flush=True)

    aurocs_c = [v for v in auroc_conformal.values() if not np.isnan(v)]
    aurocs_s = [v for v in auroc_sd.values() if not np.isnan(v)]
    covs     = [v for v in coverage.values() if not np.isnan(v)]
    print(f"\n  MEAN  conformal_AUROC={np.mean(aurocs_c):.4f}  "
          f"SD_AUROC={np.mean(aurocs_s):.4f}  "
          f"coverage={np.mean(covs):.3f}", flush=True)

    out = {
        "config": {"ckpt": str(CKPT), "split": args.split,
                    "n_cal_per_h": args.n_cal_per_h,
                    "n_test_per_h": args.n_test_per_h,
                    "K": args.K, "alpha": args.alpha,
                    "horizons": HORIZONS,
                    "method": "locally_adaptive_split_CP_KNN"},
        "auroc_conformal": auroc_conformal,
        "auroc_step_doubling": auroc_sd,
        "coverage_at_1_minus_alpha": coverage,
        "summary": {
            "mean_auroc_conformal": float(np.mean(aurocs_c)),
            "mean_auroc_step_doubling": float(np.mean(aurocs_s)),
            "mean_coverage": float(np.mean(covs)),
            "delta_auroc_sd_minus_conformal": float(
                np.mean(aurocs_s) - np.mean(aurocs_c)),
        }
    }
    out_path = RESULTS / f"euler_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
