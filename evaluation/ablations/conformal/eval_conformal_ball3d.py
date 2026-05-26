#!/usr/bin/env python3
"""Locally-adaptive split conformal prediction for Ball 3D.
Same KNN-quantile recipe as Oregonator/Euler but on 9-dim state vectors,
with scalar trajectory error rather than spatial fields."""
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
sys.path.insert(0, str(ROOT / "training" / "ball3d"))
from shortcut_ball3d import ShortcutBall3D    # noqa: E402

DT_BASE = 0.01
HORIZONS = [16, 32, 64]
CKPT = ROOT / "checkpoints" / "ball3d" / "best.pt"
DATA_DIR = ROOT / "data" / "ball3d"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    ck = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    m = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                          emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                          ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


@torch.no_grad()
def collect_set(model, split_path, n_per_h, seed):
    rng = np.random.RandomState(seed)
    feats, true_errs, e_hats, hs = [], [], [], []
    with h5py.File(split_path, "r") as f:
        states = np.array(f["states"], dtype=np.float32)
    N, T = states.shape[:2]
    for h in HORIZONS:
        for _ in range(n_per_h):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - h))
            s = torch.from_numpy(states[i, t0]).unsqueeze(0).to(DEVICE)
            tgt = torch.from_numpy(states[i, t0 + h]).unsqueeze(0).to(DEVICE)
            dt = torch.tensor([h * DT_BASE], dtype=torch.float32, device=DEVICE)
            half = dt * 0.5
            pf = model(s, dt)
            pm = model(s, half)
            pc = model(pm, half)
            te = float(torch.sqrt(((pf[:, :3] - tgt[:, :3]) ** 2).sum() + 1e-12).item())
            sd = float(torch.sqrt(((pf - pc) ** 2).sum() + 1e-12).item())
            feats.append(s[0].cpu().numpy())
            true_errs.append(te); e_hats.append(sd); hs.append(h)
    return np.stack(feats), np.array(true_errs), np.array(e_hats), np.array(hs)


def knn_score(feat_test, feat_cal, score_cal, K, q):
    d2 = ((feat_test[:, None, :] - feat_cal[None, :, :]) ** 2).sum(axis=-1)
    knn_idx = np.argsort(d2, axis=1)[:, :K]
    out = np.empty(len(feat_test))
    for i, idx in enumerate(knn_idx):
        out[i] = float(np.quantile(score_cal[idx], q))
    return out


def auroc_q75(scores, labels):
    if len(scores) < 4: return float("nan")
    thr = float(np.quantile(labels, 0.75))
    lbl = (labels > thr).astype(int)
    if lbl.sum() == 0 or lbl.sum() == len(lbl): return float("nan")
    return float(roc_auc_score(lbl, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_cal_per_h", type=int, default=80)
    ap.add_argument("--n_test_per_h", type=int, default=80)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    print(f"[conformal_ball3d] split={args.split}", flush=True)
    model = load_model()
    feat_cal, te_cal, eh_cal, h_cal = collect_set(
        model, DATA_DIR / "ball3d_val.h5", args.n_cal_per_h, seed=12345)
    print(f"  cal: N={len(feat_cal)}", flush=True)
    feat_te, te_te, eh_te, h_te = collect_set(
        model, DATA_DIR / f"ball3d_{args.split}.h5", args.n_test_per_h, seed=999)
    print(f"  test: N={len(feat_te)}", flush=True)

    conf_scores = knn_score(feat_te, feat_cal, te_cal, K=args.K, q=1 - args.alpha)

    auroc_conformal = {}; auroc_sd = {}; coverage = {}
    for h in HORIZONS:
        mask = (h_te == h)
        if mask.sum() < 4:
            auroc_conformal[h] = float("nan"); auroc_sd[h] = float("nan"); coverage[h] = float("nan")
            continue
        auroc_conformal[h] = auroc_q75(conf_scores[mask], te_te[mask])
        auroc_sd[h] = auroc_q75(eh_te[mask], te_te[mask])
        coverage[h] = float(np.mean(te_te[mask] <= conf_scores[mask]))
        print(f"  h={h:2d}  AUROC: conf={auroc_conformal[h]:.4f}  SD={auroc_sd[h]:.4f}  cov={coverage[h]:.3f}", flush=True)

    out = {
        "config": {"ckpt": str(CKPT), "split": args.split,
                     "n_cal_per_h": args.n_cal_per_h, "n_test_per_h": args.n_test_per_h,
                     "K": args.K, "alpha": args.alpha,
                     "horizons": HORIZONS, "method": "locally_adaptive_split_CP_KNN"},
        "auroc_conformal": auroc_conformal,
        "auroc_step_doubling": auroc_sd,
        "coverage_at_1_minus_alpha": coverage,
    }
    out_path = RESULTS / f"ball3d_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
