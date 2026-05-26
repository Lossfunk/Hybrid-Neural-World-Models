#!/usr/bin/env python3
"""Compare four UQ baselines for Oregonator surrogate trust signals.

Methods:
  1. Step-doubling disagreement (ours)         — 3 forward passes
  2. Random-input-perturbation TTA              — 1 + K perturbed forward passes
  3. Conformal prediction (post-hoc)           — calibrate quantiles on val
  4. Ensemble disagreement (K=3 seeds)         — K forward passes (NEEDS SEED 1+2)

For each method, compute AUROC at q75 on test/ood_near/ood_far across all
trained horizons.

Usage:
  python eval_uq_baselines_oregonator.py
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
ROOT_AB = HERE.parent
ROOT_ALL = ROOT_AB.parent
sys.path.insert(0, str(ROOT_ALL / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT_ALL / "models"))
sys.path.insert(0, str(ROOT_ALL / "training" / "oregonator"))

from eval_utils import load_model, predict, true_error, step_doubling_estimator    # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS = ["test", "ood_near", "ood_far"]
HORIZONS = [2, 4, 8, 16, 32, 64]
N_PAIRS_PER_H = 100
THRESHOLD_Q = 0.75


@torch.no_grad()
def random_tta_disagreement(model, state: torch.Tensor, dt: float,
                              K: int = 4,
                              sigma_rel: float = 0.01) -> torch.Tensor:
    """Random-perturbation TTA UQ. Adds zero-mean Gaussian noise scaled
    by per-channel std (so the perturbation is sigma_rel × std for each
    channel; comparable across envs), runs model K times, returns
    cell-wise std across samples (norm across channels).

    sigma_rel: noise scale relative to per-channel std. Default 0.01 = 1%.
    """
    if state.dim() == 3:
        state = state.unsqueeze(0)
    # Per-channel std from model's stored normalization stats
    ch_std = model.ch_std.view(1, -1, 1, 1).to(state.device)
    preds = []
    for k in range(K):
        eps = torch.randn_like(state) * (ch_std * sigma_rel)
        pred = predict(model, state + eps, dt)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)            # (K, B, C, H, W)
    std_per_cell = stack.std(dim=0)              # (B, C, H, W)
    return std_per_cell.norm(dim=1)              # (B, H, W)


# NOTE: the previous "post-hoc smooth-reference" baseline was MISLABELED as
# conformal prediction. It was actually a high-frequency-content proxy that
# does not reflect uncertainty. We removed it. A real conformal-prediction
# baseline requires a calibration set + quantile estimation, which is a
# separate workflow. We add it as a follow-up only if needed for the paper
# (the existing UQ-for-surrogate literature has these baselines well
# documented; we cite + reuse rather than re-implement at submission time).


def eval_one_split(model, ds_path: Path, split: str) -> dict:
    out = {}
    rng = np.random.RandomState(0)
    print(f"[{split}]", flush=True)
    with h5py.File(ds_path, "r") as f:
        N, T = f["states"].shape[:2]
        for h in HORIZONS:
            scores = {"step_doubling": [], "random_tta": []}
            labels_all = []
            t0 = time.time()
            for _ in range(N_PAIRS_PER_H):
                i = int(rng.randint(0, N))
                t0_idx = int(rng.randint(0, T - h))
                u0 = torch.from_numpy(np.array(f["states"][i, t0_idx])).to(DEVICE).unsqueeze(0)
                ut = torch.from_numpy(np.array(f["states"][i, t0_idx + h])).to(DEVICE).unsqueeze(0)
                dt = h * 0.05
                # Step-doubling
                e_hat_sd, pred_full = step_doubling_estimator(model, u0, dt)
                # Random TTA (per-channel-std-scaled noise)
                e_hat_tta = random_tta_disagreement(model, u0, dt, K=4,
                                                       sigma_rel=0.01)
                # True error
                e_true = true_error(pred_full, ut)
                etrue_np = e_true[0].cpu().numpy().ravel()
                thr = float(np.quantile(etrue_np, THRESHOLD_Q))
                lbl = (etrue_np > thr).astype(int)
                if lbl.sum() == 0 or lbl.sum() == len(lbl):
                    continue
                scores["step_doubling"].append(e_hat_sd[0].cpu().numpy().ravel())
                scores["random_tta"].append(e_hat_tta[0].cpu().numpy().ravel())
                labels_all.append(lbl)
            if not labels_all:
                continue
            lb = np.concatenate(labels_all)
            method_aurocs = {}
            for method, sl in scores.items():
                sc = np.concatenate(sl)
                method_aurocs[method] = float(roc_auc_score(lb, sc))
            out[h] = method_aurocs
            print(f"  h={h:3d} | "
                  f"SD={method_aurocs['step_doubling']:.3f}  "
                  f"TTA={method_aurocs['random_tta']:.3f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        ROOT_ALL / "checkpoints" / "oregonator" / "best.pt"))
    ap.add_argument("--data_dir", default=str(
        ROOT_ALL / "data" / "oregonator"))
    ap.add_argument("--out", default=str(HERE / "results" / "uq_baselines.json"))
    args = ap.parse_args()

    print(f"[uq_baselines] device={DEVICE}", flush=True)
    model = load_model(args.ckpt, device=DEVICE)
    print(f"[uq_baselines] loaded model from {args.ckpt}", flush=True)

    results = {}
    for split in SPLITS:
        ds_path = Path(args.data_dir) / f"oregonator_{split}.h5"
        if not ds_path.exists():
            print(f"  WARN missing: {ds_path}")
            continue
        results[split] = eval_one_split(model, ds_path, split)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[uq_baselines] wrote {out_path}", flush=True)

    # Summary
    print("\n=== SUMMARY (mean AUROC across horizons) ===")
    print(f"{'split':<10} | {'step-doub':>10} {'rand-TTA':>10}")
    print("-" * 45)
    for split in SPLITS:
        if split not in results:
            continue
        sd_mean = np.mean([results[split][h]["step_doubling"] for h in HORIZONS if h in results[split]])
        tta_mean = np.mean([results[split][h]["random_tta"] for h in HORIZONS if h in results[split]])
        print(f"{split:<10} | {sd_mean:>10.3f} {tta_mean:>10.3f}")


if __name__ == "__main__":
    main()
