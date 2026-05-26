#!/usr/bin/env python3
"""K-scaling: how does ensemble disagreement AUROC scale with K?

We have K=3 trained seeds. We compute AUROC for ALL non-trivial subsets:
  - K=2: {0,1}, {0,2}, {1,2}  → average gives a robust K=2 estimate
  - K=3: {0,1,2} (full)        → already have

This shows the empirical slope from K=2 → K=3. We can then extrapolate
(qualitatively) to K=10.

Output: ablations/ensemble_vs_step_doubling/results/k_scaling_{env}_{split}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent

DEFAULT_HORIZONS = [2, 4, 8, 16, 32, 64]


def euler_paths(seed):
    return ROOT / "checkpoints" / "euler2d" / "best.pt"


def oreg_paths(seed):
    return ROOT / "checkpoints" / "oregonator" / "best.pt"


@torch.no_grad()
def predict(model, state, dt):
    if state.dim() == 3:
        state = state.unsqueeze(0)
    B = state.shape[0]
    if isinstance(dt, (float, int)):
        dt = torch.full((B,), float(dt), device=state.device, dtype=torch.float32)
    return model(state, dt)


@torch.no_grad()
def ensemble_disagreement_subset(models_subset, state, dt):
    preds = [predict(m, state, dt) for m in models_subset]
    P = torch.stack(preds, dim=0)
    std = P.std(dim=0, unbiased=False)
    return torch.sqrt((std ** 2).sum(dim=1) + 1e-12)


def load_data_euler(split):
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from data_utils_2d import Euler2DDataset
    return Euler2DDataset(str(ROOT / "data" / "euler2d" / f"euler2d_v2_{split}.h5"))


def load_data_oreg(split):
    return h5py.File(ROOT / "data" / "oregonator" /
                       f"oregonator_{split}.h5", "r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["euler", "oregonator"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--auroc_thresh", type=float, default=0.75)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = [int(x) for x in args.horizons.split(",")]
    print(f"[k_scaling] env={args.env} split={args.split} device={device}", flush=True)

    if args.env == "euler":
        sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
        from eval_utils_euler import load_model, true_error, step_doubling_estimator
        model_paths = [euler_paths(s) for s in [0, 1, 2]]
        ds = load_data_euler(args.split)
        N, T, base_dt = ds.N, ds.T, ds.dt
        get_frame = lambda i, t: ds.frame(i, t).to(device)
    else:
        sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
        sys.path.insert(0, str(ROOT / "models"))
        sys.path.insert(0, str(ROOT / "training" / "oregonator"))
        from eval_utils import load_model, true_error, step_doubling_estimator
        model_paths = [oreg_paths(s) for s in [0, 1, 2]]
        f = load_data_oreg(args.split)
        states = f["states"]
        N, T = states.shape[0], states.shape[1]
        base_dt = float(f.attrs["dt_save"])
        get_frame = lambda i, t: torch.from_numpy(states[i, t]).float().to(device)

    models = [load_model(str(p), device=device) for p in model_paths]
    print(f"[k_scaling] loaded {len(models)} seeds", flush=True)

    rng = np.random.RandomState(0)
    out = {"per_h": {}}

    for h in horizons:
        if h >= T: continue
        dt_target = h * base_dt

        # Pre-compute everything: pred from each seed, true error vs seed=0.
        # Reuse the same (i, t0) draws across all subset sizes.
        pred_per_seed = {s: [] for s in range(3)}
        e_true_pool = []     # per-cell true error (for seed=0)
        e_hat_pool = []      # step-doubling on seed=0 per-cell
        labels_pool = []
        pp_etrue = []        # per-pair mean true error
        pp_ehat = []         # per-pair mean step-doubling
        pp_preds_per_seed = {s: [] for s in range(3)}     # per-pair raw cells per seed
        kept_indices = []    # indices into pred_per_seed that were KEPT (after the lbl filter)

        t_start = time.time()
        for k_idx in range(args.n_pairs):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - h))
            u0 = get_frame(i, t0).unsqueeze(0)
            ut = get_frame(i, t0 + h).unsqueeze(0)
            with torch.no_grad():
                # step-doubling on seed=0
                e_hat_map, pred_full_seed0 = step_doubling_estimator(models[0], u0, dt_target)
                # all-seed predictions
                preds_this = []
                for s_idx, m in enumerate(models):
                    if s_idx == 0:
                        preds_this.append(pred_full_seed0)
                    else:
                        preds_this.append(predict(m, u0, dt_target))
                e_true_map = true_error(pred_full_seed0, ut)
            ehat_np = e_hat_map[0].cpu().numpy().ravel()
            etrue_np = e_true_map[0].cpu().numpy().ravel()
            thr = float(np.quantile(etrue_np, args.auroc_thresh))
            lbl = (etrue_np > thr).astype(int)
            if lbl.sum() == 0 or lbl.sum() == len(lbl):
                continue
            e_hat_pool.append(ehat_np)
            e_true_pool.append(etrue_np)
            labels_pool.append(lbl)
            pp_etrue.append(float(etrue_np.mean()))
            pp_ehat.append(float(ehat_np.mean()))
            for s_idx in range(3):
                pp_preds_per_seed[s_idx].append(preds_this[s_idx][0].cpu().numpy())
            kept_indices.append(k_idx)

        if not labels_pool:
            print(f"  h={h}: no usable pairs", flush=True)
            continue

        # cell-level AUROC for SD only (constant across K)
        lb = np.concatenate(labels_pool)
        eh = np.concatenate(e_hat_pool)
        auroc_sd_cell = float(roc_auc_score(lb, eh))

        # per-pair labels
        pp_etrue_arr = np.array(pp_etrue)
        pp_thr = float(np.quantile(pp_etrue_arr, args.auroc_thresh))
        pp_lbl = (pp_etrue_arr > pp_thr).astype(int)
        if pp_lbl.sum() == 0 or pp_lbl.sum() == len(pp_lbl):
            auroc_sd_pair = float("nan")
        else:
            auroc_sd_pair = float(roc_auc_score(pp_lbl, np.array(pp_ehat)))

        # Compute ensemble AUROC for each subset size K and each subset
        # composition; average across compositions.
        result_h = {
            "n_pairs": len(labels_pool),
            "auroc_sd_cell": auroc_sd_cell,
            "auroc_sd_pair": auroc_sd_pair,
            "by_K": {},
        }
        for K in [2, 3]:
            by_subset = []
            for combo in combinations(range(3), K):
                e_dis_per_pair_cell = []
                e_dis_per_pair_mean = []
                for p_idx in range(len(labels_pool)):
                    preds_K = np.stack([pp_preds_per_seed[s][p_idx] for s in combo], axis=0)
                    std_per_cell = preds_K.std(axis=0, ddof=0)        # (C, H, W)
                    e_dis_map = np.sqrt((std_per_cell ** 2).sum(axis=0))
                    e_dis_per_pair_cell.append(e_dis_map.ravel())
                    e_dis_per_pair_mean.append(float(e_dis_map.mean()))
                ed = np.concatenate(e_dis_per_pair_cell)
                cell_auroc = float(roc_auc_score(lb, ed))
                if pp_lbl.sum() == 0 or pp_lbl.sum() == len(pp_lbl):
                    pair_auroc = float("nan")
                else:
                    pair_auroc = float(roc_auc_score(pp_lbl, np.array(e_dis_per_pair_mean)))
                by_subset.append({
                    "seeds": list(combo),
                    "cell_auroc": cell_auroc,
                    "pair_auroc": pair_auroc,
                })
            result_h["by_K"][str(K)] = {
                "subsets": by_subset,
                "mean_cell_auroc": float(np.mean([s["cell_auroc"] for s in by_subset])),
                "mean_pair_auroc": float(np.nanmean([s["pair_auroc"] for s in by_subset])),
            }

        out["per_h"][h] = result_h
        print(f"  h={h:3d}  SD pair={auroc_sd_pair:.4f} cell={auroc_sd_cell:.4f}  |  "
                f"K=2 pair={result_h['by_K']['2']['mean_pair_auroc']:.4f} "
                f"cell={result_h['by_K']['2']['mean_cell_auroc']:.4f}  |  "
                f"K=3 pair={result_h['by_K']['3']['mean_pair_auroc']:.4f} "
                f"cell={result_h['by_K']['3']['mean_cell_auroc']:.4f}  "
                f"({time.time()-t_start:.1f}s)", flush=True)

    out_path = HERE / "results" / f"k_scaling_{args.env}_{args.split}.json"
    out_path.write_text(json.dumps({
        "env": args.env, "split": args.split,
        "horizons": horizons, "n_pairs": args.n_pairs,
        "auroc_threshold_quantile": args.auroc_thresh,
        "results": out,
    }, indent=2, default=str))
    print(f"\n[k_scaling] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
