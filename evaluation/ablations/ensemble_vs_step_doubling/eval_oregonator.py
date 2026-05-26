#!/usr/bin/env python3
"""K=3 ensemble disagreement vs step-doubling — Oregonator.

Mirrors eval_euler.py for the Oregonator U-Net (3.5M params, 256² grid, 2 ch).
Outputs: ablations/ensemble_vs_step_doubling/results/oregonator_{split}.json
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
sys.path.insert(0, str(ROOT / "training" / "oregonator"))

from eval_utils import (load_model, predict, step_doubling_estimator,         # noqa: E402
                          true_error)

DEFAULT_HORIZONS = [2, 4, 8, 16, 32, 64]
DEFAULT_SEEDS = [0, 1, 2]
CKPT_BASE = ROOT / "checkpoints" / "oregonator"


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if x.size < 2 or y.size < 2: return float("nan")
    sx = x.std(); sy = y.std()
    if sx < 1e-12 or sy < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def ensemble_disagreement(models, state, dt):
    preds = []
    for m in models:
        preds.append(predict(m, state, dt))
    P = torch.stack(preds, dim=0)             # (K, B, C, H, W)
    mean = P.mean(dim=0)
    std = P.std(dim=0, unbiased=False)
    disagreement = torch.sqrt((std ** 2).sum(dim=1) + 1e-12)
    return disagreement, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_pairs_per_horizon", type=int, default=100)
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "oregonator"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--auroc_thresh", type=float, default=0.75)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"[ens_oreg] split={args.split} device={device} K={len(seeds)}", flush=True)

    models = []
    for s in seeds:
        ck = CKPT_BASE / f"seed{s}" / "best.pt"
        m = load_model(str(ck), device=device)
        models.append(m)
        print(f"  loaded seed={s} from {ck.name}", flush=True)
    ref_model = models[0]

    ds_path = Path(args.data_dir) / f"oregonator_{args.split}.h5"
    f = h5py.File(ds_path, "r")
    states = f["states"]                       # (N, T, 2, 256, 256)
    base_dt = float(f.attrs["dt_save"])
    N, T = states.shape[0], states.shape[1]
    print(f"[ens_oreg] N={N} T={T} base_dt={base_dt}", flush=True)

    rng = np.random.RandomState(0)
    results = {}
    for h in horizons:
        if h >= T:
            print(f"[ens_oreg] skip h={h}", flush=True)
            continue
        dt_target = h * base_dt
        ehat_pool, edis_pool, etrue_pool, labels_pool = [], [], [], []
        # per-pair: scalar mean per pair, used for "trajectory-level"
        # selection (Mode 3X-style coarse signal).
        pp_ehat, pp_edis, pp_etrue = [], [], []
        t_start = time.time()
        for _ in range(args.n_pairs_per_horizon):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - h))
            u0 = torch.from_numpy(states[i, t0]).float().to(device).unsqueeze(0)
            ut = torch.from_numpy(states[i, t0 + h]).float().to(device).unsqueeze(0)
            with torch.no_grad():
                e_dis, _ = ensemble_disagreement(models, u0, dt_target)
                e_hat, pred_full = step_doubling_estimator(ref_model, u0, dt_target)
                e_true = true_error(pred_full, ut)
            edis_np = e_dis[0].cpu().numpy().ravel()
            ehat_np = e_hat[0].cpu().numpy().ravel()
            etrue_np = e_true[0].cpu().numpy().ravel()
            thr = float(np.quantile(etrue_np, args.auroc_thresh))
            lbl = (etrue_np > thr).astype(int)
            if lbl.sum() == 0 or lbl.sum() == len(lbl):
                continue
            ehat_pool.append(ehat_np)
            edis_pool.append(edis_np)
            etrue_pool.append(etrue_np)
            labels_pool.append(lbl)
            pp_ehat.append(float(ehat_np.mean()))
            pp_edis.append(float(edis_np.mean()))
            pp_etrue.append(float(etrue_np.mean()))
        if not labels_pool:
            print(f"  h={h}: no usable pairs", flush=True)
            continue
        lb = np.concatenate(labels_pool)
        eh = np.concatenate(ehat_pool)
        ed = np.concatenate(edis_pool)
        et = np.concatenate(etrue_pool)
        auroc_sd = float(roc_auc_score(lb, eh))
        auroc_en = float(roc_auc_score(lb, ed))
        r_sd = pearson_r(eh, et)
        r_en = pearson_r(ed, et)
        # Per-pair AUROC (coarser, trajectory-level signal)
        pp_eh = np.array(pp_ehat); pp_ed = np.array(pp_edis); pp_et = np.array(pp_etrue)
        thr_pp = float(np.quantile(pp_et, args.auroc_thresh))
        lbl_pp = (pp_et > thr_pp).astype(int)
        if lbl_pp.sum() == 0 or lbl_pp.sum() == len(lbl_pp):
            auroc_sd_pp = float("nan"); auroc_en_pp = float("nan")
        else:
            auroc_sd_pp = float(roc_auc_score(lbl_pp, pp_eh))
            auroc_en_pp = float(roc_auc_score(lbl_pp, pp_ed))

        results[h] = {
            "n_pairs": len(labels_pool),
            "auroc_step_doubling": auroc_sd,
            "auroc_ensemble":      auroc_en,
            "pearson_r_step_doubling": r_sd,
            "pearson_r_ensemble":      r_en,
            "delta_auroc_sd_minus_en": auroc_sd - auroc_en,
            "per_pair_auroc_step_doubling": auroc_sd_pp,
            "per_pair_auroc_ensemble":      auroc_en_pp,
            "per_pair_delta_sd_minus_en":   auroc_sd_pp - auroc_en_pp if not np.isnan(auroc_sd_pp) else float("nan"),
        }
        print(f"  h={h:3d}  dt={dt_target:.4f}  cell-AUROC: "
                f"SD={auroc_sd:.4f}  ENS={auroc_en:.4f}  Δ={auroc_sd - auroc_en:+.4f}  "
                f"|  pair-AUROC: SD={auroc_sd_pp:.4f}  ENS={auroc_en_pp:.4f}  "
                f"({time.time()-t_start:.1f}s)",
                flush=True)
    f.close()

    out_path = Path(args.out) if args.out else (
        HERE / "results" / f"oregonator_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "env": "oregonator",
        "split": args.split,
        "horizons": horizons,
        "n_pairs_per_horizon": args.n_pairs_per_horizon,
        "K": len(seeds),
        "seeds": seeds,
        "auroc_threshold_quantile": args.auroc_thresh,
        "per_h_results": results,
    }, indent=2))
    print(f"\n[ens_oreg] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
