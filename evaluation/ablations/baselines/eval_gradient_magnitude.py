#!/usr/bin/env python3
"""Gradient-magnitude indicator baseline (Oregonator + Euler).

For PDE state s: |∇s| per cell, then thresholded — classical idea that
"sharp gradients = harder regions". Compared to step-doubling AUROC.

For Euler: gradient of pressure (the canonical shock indicator).
For Oregonator: gradient of u (the activator field, which fronts ride on).

Output: ablations/baselines/results/gradient_magnitude_{env}.json
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
GAMMA = 1.4


def grad_mag_2d(field: np.ndarray) -> np.ndarray:
    """|∇field| per cell."""
    gy, gx = np.gradient(field)
    return np.sqrt(gx ** 2 + gy ** 2)


def pressure_from_q(q: np.ndarray) -> np.ndarray:
    """Pressure from conserved Euler state (4, H, W)."""
    rho, rhou, rhov, E = q[0], q[1], q[2], q[3]
    rho_safe = np.maximum(rho, 1e-8)
    kinetic = 0.5 * (rhou * rhou + rhov * rhov) / rho_safe
    return np.maximum((GAMMA - 1.0) * (E - kinetic), 1e-8)


def run_oregonator():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "training" / "oregonator"))
    from eval_utils import load_model, true_error, step_doubling_estimator

    ckpt = ROOT / "checkpoints" / "oregonator" / "best.pt"
    model = load_model(str(ckpt), device=DEVICE)

    out = {}
    for split in SPLITS:
        print(f"\n=== Oregonator {split} ===", flush=True)
        h5_path = ROOT / "data" / "oregonator" / f"oregonator_{split}.h5"
        with h5py.File(h5_path, "r") as f:
            states = f["states"]
            base_dt = float(f.attrs["dt_save"])
            N, T = states.shape[0], states.shape[1]
            split_results = {}
            for h in HORIZONS:
                if h >= T: continue
                dt_target = h * base_dt
                rng = np.random.RandomState(42)
                n_pairs = 100
                cell_lbl = []; cell_grad = []; cell_sd = []
                pair_lbl = []; pair_grad = []; pair_sd = []
                pp_etrue = []
                for _ in range(n_pairs):
                    i = int(rng.randint(0, N))
                    t0 = int(rng.randint(0, T - h))
                    u0 = torch.from_numpy(states[i, t0]).float().to(DEVICE).unsqueeze(0)
                    ut = torch.from_numpy(states[i, t0 + h]).float().to(DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        e_hat_map, pred_full = step_doubling_estimator(model, u0, dt_target)
                        e_true_map = true_error(pred_full, ut)
                    sd_np = e_hat_map[0].cpu().numpy().ravel()
                    et_np = e_true_map[0].cpu().numpy().ravel()
                    # Gradient on u channel of initial state
                    u_field = states[i, t0, 0]                  # (H, W)
                    grad_field = grad_mag_2d(u_field).ravel()
                    thr = float(np.quantile(et_np, 0.75))
                    lbl = (et_np > thr).astype(int)
                    if lbl.sum() == 0 or lbl.sum() == len(lbl):
                        continue
                    cell_lbl.append(lbl); cell_grad.append(grad_field); cell_sd.append(sd_np)
                    pp_etrue.append(float(et_np.mean()))
                    pair_grad.append(float(grad_field.mean()))
                    pair_sd.append(float(sd_np.mean()))
                if not cell_lbl:
                    continue
                cell_lbl = np.concatenate(cell_lbl)
                cell_grad = np.concatenate(cell_grad)
                cell_sd = np.concatenate(cell_sd)
                pp_etrue = np.array(pp_etrue)
                pp_thr = float(np.quantile(pp_etrue, 0.75))
                pp_lbl = (pp_etrue > pp_thr).astype(int)
                if pp_lbl.sum() == 0 or pp_lbl.sum() == len(pp_lbl):
                    pp_grad_auroc = float("nan"); pp_sd_auroc = float("nan")
                else:
                    pp_grad_auroc = float(roc_auc_score(pp_lbl, np.array(pair_grad)))
                    pp_sd_auroc = float(roc_auc_score(pp_lbl, np.array(pair_sd)))
                split_results[h] = {
                    "n_pairs": int(len(pair_grad)),
                    "auroc_cell_step_doubling": float(roc_auc_score(cell_lbl, cell_sd)),
                    "auroc_cell_grad_mag":      float(roc_auc_score(cell_lbl, cell_grad)),
                    "auroc_pair_step_doubling": pp_sd_auroc,
                    "auroc_pair_grad_mag":      pp_grad_auroc,
                }
                r = split_results[h]
                print(f"  h={h:2d}  cell SD={r['auroc_cell_step_doubling']:.4f} "
                        f"|∇u|={r['auroc_cell_grad_mag']:.4f}  |  "
                        f"pair SD={r['auroc_pair_step_doubling']:.4f} "
                        f"|∇u|={r['auroc_pair_grad_mag']:.4f}", flush=True)
            out[split] = split_results

    out_path = HERE / "results" / "gradient_magnitude_oregonator.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


def run_euler():
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils_euler import load_model, true_error, step_doubling_estimator
    from data_utils_2d import Euler2DDataset

    ckpt = ROOT / "checkpoints" / "euler2d" / "best.pt"
    model = load_model(str(ckpt), device=DEVICE)

    out = {}
    for split in SPLITS:
        print(f"\n=== Euler {split} ===", flush=True)
        ds = Euler2DDataset(str(ROOT / "data" / "euler2d" / f"euler2d_v2_{split}.h5"))
        N, T, base_dt = ds.N, ds.T, ds.dt
        split_results = {}
        for h in HORIZONS:
            if h >= T: continue
            dt_target = h * base_dt
            rng = np.random.RandomState(42)
            n_pairs = 100
            cell_lbl = []; cell_grad = []; cell_sd = []
            pp_etrue = []; pair_grad = []; pair_sd = []
            for _ in range(n_pairs):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                u0 = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
                ut = ds.frame(i, t0 + h).to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    e_hat_map, pred_full = step_doubling_estimator(model, u0, dt_target)
                    e_true_map = true_error(pred_full, ut)
                sd_np = e_hat_map[0].cpu().numpy().ravel()
                et_np = e_true_map[0].cpu().numpy().ravel()
                # Pressure gradient (canonical Euler shock indicator)
                q_init = u0[0].cpu().numpy()
                p_field = pressure_from_q(q_init)
                grad_field = grad_mag_2d(p_field).ravel()
                thr = float(np.quantile(et_np, 0.75))
                lbl = (et_np > thr).astype(int)
                if lbl.sum() == 0 or lbl.sum() == len(lbl): continue
                cell_lbl.append(lbl); cell_grad.append(grad_field); cell_sd.append(sd_np)
                pp_etrue.append(float(et_np.mean()))
                pair_grad.append(float(grad_field.mean()))
                pair_sd.append(float(sd_np.mean()))
            if not cell_lbl: continue
            cell_lbl = np.concatenate(cell_lbl)
            cell_grad = np.concatenate(cell_grad)
            cell_sd = np.concatenate(cell_sd)
            pp_etrue = np.array(pp_etrue)
            pp_thr = float(np.quantile(pp_etrue, 0.75))
            pp_lbl = (pp_etrue > pp_thr).astype(int)
            if pp_lbl.sum() == 0 or pp_lbl.sum() == len(pp_lbl):
                pp_grad_auroc = float("nan"); pp_sd_auroc = float("nan")
            else:
                pp_grad_auroc = float(roc_auc_score(pp_lbl, np.array(pair_grad)))
                pp_sd_auroc = float(roc_auc_score(pp_lbl, np.array(pair_sd)))
            split_results[h] = {
                "n_pairs": int(len(pair_grad)),
                "auroc_cell_step_doubling": float(roc_auc_score(cell_lbl, cell_sd)),
                "auroc_cell_grad_mag":      float(roc_auc_score(cell_lbl, cell_grad)),
                "auroc_pair_step_doubling": pp_sd_auroc,
                "auroc_pair_grad_mag":      pp_grad_auroc,
            }
            r = split_results[h]
            print(f"  h={h:2d}  cell SD={r['auroc_cell_step_doubling']:.4f} "
                    f"|∇p|={r['auroc_cell_grad_mag']:.4f}  |  "
                    f"pair SD={r['auroc_pair_step_doubling']:.4f} "
                    f"|∇p|={r['auroc_pair_grad_mag']:.4f}", flush=True)
        out[split] = split_results
        ds.close()

    out_path = HERE / "results" / "gradient_magnitude_euler.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


def main():
    print("=== Gradient-magnitude baseline ===", flush=True)
    run_oregonator()
    run_euler()


if __name__ == "__main__":
    main()
