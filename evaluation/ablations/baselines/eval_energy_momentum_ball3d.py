#!/usr/bin/env python3
"""Energy and momentum residual as Ball3D trust signal (audit risk #5).

For each predicted state in Ball3D:
  energy(pos, vel) = m * g * pos[2] + 0.5 * m * |vel|^2
  momentum(vel)    = m * vel  (3-vector; magnitude m * |vel|)

Both should be conserved between collisions. After a collision they change
predictably (vel reflects, energy preserved if e=1, lost otherwise). Step-
doubling competition: can the energy/momentum violation magnitude predict
which trajectories have high error?

For each pair (i, t0, h):
  predicted state s_pred = f(s_init, h*dt)
  energy violation = |E(s_pred) - E(s_init)|     (assuming elastic + gravity step)
  momentum violation = |P(s_pred) - P(s_init,t)| (with gravity correction)
  AUROC of these signals predicting per-pair true error rank.

Output: ablations/baselines/results/energy_momentum_ball3d.json
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
sys.path.insert(0, str(ROOT / "training" / "ball3d"))
sys.path.insert(0, str(ROOT / "data_generation" / "ball3d"))

from shortcut_ball3d import ShortcutBall3D                                    # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT_BASE = 0.01
HORIZONS = [2, 4, 8, 16, 32, 64]
SPLITS = ["test", "ood_near", "ood_far"]


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    model = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                              emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                              ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def step_doubling_e_hat(model, s, dt):
    """Returns scalar ê per row in batch (norm across 9 state dims)."""
    if s.dim() == 1: s = s.unsqueeze(0)
    B = s.shape[0]
    dt_t = torch.full((B,), float(dt), device=s.device)
    half_t = dt_t * 0.5
    pf = model(s, dt_t)
    pm = model(s, half_t)
    pc = model(pm, half_t)
    return torch.sqrt(((pf - pc) ** 2).sum(dim=1) + 1e-12), pf


def energy(s_np: np.ndarray, gravity: float) -> np.ndarray:
    """E = m*|g|*z + 0.5*m*|v|^2; m=1, gravity is signed scalar (negative)."""
    pos = s_np[..., 0:3]
    vel = s_np[..., 3:6]
    return abs(gravity) * pos[..., 2] + 0.5 * (vel * vel).sum(axis=-1)


def momentum_magnitude(s_np: np.ndarray) -> np.ndarray:
    """|p| = m*|v|; m=1."""
    vel = s_np[..., 3:6]
    return np.linalg.norm(vel, axis=-1)


def main():
    ckpt = ROOT / "checkpoints" / "ball3d" / "best.pt"
    model = load_model(str(ckpt))
    print(f"Loaded {ckpt}", flush=True)

    out = {}
    for split in SPLITS:
        print(f"\n=== Ball3D {split} ===", flush=True)
        h5_path = ROOT / "data" / "ball3d" / f"ball3d_{split}.h5"
        with h5py.File(h5_path, "r") as f:
            states = np.array(f["states"])               # (N, T, 9)
            meta_list = json.loads(f.attrs["metadata_json"])["per_traj_meta"]
        N, T, _ = states.shape
        print(f"  N={N} T={T}", flush=True)

        per_h_results = {}
        for h in HORIZONS:
            if h >= T: continue
            dt_target = h * DT_BASE
            rng = np.random.RandomState(42)         # consistent per (split, h)
            n_pairs = 200
            pp_e_dE = []     # |ΔE|
            pp_e_dP = []     # |ΔP|
            pp_e_hat = []    # step-doubling
            pp_e_true = []   # true error
            for _ in range(n_pairs):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s_init = states[i, t0]
                s_target = states[i, t0 + h]
                grav = float(meta_list[i]["gravity"])

                s_init_t = torch.from_numpy(s_init).float().to(DEVICE)
                e_hat_b, pf = step_doubling_e_hat(model, s_init_t, dt_target)
                pred = pf[0].cpu().numpy()

                # Energy/momentum at predicted state vs initial state
                # (ignore exact ballistic prediction; just measure violation
                # of "no work done by walls" assumption)
                E_init = energy(s_init, grav)
                E_pred = energy(pred, grav)
                P_init = momentum_magnitude(s_init)
                P_pred = momentum_magnitude(pred)
                pp_e_dE.append(abs(E_pred - E_init))
                pp_e_dP.append(abs(P_pred - P_init))
                pp_e_hat.append(float(e_hat_b[0].cpu().item()))
                # True error = ‖pred - GT‖ across 9 state dims (full state)
                pp_e_true.append(float(np.sqrt(((pred - s_target) ** 2).sum())))

            pp_e_dE = np.array(pp_e_dE)
            pp_e_dP = np.array(pp_e_dP)
            pp_e_hat = np.array(pp_e_hat)
            pp_e_true = np.array(pp_e_true)

            thr = float(np.quantile(pp_e_true, 0.75))
            lbl = (pp_e_true > thr).astype(int)
            if lbl.sum() == 0 or lbl.sum() == len(lbl):
                continue
            per_h_results[h] = {
                "n_pairs": n_pairs,
                "auroc_step_doubling": float(roc_auc_score(lbl, pp_e_hat)),
                "auroc_energy_residual": float(roc_auc_score(lbl, pp_e_dE)),
                "auroc_momentum_residual": float(roc_auc_score(lbl, pp_e_dP)),
                "mean_dE": float(pp_e_dE.mean()),
                "mean_dP": float(pp_e_dP.mean()),
                "mean_e_hat": float(pp_e_hat.mean()),
                "mean_e_true": float(pp_e_true.mean()),
            }
            r = per_h_results[h]
            print(f"  h={h:2d}: SD={r['auroc_step_doubling']:.4f}  "
                    f"|ΔE|={r['auroc_energy_residual']:.4f}  "
                    f"|ΔP|={r['auroc_momentum_residual']:.4f}", flush=True)
        out[split] = per_h_results

    out_path = HERE / "results" / "energy_momentum_ball3d.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
