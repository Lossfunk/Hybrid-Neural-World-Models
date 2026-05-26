#!/usr/bin/env python3
"""Combined evals for Ball3D shortcut model.

Adapts C1, C2, C3 from PDE world to non-PDE rigid-body:

  C1: errors localize at "events" (bounce moments).
      For Ball3D: an "event mask" is a pair where the ball was in contact
      with any wall during the prediction window (z ≤ 2*r OR within 2*r of
      any side wall, OR a sign-flip in any velocity component between t0
      and t0+h).
      Ratio = mean(e | event) / mean(e | free-flight).

  C2: smooth-region (free-flight) error scales linearly with Δt.
      For Ball3D: error in free-flight pairs grows linearly under exact
      ballistic dynamics. We measure mean error in free-flight pairs at
      each horizon and report e/Δt.

  C3: step-doubling ê predicts true error e.
      For Ball3D: per-pair scalar ê = ‖f(s, T) − f(f(s, T/2), T/2)‖_2
      and per-pair e = ‖f(s, T) − GT‖_2. AUROC at q75 across many pairs.

Mode 1 + Mode 2 timing also reported per horizon.
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "envs"))

from shortcut_ball3d import ShortcutBall3D       # noqa: E402
from ball3d_env import Ball3DEnv                  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
DT_BASE = 0.01


def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = ShortcutBall3D(
        state_dim=9, hidden_dim=cfg["hidden_dim"], emb_dim=cfg["emb_dim"],
        n_blocks=cfg["n_blocks"],
        ch_mean=ckpt["ch_mean"], ch_std=ckpt["ch_std"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def step_doubling(model, s: torch.Tensor, dt: float):
    """Returns (e_hat_scalar_per_sample, pred_full).
    s: (B, 9), dt: scalar. e_hat is L2 norm per-sample."""
    if s.dim() == 1:
        s = s.unsqueeze(0)
    B = s.shape[0]
    dt_t = torch.full((B,), float(dt), device=s.device, dtype=torch.float32)
    half_t = torch.full((B,), float(dt) * 0.5, device=s.device, dtype=torch.float32)
    pred_full = model(s, dt_t)
    pred_mid = model(s, half_t)
    pred_chain = model(pred_mid, half_t)
    e_hat = torch.sqrt(((pred_full - pred_chain) ** 2).sum(dim=1) + 1e-12)
    return e_hat, pred_full


def is_event_pair(s_traj_window: np.ndarray, ball_radius: float = 0.05,
                    L: float = 0.5) -> bool:
    """A pair (s_t0, s_t1) involves an 'event' if any frame between t0 and t1
    has the ball touching a wall (within 2r), or any velocity component
    flips sign between t0 and t1 (signaling a bounce occurred)."""
    pos = s_traj_window[:, :3]
    vel = s_traj_window[:, 3:6]
    # Wall contact at any frame in window
    near_floor = pos[:, 2] <= 2 * ball_radius
    near_x = np.abs(pos[:, 0]) >= L - 2 * ball_radius
    near_y = np.abs(pos[:, 1]) >= L - 2 * ball_radius
    near_z = pos[:, 2] >= 1.0 - 2 * ball_radius
    near_any = near_floor | near_x | near_y | near_z
    if near_any.any():
        return True
    # Sign flip in any velocity component → bounce
    sign_flip = (vel[:-1, :] * vel[1:, :] < -1e-6).any()
    return bool(sign_flip)


def eval_split(model, split_path: str, n_pairs_per_h: int = 200,
                seed: int = 42) -> dict:
    """Run all evals on one split and return metrics dict."""
    with h5py.File(split_path, "r") as f:
        states = np.array(f["states"], dtype=np.float32)
    N, T, D = states.shape
    print(f"  N={N}, T={T}, D={D}", flush=True)

    rng = np.random.RandomState(seed)
    out = {"per_horizon": {}}

    for h in HORIZONS:
        if h >= T:
            continue
        dt_target = h * DT_BASE
        e_true_list = []
        e_hat_list = []
        is_event_list = []
        s0_batch = []
        target_batch = []
        # Sample pairs
        triples = []
        for _ in range(n_pairs_per_h):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - h))
            triples.append((i, t0))
        # Build batches
        s0s = np.stack([states[i, t0] for i, t0 in triples])
        sts = np.stack([states[i, t0 + h] for i, t0 in triples])
        s0_t = torch.from_numpy(s0s).to(DEVICE)
        st_t = torch.from_numpy(sts).to(DEVICE)
        # ê + pred_full
        e_hat, pred_full = step_doubling(model, s0_t, dt_target)
        # e_true
        e_true = torch.sqrt(((pred_full - st_t) ** 2).sum(dim=1) + 1e-12)
        e_hat_np = e_hat.cpu().numpy()
        e_true_np = e_true.cpu().numpy()
        # Event mask: per-pair, look at trajectory window [t0, t0+h]
        events = np.array([
            is_event_pair(states[i, t0:t0 + h + 1])
            for (i, t0) in triples
        ])
        # C1: ratio of event-error to free-flight-error
        if events.sum() >= 5 and (~events).sum() >= 5:
            e_event = float(e_true_np[events].mean())
            e_smooth = float(e_true_np[~events].mean())
            ratio = e_event / max(e_smooth, 1e-12)
        else:
            e_event = e_smooth = ratio = float("nan")
        # C2: free-flight mean error (the "smooth" baseline, and e/Δt)
        c2_e_smooth = e_smooth
        c2_eps = c2_e_smooth / dt_target if not np.isnan(c2_e_smooth) else float("nan")
        # C3: AUROC of ê vs true-error-top-25%
        thr = float(np.quantile(e_true_np, 0.75))
        labels = (e_true_np > thr).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(labels, e_hat_np))
        # Pearson r for legacy comparison
        r = float(np.corrcoef(e_hat_np, e_true_np)[0, 1]) if e_hat_np.std() > 1e-12 else float("nan")
        # Mean e and ê
        e_mean = float(e_true_np.mean())
        ehat_mean = float(e_hat_np.mean())

        out["per_horizon"][h] = {
            "n_pairs": int(n_pairs_per_h),
            "n_events": int(events.sum()),
            "dt": dt_target,
            "e_event": e_event, "e_smooth": e_smooth, "ratio": ratio,
            "c2_e_smooth_per_dt": c2_eps,
            "auroc_q75": auroc,
            "pearson_r": r,
            "e_true_mean": e_mean,
            "e_hat_mean": ehat_mean,
        }
        print(f"  h={h:3d}  Δt={dt_target:.3f}s  "
              f"event={int(events.sum()):3d}/{n_pairs_per_h}  "
              f"e_event={e_event:.4f} e_smooth={e_smooth:.4f}  ratio={ratio:.2f}× | "
              f"AUROC={auroc:.3f}  r={r:.3f}  | e_mean={e_mean:.4f}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" /
                                            "shortcut_ball3d" / "seed0" / "best.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--n_pairs_per_h", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results" / "ball3d_evals.json"))
    args = ap.parse_args()

    print(f"[ball3d_evals] device={DEVICE}", flush=True)
    model = load_model(args.ckpt)
    print(f"[ball3d_evals] params: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    results = {}
    for split in ["test", "ood_near", "ood_far"]:
        path = Path(args.data_dir) / f"ball3d_{split}.h5"
        if not path.exists():
            print(f"  WARN missing: {path}")
            continue
        print(f"\n=== {split} ===", flush=True)
        results[split] = eval_split(model, str(path),
                                       n_pairs_per_h=args.n_pairs_per_h)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[ball3d_evals] wrote {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'split':<10} {'h':>3} | {'event_frac':>10} {'C1 ratio':>9} | "
          f"{'C2 e/Δt':>9} | {'C3 AUROC':>9}")
    print("-" * 80)
    for split in ["test", "ood_near", "ood_far"]:
        if split not in results:
            continue
        for h in HORIZONS:
            d = results[split]["per_horizon"].get(h, None)
            if d is None: continue
            ev_frac = d["n_events"] / d["n_pairs"]
            print(f"{split:<10} {h:>3} | {ev_frac:>10.3f} {d['ratio']:>8.2f}× | "
                  f"{d['c2_e_smooth_per_dt']:>9.4f} | {d['auroc_q75']:>9.3f}")
        print()


if __name__ == "__main__":
    main()
