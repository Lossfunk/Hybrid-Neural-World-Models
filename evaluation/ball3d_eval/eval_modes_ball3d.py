#!/usr/bin/env python3
"""Mode 1 + Mode 2 (trust-aware fallback) for Ball3D.

Mode 1: single FiLM forward pass at target_dt.
Mode 2: compute step-doubling ê.scalar. If below tau_global → return surrogate.
        Else → re-simulate from t=0 using Ball3DEnv with the same params.

Same methodology as Oregonator/Euler:
  Stage 1: calibrate τ_global per horizon by quantile of ê across pool.
  Stage 2: per (split, horizon) eval Mode 1, Mode 2, solver_full timings + RMSE.

Random t0 per (traj, h) to match validate() and avoid the "t=0 IC" trap.
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

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "envs"))

from shortcut_ball3d import ShortcutBall3D       # noqa: E402
from ball3d_env import Ball3DEnv                  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


@torch.no_grad()
def mode1(model, s: torch.Tensor, dt: float):
    if s.dim() == 1:
        s = s.unsqueeze(0)
    t0 = time.time()
    pred = model(s, torch.full((s.shape[0],), dt, device=s.device, dtype=torch.float32))
    wall = time.time() - t0
    return {"method": "mode1", "final_state": pred[0].cpu().numpy(),
            "wall_time_s": wall, "n_surrogate_calls": 1, "n_solver_calls": 0}


def solver_full_ball3d(s_init_np: np.ndarray, dt: float, restitution: float,
                         gravity: float):
    """Re-simulate forward by dt seconds using Ball3DEnv from given state."""
    env = Ball3DEnv()
    # Inject state directly (skip env.reset random sampling)
    env._state = s_init_np.astype(np.float64).copy()
    env._restitution = float(restitution)
    env._gravity_vec = np.array([0.0, 0.0, float(gravity)])
    t0 = time.time()
    out = env.step(dt)
    wall = time.time() - t0
    return {"method": "solver", "final_state": out,
            "wall_time_s": wall, "n_surrogate_calls": 0,
            "n_solver_calls": 1}


@torch.no_grad()
def mode2(model, s: torch.Tensor, dt: float, tau_global: float,
            restitution: float, gravity: float):
    if s.dim() == 1:
        s = s.unsqueeze(0)
    t0 = time.time()
    e_hat, pred_full = step_doubling(model, s, dt)
    e_hat_scalar = float(e_hat[0].item())
    used_solver = e_hat_scalar >= tau_global
    if not used_solver:
        out = pred_full[0].cpu().numpy()
        n_solver = 0
    else:
        sv = solver_full_ball3d(s[0].cpu().numpy(), dt, restitution, gravity)
        out = sv["final_state"]
        n_solver = sv["n_solver_calls"]
    wall = time.time() - t0
    return {"method": "mode2", "final_state": out,
            "wall_time_s": wall, "n_surrogate_calls": 3,
            "n_solver_calls": n_solver, "tau": tau_global,
            "e_hat": e_hat_scalar, "used_solver": used_solver}


def calibrate_tau(model, states: np.ndarray, traj_meta: list, traj_idxs: list,
                    horizons: list, quantile: float = 0.75):
    out = {}
    rng = np.random.RandomState(12345)
    T = states.shape[1]
    for h in horizons:
        if h >= T: continue
        es = []
        for ti in traj_idxs:
            t0 = int(rng.randint(0, T - h))
            s = torch.from_numpy(states[ti, t0]).to(DEVICE)
            e_hat, _ = step_doubling(model, s, h * DT_BASE)
            es.append(float(e_hat[0].item()))
        out[h] = float(np.quantile(np.array(es), quantile))
    return out


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def eval_split(model, split_path: str, horizons: list, n_trajs: int = 8,
                quantile: float = 0.75, seed: int = 0):
    with h5py.File(split_path, "r") as f:
        states = np.array(f["states"], dtype=np.float32)
        meta = json.loads(f.attrs["metadata_json"])
    traj_meta = meta["per_traj_meta"]
    rng = np.random.RandomState(seed)
    traj_idxs = rng.choice(states.shape[0], size=min(n_trajs, states.shape[0]),
                             replace=False).tolist()
    print(f"  N={states.shape[0]}, trajs={traj_idxs}", flush=True)

    print(f"  calibrating τ at q={quantile}...", flush=True)
    tau_per_h = calibrate_tau(model, states, traj_meta, traj_idxs, horizons,
                                quantile=quantile)
    print(f"  τ_per_h: {tau_per_h}", flush=True)

    rng_t0 = np.random.RandomState(seed + 9999)
    rows = []
    # Warm
    s_warm = torch.from_numpy(states[traj_idxs[0], 0]).to(DEVICE)
    with torch.no_grad():
        _ = model(s_warm.unsqueeze(0), torch.tensor([DT_BASE], device=DEVICE))

    for ti in traj_idxs:
        m = traj_meta[ti]
        rest, grav = float(m["restitution"]), float(m["gravity"])
        for h in horizons:
            if h >= states.shape[1]: continue
            t0 = int(rng_t0.randint(0, states.shape[1] - h))
            s_init = states[ti, t0]
            target = states[ti, t0 + h]
            s_init_t = torch.from_numpy(s_init).to(DEVICE)
            dt = h * DT_BASE

            # Mode 1
            r1 = mode1(model, s_init_t, dt)
            r1["rmse"] = rmse(r1["final_state"], target)
            r1["traj_idx"] = ti; r1["target_steps"] = h; r1["t0"] = t0
            r1.pop("final_state", None)
            rows.append(r1)

            # Mode 2
            r2 = mode2(model, s_init_t, dt, tau_per_h[h], rest, grav)
            r2["rmse"] = rmse(r2["final_state"], target)
            r2["traj_idx"] = ti; r2["target_steps"] = h; r2["t0"] = t0
            r2.pop("final_state", None)
            rows.append(r2)

            # Solver
            sv = solver_full_ball3d(s_init, dt, rest, grav)
            sv["rmse"] = rmse(sv["final_state"], target)
            sv["traj_idx"] = ti; sv["target_steps"] = h; sv["t0"] = t0
            sv.pop("final_state", None)
            rows.append(sv)

            print(f"  traj {ti:3d}  h={h:3d}  t0={t0:3d}  "
                  f"M1 rmse={r1['rmse']:.4f} wall={r1['wall_time_s']*1000:.1f}ms | "
                  f"M2 rmse={r2['rmse']:.4f} wall={r2['wall_time_s']*1000:.1f}ms "
                  f"solver={r2['used_solver']} | "
                  f"SV rmse={sv['rmse']:.4f} wall={sv['wall_time_s']*1000:.1f}ms",
                  flush=True)

    return {"rows": rows, "tau_per_h": tau_per_h, "traj_idxs": traj_idxs,
            "quantile": quantile}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" /
                                            "shortcut_ball3d" / "seed0" / "best.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--horizons", default="2,4,8,16,32,64")
    ap.add_argument("--n_trajs", type=int, default=8)
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--out", default=str(ROOT / "results" / "ball3d_modes.json"))
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    print(f"[modes] device={DEVICE}  horizons={horizons}", flush=True)
    model = load_model(args.ckpt)

    results = {}
    for split in ["test", "ood_near", "ood_far"]:
        path = Path(args.data_dir) / f"ball3d_{split}.h5"
        if not path.exists():
            continue
        print(f"\n=== {split} ===", flush=True)
        results[split] = eval_split(model, str(path), horizons,
                                       n_trajs=args.n_trajs,
                                       quantile=args.quantile)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[modes] wrote {out_path}", flush=True)

    # Summary
    from collections import defaultdict
    print("\n=== SUMMARY ===")
    print(f"{'split':<10} {'h':>3} | {'M1 wall':>8} {'M2 wall':>8} {'SV wall':>8} | "
          f"{'M1 RMSE':>9} {'M2 RMSE':>9} {'SV RMSE':>9} | {'M2 vs SV':>9}")
    print("-" * 100)
    for split in ["test", "ood_near", "ood_far"]:
        if split not in results: continue
        by = defaultdict(list)
        for r in results[split]["rows"]:
            by[(r["target_steps"], r["method"])].append(r)
        for h in horizons:
            m1 = by.get((h, "mode1"), [])
            m2 = by.get((h, "mode2"), [])
            sv = by.get((h, "solver"), [])
            if not (m1 and m2 and sv): continue
            m1w = np.mean([r["wall_time_s"] for r in m1])
            m2w = np.mean([r["wall_time_s"] for r in m2])
            svw = np.mean([r["wall_time_s"] for r in sv])
            m1r = np.mean([r["rmse"] for r in m1])
            m2r = np.mean([r["rmse"] for r in m2])
            svr = np.mean([r["rmse"] for r in sv])
            print(f"{split:<10} {h:>3} | {m1w*1000:>6.1f}ms {m2w*1000:>6.1f}ms {svw*1000:>6.1f}ms | "
                  f"{m1r:>9.4f} {m2r:>9.4f} {svr:>9.4f} | {svw/m2w:>8.2f}×")
        print()


if __name__ == "__main__":
    main()
