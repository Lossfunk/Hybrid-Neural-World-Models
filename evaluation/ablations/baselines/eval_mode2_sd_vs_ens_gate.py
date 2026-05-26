#!/usr/bin/env python3
"""Mode 2 RMSE comparison: SD gate vs ENS gate vs no-gate (Mode 1) vs full solver.

For each pair (i, t0, h):
  - Compute step-doubling ê (1 model)
  - Compute ensemble disagreement (3 models)
  - Mode 1 prediction RMSE = ‖f(s,h*dt) − GT‖
  - Solver prediction RMSE  ≈ 0 (gold standard)

Threshold each gate at q75 of its values across pairs to choose top-25%
"hard" trajectories. Mode 2 returns:
  - Solver result for triggered (~25%) pairs
  - Mode 1 result for untriggered (~75%) pairs

Mean Mode 2 RMSE = trigger_rate * solver_RMSE + (1-trigger_rate) * M1_RMSE
                 = (1 - trigger_rate) * mean(M1_RMSE | not triggered)
                   (assuming solver is exact)

The KEY question: when triggered cells are chosen by SD vs ENS, which
selection produces lower untriggered M1 RMSE?  → that's the better gate.

Output: ablations/baselines/results/mode2_sd_vs_ens_{env}_{split}.json
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
ROOT = HERE.parent.parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = [2, 4, 8, 16, 32, 64]
N_PAIRS = 100


def euler_paths(seed):
    return ROOT / "checkpoints" / "euler2d" / "best.pt"


def oreg_paths(seed):
    return ROOT / "checkpoints" / "oregonator" / "best.pt"


def ball_paths(seed):
    return ROOT / "checkpoints" / "ball3d" / "best.pt"


@torch.no_grad()
def predict_pde(model, state, dt):
    if state.dim() == 3: state = state.unsqueeze(0)
    B = state.shape[0]
    if isinstance(dt, (float, int)):
        dt = torch.full((B,), float(dt), device=state.device)
    return model(state, dt)


@torch.no_grad()
def predict_ball(model, state, dt):
    if state.dim() == 1: state = state.unsqueeze(0)
    B = state.shape[0]
    dt_t = torch.full((B,), float(dt), device=state.device)
    return model(state, dt_t)


@torch.no_grad()
def step_doubling_pde(model, state, dt):
    pf = predict_pde(model, state, dt)
    pm = predict_pde(model, state, dt * 0.5)
    pc = predict_pde(model, pm, dt * 0.5)
    e = torch.sqrt(((pf - pc) ** 2).sum(dim=1) + 1e-12)
    return e, pf


@torch.no_grad()
def step_doubling_ball(model, state, dt):
    pf = predict_ball(model, state, dt)
    pm = predict_ball(model, state, dt * 0.5)
    pc = predict_ball(model, pm, dt * 0.5)
    e = torch.sqrt(((pf - pc) ** 2).sum(dim=1) + 1e-12)
    return e, pf


@torch.no_grad()
def ensemble_pde(models, state, dt):
    preds = [predict_pde(m, state, dt) for m in models]
    P = torch.stack(preds, dim=0)
    std = P.std(dim=0, unbiased=False)
    return torch.sqrt((std ** 2).sum(dim=1) + 1e-12), P[0]


@torch.no_grad()
def ensemble_ball(models, state, dt):
    preds = [predict_ball(m, state, dt) for m in models]
    P = torch.stack(preds, dim=0)
    std = P.std(dim=0, unbiased=False)
    return torch.sqrt((std ** 2).sum(dim=1) + 1e-12), P[0]


def run_oregonator(split: str):
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "training" / "oregonator"))
    from eval_utils import load_model

    models = [load_model(str(oreg_paths(s)), device=DEVICE) for s in [0, 1, 2]]
    h5_path = ROOT / "data" / "oregonator" / f"oregonator_{split}.h5"
    with h5py.File(h5_path, "r") as f:
        states = f["states"]
        base_dt = float(f.attrs["dt_save"])
        N, T = states.shape[0], states.shape[1]
        out = {}
        for h in HORIZONS:
            if h >= T: continue
            dt_target = h * base_dt
            rng = np.random.RandomState(42)
            sd_score = []; ens_score = []; m1_rmse = []
            for _ in range(N_PAIRS):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                u0 = torch.from_numpy(states[i, t0]).float().to(DEVICE).unsqueeze(0)
                ut = torch.from_numpy(states[i, t0 + h]).float().to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    sd_map, pred_seed0 = step_doubling_pde(models[0], u0, dt_target)
                    ens_map, _ = ensemble_pde(models, u0, dt_target)
                rmse = float(torch.sqrt(((pred_seed0 - ut) ** 2).mean()).item())
                sd_score.append(float(sd_map.mean().item()))
                ens_score.append(float(ens_map.mean().item()))
                m1_rmse.append(rmse)
            out[h] = mode2_compare(sd_score, ens_score, m1_rmse, n_pairs=N_PAIRS)
            r = out[h]
            print(f"  Oregonator {split} h={h}: M1 RMSE={r['m1_rmse_mean']:.3f}  "
                    f"SD-gate Mode2 RMSE={r['m2_sd_gate_rmse']:.3f}  "
                    f"ENS-gate Mode2 RMSE={r['m2_ens_gate_rmse']:.3f}  "
                    f"Δ(ENS-SD)={r['m2_ens_gate_rmse'] - r['m2_sd_gate_rmse']:+.3f}",
                    flush=True)
    out_path = HERE / "results" / f"mode2_sd_vs_ens_oregonator_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path}", flush=True)


def run_euler(split: str):
    sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
    sys.path.insert(0, str(ROOT / "training" / "euler2d"))
    sys.path.insert(0, str(ROOT / "models"))
    from eval_utils_euler import load_model
    from data_utils_2d import Euler2DDataset

    models = [load_model(str(euler_paths(s)), device=DEVICE) for s in [0, 1, 2]]
    ds = Euler2DDataset(str(ROOT / "data" / "euler2d" / f"euler2d_v2_{split}.h5"))
    out = {}
    for h in HORIZONS:
        if h >= ds.T: continue
        dt_target = h * ds.dt
        rng = np.random.RandomState(42)
        sd_score = []; ens_score = []; m1_rmse = []
        for _ in range(N_PAIRS):
            i = int(rng.randint(0, ds.N))
            t0 = int(rng.randint(0, ds.T - h))
            u0 = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
            ut = ds.frame(i, t0 + h).to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                sd_map, pred_seed0 = step_doubling_pde(models[0], u0, dt_target)
                ens_map, _ = ensemble_pde(models, u0, dt_target)
            rmse = float(torch.sqrt(((pred_seed0 - ut) ** 2).mean()).item())
            sd_score.append(float(sd_map.mean().item()))
            ens_score.append(float(ens_map.mean().item()))
            m1_rmse.append(rmse)
        out[h] = mode2_compare(sd_score, ens_score, m1_rmse, n_pairs=N_PAIRS)
        r = out[h]
        print(f"  Euler {split} h={h}: M1 RMSE={r['m1_rmse_mean']:.3f}  "
                f"SD-gate Mode2 RMSE={r['m2_sd_gate_rmse']:.3f}  "
                f"ENS-gate Mode2 RMSE={r['m2_ens_gate_rmse']:.3f}  "
                f"Δ(ENS-SD)={r['m2_ens_gate_rmse'] - r['m2_sd_gate_rmse']:+.3f}",
                flush=True)
    out_path = HERE / "results" / f"mode2_sd_vs_ens_euler_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path}", flush=True)
    ds.close()


def run_ball3d(split: str):
    sys.path.insert(0, str(ROOT / "training" / "ball3d"))
    sys.path.insert(0, str(ROOT / "data_generation" / "ball3d"))
    from shortcut_ball3d import ShortcutBall3D

    def loadm(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ck["config"]
        m = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                              emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                              ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        return m

    models = [loadm(str(ball_paths(s))) for s in [0, 1, 2] if ball_paths(s).exists()]
    if len(models) < 2:
        print(f"  Skipping Ball3D — only {len(models)} seeds available", flush=True)
        return
    h5_path = ROOT / "data" / "ball3d" / f"ball3d_{split}.h5"
    with h5py.File(h5_path, "r") as f:
        all_states = np.array(f["states"])               # (N, T, 9)
    N, T, _ = all_states.shape
    DT_BASE = 0.01

    out = {}
    for h in HORIZONS:
        if h >= T: continue
        dt_target = h * DT_BASE
        rng = np.random.RandomState(42)
        sd_score = []; ens_score = []; m1_rmse = []
        for _ in range(N_PAIRS):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - h))
            s_init = torch.from_numpy(all_states[i, t0]).float().to(DEVICE)
            s_target_np = all_states[i, t0 + h]
            with torch.no_grad():
                sd_b, pred_seed0 = step_doubling_ball(models[0], s_init, dt_target)
                ens_b, _ = ensemble_ball(models, s_init, dt_target)
            pred_np = pred_seed0[0].cpu().numpy()
            rmse = float(np.sqrt(((pred_np - s_target_np) ** 2).mean()))
            sd_score.append(float(sd_b.item()))
            ens_score.append(float(ens_b.item()))
            m1_rmse.append(rmse)
        out[h] = mode2_compare(sd_score, ens_score, m1_rmse, n_pairs=N_PAIRS)
        r = out[h]
        print(f"  Ball3D {split} h={h}: M1 RMSE={r['m1_rmse_mean']:.4f}  "
                f"SD-gate Mode2 RMSE={r['m2_sd_gate_rmse']:.4f}  "
                f"ENS-gate Mode2 RMSE={r['m2_ens_gate_rmse']:.4f}  "
                f"Δ(ENS-SD)={r['m2_ens_gate_rmse'] - r['m2_sd_gate_rmse']:+.4f}",
                flush=True)
    out_path = HERE / "results" / f"mode2_sd_vs_ens_ball3d_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path}", flush=True)


def mode2_compare(sd_score, ens_score, m1_rmse, n_pairs):
    """At q75 trigger threshold, what's the resulting Mode 2 mean RMSE?
    The triggered pairs go to solver (assumed RMSE=0 for the gold standard
    comparison; we're measuring 'untriggered RMSE' to compare gates fairly).

    Lower untriggered RMSE = better gate (catches the high-RMSE pairs)."""
    sd = np.array(sd_score); en = np.array(ens_score); m1 = np.array(m1_rmse)
    sd_thr = float(np.quantile(sd, 0.75))
    en_thr = float(np.quantile(en, 0.75))
    sd_trig = sd > sd_thr           # bool
    en_trig = en > en_thr
    sd_untrig = ~sd_trig
    en_untrig = ~en_trig
    # mean M1 RMSE among UNTRIGGERED pairs (i.e., what survives Mode 2)
    sd_un_mean_m1 = float(m1[sd_untrig].mean()) if sd_untrig.sum() > 0 else float("nan")
    en_un_mean_m1 = float(m1[en_untrig].mean()) if en_untrig.sum() > 0 else float("nan")
    # Mode 2 mean RMSE = (untrig fraction) * (untrig M1 mean) + (trig fraction) * 0
    # Equivalent ranking but expressed at full-pair granularity:
    sd_m2 = sd_un_mean_m1 * (sd_untrig.sum() / len(m1))
    en_m2 = en_un_mean_m1 * (en_untrig.sum() / len(m1))

    return {
        "n_pairs": n_pairs,
        "trigger_rate_sd": float(sd_trig.mean()),
        "trigger_rate_ens": float(en_trig.mean()),
        "m1_rmse_mean": float(m1.mean()),
        "untriggered_m1_rmse_sd_gate":  sd_un_mean_m1,
        "untriggered_m1_rmse_ens_gate": en_un_mean_m1,
        "m2_sd_gate_rmse":  sd_m2,
        "m2_ens_gate_rmse": en_m2,
        "delta_ens_minus_sd": en_m2 - sd_m2,
        "captured_high_rmse_sd":  float(m1[sd_trig].mean()) if sd_trig.sum() > 0 else float("nan"),
        "captured_high_rmse_ens": float(m1[en_trig].mean()) if en_trig.sum() > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="oregonator,euler,ball3d")
    ap.add_argument("--splits", default="test,ood_near,ood_far")
    args = ap.parse_args()
    envs = args.envs.split(",")
    splits = args.splits.split(",")
    for env in envs:
        for split in splits:
            print(f"\n=== {env} {split} ===", flush=True)
            if env == "oregonator":
                run_oregonator(split)
            elif env == "euler":
                run_euler(split)
            elif env == "ball3d":
                run_ball3d(split)


if __name__ == "__main__":
    main()
