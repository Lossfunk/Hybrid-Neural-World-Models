#!/usr/bin/env python3
"""Generate 3D ball trajectory dataset with train/val/test/ood_near/ood_far splits.

Each trajectory: T=100 frames at dt_save=0.01s (1 second total).
State: 9-dim (pos 3, linvel 3, angvel 3).

Per-trajectory parameters (varied):
  - restitution e ∈ [0.7, 0.95]      (train)
  - gravity g  ∈ [-10.5, -9.0] m/s²  (train)
  - initial v_mag  ∈ [1.0, 3.0]
  - initial w_max  ∈ [-5.0, 5.0]^3
  - initial pos    uniform in box

OOD splits: shift restitution + gravity:
  ood_near: restitution [0.5, 0.7] ∪ [0.95, 0.99]
            gravity     [-12, -10.5] ∪ [-9.0, -7.5]
  ood_far:  restitution [0.3, 0.5]
            gravity     [-15, -12] ∪ [-7.5, -5]

Output: data/ball3d_<split>.h5 with shape (N, T+1, 9).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from ball3d_env import Ball3DEnv     # noqa: E402

T_FRAMES = 100
DT_SAVE = 0.01

PARAMS = {
    "train": {
        "restitution_bands": [(0.7, 0.95)],
        "gravity_bands":     [(-10.5, -9.0)],
        "v_mag_band":        (1.0, 3.0),
        "w_max":             5.0,
    },
    "val": {
        "restitution_bands": [(0.7, 0.95)],
        "gravity_bands":     [(-10.5, -9.0)],
        "v_mag_band":        (1.0, 3.0),
        "w_max":             5.0,
    },
    "test": {
        "restitution_bands": [(0.7, 0.95)],
        "gravity_bands":     [(-10.5, -9.0)],
        "v_mag_band":        (1.0, 3.0),
        "w_max":             5.0,
    },
    "ood_near": {
        "restitution_bands": [(0.5, 0.7), (0.95, 0.99)],
        "gravity_bands":     [(-12.0, -10.5), (-9.0, -7.5)],
        "v_mag_band":        (1.0, 3.0),
        "w_max":             5.0,
    },
    "ood_far": {
        "restitution_bands": [(0.3, 0.5)],
        "gravity_bands":     [(-15.0, -12.0), (-7.5, -5.0)],
        "v_mag_band":        (1.0, 3.0),
        "w_max":             5.0,
    },
}

SEED_OFFSETS = {
    "train":    0,
    "val":      100_000,
    "test":     200_000,
    "ood_near": 900_000,
    "ood_far":  1_000_000,
}


def _pick_band(rng: np.random.RandomState, bands: list) -> float:
    band = bands[rng.randint(len(bands))]
    return float(rng.uniform(band[0], band[1]))


def generate_split(split: str, n: int, out_path: Path) -> dict:
    cfg = PARAMS[split]
    base_seed = SEED_OFFSETS[split]
    env = Ball3DEnv()
    states_arr = np.empty((n, T_FRAMES + 1, 9), dtype=np.float32)
    metas = []
    t_start = time.time()
    for i in range(n):
        seed = base_seed + i
        rng = np.random.RandomState(seed)
        e = _pick_band(rng, cfg["restitution_bands"])
        g = _pick_band(rng, cfg["gravity_bands"])
        # Use a fresh seed for env reset (decoupled from param sampling)
        env_seed = seed * 31 + 7
        env.reset(seed=env_seed, v_min=cfg["v_mag_band"][0], v_max=cfg["v_mag_band"][1],
                    w_max=cfg["w_max"], restitution=e, gravity=g)
        states_arr[i] = env.rollout(T_FRAMES, DT_SAVE)
        metas.append({"seed": seed, "restitution": e, "gravity": g})
        if (i + 1) % max(1, n // 10) == 0:
            print(f"  [{split}] {i+1}/{n}  ({time.time()-t_start:.1f}s)",
                  flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("states", data=states_arr, dtype=np.float32,
                          chunks=(1, T_FRAMES + 1, 9))
        f.attrs["dt_save"] = DT_SAVE
        f.attrs["env_name"] = "ball3d"
        f.attrs["state_layout"] = json.dumps({
            "indices": ["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"],
            "L_box": 0.5, "H_box": 1.0, "ball_radius": 0.05,
        })
        f.attrs["metadata_json"] = json.dumps({
            "split": split, "n_trajectories": n, "trajectory_length": T_FRAMES + 1,
            "seed_base": base_seed, "params_config": cfg,
            "per_traj_meta": metas,
        })
    print(f"  [{split}] wrote {out_path}  ({time.time()-t_start:.1f}s, "
          f"size={out_path.stat().st_size/1e6:.1f} MB)", flush=True)
    return {"split": split, "n": n, "wall": time.time() - t_start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "ball3d"))
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_ood_near", type=int, default=200)
    ap.add_argument("--n_ood_far", type=int, default=200)
    ap.add_argument("--smoke", action="store_true",
                     help="smoke test with tiny N")
    args = ap.parse_args()

    if args.smoke:
        sizes = {"train": 4, "val": 2, "test": 2, "ood_near": 2, "ood_far": 2}
    else:
        sizes = {
            "train":    args.n_train,
            "val":      args.n_val,
            "test":     args.n_test,
            "ood_near": args.n_ood_near,
            "ood_far":  args.n_ood_far,
        }
    out_dir = Path(args.out_dir)
    summaries = []
    for split, n in sizes.items():
        out_path = out_dir / f"ball3d_{split}.h5"
        s = generate_split(split, n, out_path)
        summaries.append(s)
    total_wall = sum(s["wall"] for s in summaries)
    print(f"\n[done] total wall: {total_wall:.1f}s. Splits:")
    for s in summaries:
        print(f"  {s['split']:>10}: n={s['n']}, wall={s['wall']:.1f}s")


if __name__ == "__main__":
    main()
