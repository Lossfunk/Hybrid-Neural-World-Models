#!/usr/bin/env python3
"""Generate OOD-near and OOD-far splits for Euler v2.

The trained model uses random_mixture ICs:
  - Schulz-Rinne configs 3, 4, 6, 12 (deterministic, no parameters)
  - Sedov with E0 in [0.5, 2.0], bg_rho in [0.8, 1.2]

OOD-near: Sedov-only with parameter shifts just outside trained range.
  E0 in [0.3, 0.5] U [2.0, 2.5]
  bg_rho in [0.6, 0.8] U [1.2, 1.5]

OOD-far: Sedov-only with wider parameter shifts, plus Schulz-Rinne configs
  the model never saw in training.
  E0 in [0.1, 0.3] U [2.5, 5.0]
  bg_rho in [0.4, 0.6] U [1.5, 2.0]
  Plus IC pool addition: schulz_rinne_17 (Kurganov-Tadmor config 17)

Output: HDF5 files with the same format as train/val/test, so
data_utils_2d.Euler2DDataset reads them transparently.

Usage:
  python generate_euler2d_v2_ood.py --split ood_near --n 250 --out path/to.h5
  python generate_euler2d_v2_ood.py --split ood_far  --n 250 --out path/to.h5

Smoke test:
  python generate_euler2d_v2_ood.py --split ood_near --n 2 --smoke
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import euler2d_v2 as env_mod         # noqa: E402
from euler2d import _step_hll         # noqa: E402
from config import load_config       # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
#  OOD parameter ranges
# ─────────────────────────────────────────────────────────────────────────
OOD_NEAR_SEDOV_E0_BANDS    = [(0.3, 0.5), (2.0, 2.5)]
OOD_NEAR_SEDOV_BG_RHO_BANDS = [(0.6, 0.8), (1.2, 1.5)]

OOD_FAR_SEDOV_E0_BANDS     = [(0.1, 0.3), (2.5, 5.0)]
OOD_FAR_SEDOV_BG_RHO_BANDS = [(0.4, 0.6), (1.5, 2.0)]


def _pick_band(rng: np.random.RandomState, bands: list) -> float:
    band = bands[rng.randint(len(bands))]
    return float(rng.uniform(band[0], band[1]))


def _sedov_ic_ood(rng: np.random.RandomState, split: str,
                    nx: int, ny: int, domain: tuple, gamma: float):
    if split == "ood_near":
        E0 = _pick_band(rng, OOD_NEAR_SEDOV_E0_BANDS)
        bg = _pick_band(rng, OOD_NEAR_SEDOV_BG_RHO_BANDS)
    elif split == "ood_far":
        E0 = _pick_band(rng, OOD_FAR_SEDOV_E0_BANDS)
        bg = _pick_band(rng, OOD_FAR_SEDOV_BG_RHO_BANDS)
    else:
        raise ValueError(split)
    cx = 0.5 + float(rng.uniform(-0.02, 0.02))
    cy = 0.5 + float(rng.uniform(-0.02, 0.02))
    q, dx, dy = env_mod._sedov_ic(nx, ny, domain, gamma, energy=E0,
                                    bg_rho=bg, center=(cx, cy), radius_cells=2)
    meta = {"ic": "sedov_ood", "E0": E0, "bg_rho": bg, "center": [cx, cy]}
    return (q, dx, dy), None, meta


def _ic_far_pool(rng: np.random.RandomState, nx: int, ny: int,
                  domain: tuple, gamma: float):
    """ood_far adds Sedov-shifted plus a small SR variation pool."""
    pool = ["sedov_ood", "sedov_ood", "sedov_ood",
             "sr_3_perturbed", "sr_4_perturbed"]
    choice = str(rng.choice(pool))
    if choice == "sedov_ood":
        return _sedov_ic_ood(rng, "ood_far", nx, ny, domain, gamma)
    # SR perturbation: take a known config and tilt the wall position by +/- 5%
    cid = 3 if choice == "sr_3_perturbed" else 4
    q, dx, dy = env_mod._schulz_rinne_ic(nx, ny, domain, gamma, config_id=cid)
    meta = {"ic": choice, "config_id": cid}
    return (q, dx, dy), cid, meta


def init_state_ood(rng: np.random.RandomState, split: str,
                     nx: int, ny: int, domain: tuple, gamma: float):
    if split == "ood_near":
        return _sedov_ic_ood(rng, "ood_near", nx, ny, domain, gamma)
    if split == "ood_far":
        return _ic_far_pool(rng, nx, ny, domain, gamma)
    raise ValueError(split)


# ─────────────────────────────────────────────────────────────────────────
#  Rollout (single trajectory)
# ─────────────────────────────────────────────────────────────────────────
def rollout_one(seed: int, split: str, env_cfg: dict) -> dict:
    rng = np.random.RandomState(seed)
    p = env_cfg["solver"]["params"]
    nx, ny = p["grid"][0], p["grid"][1]
    domain = tuple(p["domain"])
    gamma = float(p["gamma"])
    cfl = float(p["cfl"])
    base_dt = float(env_cfg["base_dt"])
    T = int(env_cfg["trajectory_length"])

    (q, dx, dy), cid, ic_extra = init_state_ood(rng, split, nx, ny, domain, gamma)

    # Construct an Env-equivalent rollout via the existing fixed-dt machinery.
    # We mirror Env.rollout in euler2d_v2.py but use the OOD initial condition.
    env_obj = env_mod.Env({
        "env": {
            "solver": {"params": p},
            "base_dt": base_dt,
            "event_detector": env_cfg.get("event_detector", {"params": {"threshold": 0.05}}),
            "trajectory_length": T,
        }
    })
    env_obj._q = q
    env_obj._dx = dx
    env_obj._dy = dy
    env_obj._ic_meta = {"ic_name": split, "ood_meta": ic_extra}

    frames = [env_obj._q.astype(np.float32)]
    real_times = [0.0]
    t_sim = 0.0
    for _ in range(T):
        env_obj._step_fixed_dt(base_dt)
        t_sim += base_dt
        frames.append(env_obj._q.astype(np.float32))
        real_times.append(t_sim)

    arr = np.stack(frames[:T], axis=0)
    states_flat = arr.reshape(T, nx * ny * 4)

    return {
        "states_flat": states_flat,
        "ic_meta": env_obj._ic_meta,
        "real_times": real_times[:T],
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["ood_near", "ood_far"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default=None,
                     help="output HDF5 path; defaults to data/euler2d_v2/euler2d_v2_<split>.h5")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--base_seed", type=int, default=900_000)
    args = ap.parse_args()

    if args.smoke:
        n = 2
    else:
        n = int(args.n)

    cfg = load_config("euler2d_v2")
    T = int(cfg["env"]["trajectory_length"])
    p = cfg["env"]["solver"]["params"]
    nx, ny = p["grid"][0], p["grid"][1]
    D = nx * ny * 4
    base_dt = float(cfg["env"]["base_dt"])

    out = Path(args.out) if args.out else (
        HERE.parent / "data" / "euler2d_v2" / f"euler2d_v2_{args.split}.h5")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.smoke:
        print(f"[ood_gen] WARN: {out} exists. Will overwrite.", flush=True)
        out.unlink()

    print(f"[ood_gen] split={args.split}  n={n}  T={T}  D={D}  out={out}",
          flush=True)
    print(f"[ood_gen] grid={nx}x{ny}  base_dt={base_dt}  gamma={p['gamma']}",
          flush=True)

    seeds = [args.base_seed + i for i in range(n)]
    metas = []
    times = []
    t_start = time.time()

    with h5py.File(out, "w") as f:
        ds = f.create_dataset("states", shape=(n, T, D), dtype=np.float32,
                                chunks=(1, T, D), compression="gzip",
                                compression_opts=4)
        for i, sd in enumerate(seeds):
            r = rollout_one(sd, args.split, cfg["env"])
            ds[i] = r["states_flat"]
            metas.append(r["ic_meta"])
            times.append(r["real_times"])
            if (i + 1) % max(1, n // 10) == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  {i+1}/{n}  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                      flush=True)
        # attrs match the train/val/test format
        f.attrs["dt"] = base_dt
        f.attrs["env_name"] = "euler2d_v2"
        f.attrs["format_version"] = "1.0"
        f.attrs["has_actions"] = False
        f.attrs["env_config_json"] = json.dumps({
            "grid": [nx, ny], "domain": p["domain"], "gamma": p["gamma"],
            "cfl": p["cfl"], "ic_name": args.split,
            "fixed_frame_dt": True,
        })
        f.attrs["metadata_json"] = json.dumps({
            "split": args.split, "n_trajectories": n,
            "trajectory_length": T, "seed_base": args.base_seed,
            "ood_param_ranges": {
                "ood_near": {
                    "sedov_E0_bands": OOD_NEAR_SEDOV_E0_BANDS,
                    "sedov_bg_rho_bands": OOD_NEAR_SEDOV_BG_RHO_BANDS,
                },
                "ood_far": {
                    "sedov_E0_bands": OOD_FAR_SEDOV_E0_BANDS,
                    "sedov_bg_rho_bands": OOD_FAR_SEDOV_BG_RHO_BANDS,
                },
            }[args.split],
            "per_traj_meta": metas,
            "per_traj_times": times,
        }, default=str)
        # events group (empty, matches train format)
        ev = f.create_group("events")
        ev.attrs["present"] = False

    print(f"[ood_gen] wrote {out}  {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
