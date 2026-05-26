#!/usr/bin/env python3
"""Generate the Oregonator 2D shortcut-model dataset (Phase O.3).

Produces five HDF5 files under data/oregonator/:
    oregonator_train.h5     (1200 trajs)
    oregonator_val.h5       ( 150 trajs)
    oregonator_test.h5      ( 150 trajs)
    oregonator_ood_near.h5  ( 250 trajs)
    oregonator_ood_far.h5   ( 250 trajs)

Each file's structure (HDF5):
    /states     (N, T, 2, H, W) float32, gzip-compressed (chunks per-traj)
    /params     (N, 4) float32 — columns [f, eps, q, D]
    /ic_types   (N,)  S16 — "spiral" / "target" / "random"
    /seeds      (N,)  i4
    attrs:      split_name, n_save, dt_save, L_x, L_y, n_x, n_y, total_time,
                created_at_iso

Parameter sampling:
    ID:       f ~ U(0.5, 2.0),   eps ~ U(0.02, 0.08)
    OOD-near: f ∈ [0.4, 0.5] ∪ [2.0, 2.2], eps ∈ [0.015, 0.02] ∪ [0.08, 0.10]
              (mixed: 1/3 f-OOD, 1/3 eps-OOD, 1/3 both-OOD)
    OOD-far:  f ∈ [0.3, 0.4] ∪ [2.2, 2.5], eps ∈ [0.010, 0.015] ∪ [0.10, 0.15]
              (same mixing scheme)
    q = 0.002, D = 1.0 fixed across all splits.

IC mix (per the user spec): 50% spiral, 30% target, 20% random+burn-in.
Seeds are partitioned by split to guarantee no overlap across splits.

Usage:
    python generate_dataset.py --smoke                  # 5 trajs/split, ~5 min total
    python generate_dataset.py                          # full 2000 trajs, ~3 hr 4-way
    python generate_dataset.py --workers 4              # explicit worker count
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

# Worker-side import deferred to inside _run_one to keep mp pickling clean.

DATA_DIR = ROOT / "data" / "oregonator"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Grid / time discretization (verified working in O.2)
N_X = 256
N_Y = 256
L_X = 100.0
L_Y = 100.0
DT_SAVE = 0.05
TOTAL_TIME = 10.0
N_SAVE = int(TOTAL_TIME / DT_SAVE) + 1   # 201 frames

# IC mix probabilities
IC_MIX = (("spiral", 0.50), ("target", 0.30), ("random", 0.20))

# Seed offsets per split — disjoint ranges, no overlap possible
SEED_OFFSETS = {
    "train":     0,
    "val":       100_000,
    "test":      200_000,
    "ood_near":  300_000,
    "ood_far":   400_000,
}

# Trajectory counts per split
SPLIT_COUNTS = {
    "train":    1200,
    "val":      150,
    "test":     150,
    "ood_near": 250,
    "ood_far":  250,
}


# ── Parameter sampling ────────────────────────────────────────────────────

def sample_id(rng):
    return dict(f=rng.uniform(0.5, 2.0),
                eps=rng.uniform(0.02, 0.08),
                q=0.002, D=1.0)


def _sample_ood(rng, f_ranges, eps_ranges, id_f=(0.5, 2.0), id_eps=(0.02, 0.08)):
    """Sample with OOD mixing: 1/3 f-OOD (eps-ID), 1/3 eps-OOD (f-ID),
    1/3 both-OOD."""
    mode = int(rng.choice(3))
    if mode == 0:
        f = rng.uniform(*f_ranges[int(rng.choice(len(f_ranges)))])
        eps = rng.uniform(*id_eps)
    elif mode == 1:
        f = rng.uniform(*id_f)
        eps = rng.uniform(*eps_ranges[int(rng.choice(len(eps_ranges)))])
    else:
        f = rng.uniform(*f_ranges[int(rng.choice(len(f_ranges)))])
        eps = rng.uniform(*eps_ranges[int(rng.choice(len(eps_ranges)))])
    return dict(f=f, eps=eps, q=0.002, D=1.0)


def sample_ood_near(rng):
    return _sample_ood(rng,
                       f_ranges=[(0.4, 0.5), (2.0, 2.2)],
                       eps_ranges=[(0.015, 0.02), (0.08, 0.10)])


def sample_ood_far(rng):
    return _sample_ood(rng,
                       f_ranges=[(0.3, 0.4), (2.2, 2.5)],
                       eps_ranges=[(0.010, 0.015), (0.10, 0.15)])


SAMPLERS = {"train": sample_id, "val": sample_id, "test": sample_id,
            "ood_near": sample_ood_near, "ood_far": sample_ood_far}


def sample_ic(rng) -> str:
    r = rng.random()
    cum = 0.0
    for name, p in IC_MIX:
        cum += p
        if r < cum:
            return name
    return IC_MIX[-1][0]


# ── Single-trajectory worker ──────────────────────────────────────────────

def _run_one(args) -> tuple:
    """Run a single trajectory.

    Args
    ----
    args: tuple (seed, params, ic_type)

    Returns
    -------
    (seed, params, ic_type, states)  where states is (T, 2, H, W) float32
    """
    seed, params, ic_type = args
    # Worker-side import — avoids forcing the main process to import torch/etc
    sys.path.insert(0, str(HERE))
    from oregonator2d_tyson import OregonatorTyson2D, TysonParams

    p = TysonParams(eps=params["eps"], q=params["q"], f=params["f"],
                     D=params["D"])
    sim = OregonatorTyson2D(n_x=N_X, n_y=N_Y, L_x=L_X, L_y=L_Y, params=p)

    if ic_type == "spiral":
        # Slight v_refractory variation per seed for diversity
        rng = np.random.RandomState(int(seed))
        v_ref = float(rng.uniform(0.3, 0.7))
        sim.reset_spiral(v_refractory=v_ref)
    elif ic_type == "target":
        rng = np.random.RandomState(int(seed))
        x0 = float(rng.uniform(0.3, 0.7))
        y0 = float(rng.uniform(0.3, 0.7))
        r = float(rng.uniform(0.03, 0.08))
        sim.reset_target(x0_frac=x0, y0_frac=y0, r_frac=r)
    elif ic_type == "random":
        sim.reset_random(seed=int(seed), burn_in_steps=200, burn_in_dt=0.05)
    else:
        raise ValueError(f"unknown ic_type: {ic_type}")

    out = sim.rollout(n_save_frames=N_SAVE, dt_save=DT_SAVE,
                       progress_every=0)
    return (seed, params, ic_type, out["states"])


# ── HDF5 writer ───────────────────────────────────────────────────────────

def _create_h5(path: Path, n_trajs: int, split_name: str):
    """Create an HDF5 file with resizable datasets so that a partial run
    leaves a *valid* file (current size = N trajs actually written)."""
    f = h5py.File(path, "w")
    f.create_dataset(
        "states", shape=(0, N_SAVE, 2, N_Y, N_X),
        maxshape=(None, N_SAVE, 2, N_Y, N_X), dtype="float32",
        chunks=(1, N_SAVE, 2, N_Y, N_X), compression="gzip", compression_opts=4,
    )
    f.create_dataset("params", shape=(0, 4), maxshape=(None, 4), dtype="float32")
    f.create_dataset("ic_types", shape=(0,), maxshape=(None,), dtype="S16")
    f.create_dataset("seeds", shape=(0,), maxshape=(None,), dtype="i4")
    f.attrs["split_name"] = split_name
    f.attrs["target_n_trajs"] = n_trajs
    f.attrs["n_save"] = N_SAVE
    f.attrs["dt_save"] = DT_SAVE
    f.attrs["total_time"] = TOTAL_TIME
    f.attrs["L_x"] = L_X
    f.attrs["L_y"] = L_Y
    f.attrs["n_x"] = N_X
    f.attrs["n_y"] = N_Y
    f.attrs["created_at_iso"] = datetime.datetime.now().isoformat(timespec="seconds")
    f.attrs["model"] = "OregonatorTyson2D"
    return f


def _append_traj(f, write_idx, seed, params, ic_type, states):
    """Resize datasets and write one trajectory atomically."""
    new_size = write_idx + 1
    f["states"].resize((new_size, N_SAVE, 2, N_Y, N_X))
    f["params"].resize((new_size, 4))
    f["ic_types"].resize((new_size,))
    f["seeds"].resize((new_size,))
    f["states"][write_idx] = states
    f["params"][write_idx] = [params["f"], params["eps"], params["q"], params["D"]]
    f["ic_types"][write_idx] = ic_type.encode()
    f["seeds"][write_idx] = seed
    f.flush()


def generate_split(split_name: str, n_trajs: int, n_workers: int,
                    out_path: Path, smoke: bool):
    """Generate one split's worth of trajectories with multiprocessing."""
    sampler = SAMPLERS[split_name]
    seed_offset = SEED_OFFSETS[split_name]

    # Build the (seed, params, ic_type) arglist from a deterministic master rng
    master_rng = np.random.RandomState(seed_offset + 1)   # +1 to differ from per-traj seeds
    arglist = []
    for k in range(n_trajs):
        traj_seed = seed_offset + k
        params = sampler(master_rng)
        ic_type = sample_ic(master_rng)
        arglist.append((traj_seed, params, ic_type))

    print(f"[{split_name}] writing {out_path}  n_trajs={n_trajs}  workers={n_workers}",
          flush=True)
    f = _create_h5(out_path, n_trajs, split_name)
    t0 = time.time()
    write_idx = 0
    log_every = max(1, n_trajs // 20)

    try:
        if n_workers <= 1 or (smoke and n_workers > 1 and len(arglist) <= 8):
            # Serial path: forced when n_workers=1 or smoke is small
            for args in arglist:
                seed, params, ic_type, states = _run_one(args)
                _append_traj(f, write_idx, seed, params, ic_type, states)
                write_idx += 1
                if write_idx % log_every == 0 or write_idx == n_trajs:
                    eta_min = (time.time() - t0) / write_idx * (n_trajs - write_idx) / 60
                    print(f"  [{split_name}] {write_idx:4d}/{n_trajs}  "
                          f"wall={time.time()-t0:.0f}s  ETA={eta_min:.1f}min",
                          flush=True)
        else:
            with mp.Pool(processes=n_workers) as pool:
                for seed, params, ic_type, states in pool.imap_unordered(_run_one, arglist):
                    _append_traj(f, write_idx, seed, params, ic_type, states)
                    write_idx += 1
                    if write_idx % log_every == 0 or write_idx == n_trajs:
                        eta_min = (time.time() - t0) / write_idx * (n_trajs - write_idx) / 60
                        print(f"  [{split_name}] {write_idx:4d}/{n_trajs}  "
                              f"wall={time.time()-t0:.0f}s  ETA={eta_min:.1f}min",
                              flush=True)
    finally:
        f.close()
    elapsed = time.time() - t0
    print(f"[{split_name}] DONE  wall={elapsed:.1f}s ({elapsed/60:.1f} min)  "
          f"avg/traj={elapsed/n_trajs:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                     help="Generate only 5 trajs/split for pipeline validation")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_COUNTS.keys()),
                     help="Subset of splits to generate")
    args = ap.parse_args()

    print(f"[gen] starting at {datetime.datetime.now().isoformat(timespec='seconds')}",
          flush=True)
    print(f"[gen] N_X={N_X} N_Y={N_Y} L={L_X}x{L_Y}  dt_save={DT_SAVE}  "
          f"n_save={N_SAVE}  total_time={TOTAL_TIME}", flush=True)
    print(f"[gen] workers={args.workers}  smoke={args.smoke}", flush=True)

    # On smoke, override counts
    counts = {k: (5 if args.smoke else v) for k, v in SPLIT_COUNTS.items()}
    print(f"[gen] counts: {counts}", flush=True)

    # Save the dataset config
    config = {
        "N_X": N_X, "N_Y": N_Y, "L_X": L_X, "L_Y": L_Y,
        "DT_SAVE": DT_SAVE, "TOTAL_TIME": TOTAL_TIME, "N_SAVE": N_SAVE,
        "SEED_OFFSETS": SEED_OFFSETS,
        "SPLIT_COUNTS": counts,
        "IC_MIX": IC_MIX,
        "smoke": args.smoke,
        "workers": args.workers,
    }
    config_path = DATA_DIR / "dataset_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[gen] config saved: {config_path}", flush=True)

    t_total = time.time()
    for split_name in args.splits:
        out_path = DATA_DIR / f"oregonator_{split_name}.h5"
        if smoke_skip := (out_path.exists() and not args.smoke):
            print(f"[{split_name}] {out_path} already exists — skipping. "
                  f"(rm to regenerate)", flush=True)
            continue
        generate_split(split_name, counts[split_name], args.workers, out_path,
                        args.smoke)

    print(f"[gen] all splits done. total wall = {(time.time()-t_total)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
