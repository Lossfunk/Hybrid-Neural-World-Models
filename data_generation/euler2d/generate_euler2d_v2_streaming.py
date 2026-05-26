#!/usr/bin/env python3
"""Streaming generator for 2D Euler v2.

The shared generate_data.py concatenates all trajectories in RAM before writing.
At 128² × T=100 × 4 channels × float32 = 26 MB/traj, 500 training trajs would
need 13 GB peak RAM — we only have 14 GB total, so this OOMs.

This streaming version:
  - Creates the HDF5 file up front with a resizable `states` dataset
  - Rolls out one trajectory at a time (serial, deterministic)
  - Appends each traj's states chunk directly to disk; frees the RAM copy
  - Accumulates only per-traj event metadata in RAM (tiny)

Output format is identical to save_hdf5 in trajectory_format.py; downstream
code (load_split etc.) works unchanged.

Determinism: each split is generated from a deterministic seed sequence,
same as generate_data._seeds_for_split. sha256 of the written file is
byte-identical on repeat.
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
sys.path.insert(0, str(HERE))

from config import load_config
from repro import set_all_seeds, hash_dataset
from trajectory_format import _jsonify
import importlib


def _seeds_for_split(base_seed: int, split: str, n: int):
    offset = {"train": 0, "val": 100_000, "test": 200_000}[split]
    return [base_seed + offset + i for i in range(n)]


def _write_events_group(f: h5py.File, event_times, event_types, event_locs):
    ev = f.create_group("events")
    ev.attrs["present"] = True
    N = len(event_times)
    for i in range(N):
        g = ev.create_group(f"traj_{i:05d}")
        times = np.asarray(event_times[i], dtype=np.float64) \
                if event_times[i] is not None else np.zeros(0, dtype=np.float64)
        g.create_dataset("times", data=times)

        if event_types[i] is not None:
            types = [str(t).encode("utf-8") for t in event_types[i]]
            g.create_dataset(
                "types",
                data=np.asarray(types, dtype="S64") if types else np.zeros(0, dtype="S64"),
            )

        if event_locs[i] is not None:
            locs = event_locs[i]
            if len(locs) > 0:
                arr = np.stack([np.asarray(loc, dtype=np.float64).reshape(-1) for loc in locs])
                g.create_dataset("locations", data=arr)
                g.attrs["location_shape"] = np.asarray(locs[0]).shape
            else:
                g.create_dataset("locations", data=np.zeros((0, 0), dtype=np.float64))


def generate_split_streaming(env_name: str, cfg, split: str, n: int,
                              base_seed: int, out_path: Path) -> dict:
    env_module = importlib.import_module(env_name)
    T = int(cfg["env"]["trajectory_length"])
    seeds = _seeds_for_split(base_seed, split, n)

    # Peek one rollout to learn D
    probe_env = env_module.Env(cfg)
    probe = probe_env.rollout(T=T, seed=seeds[0])
    D = probe.states.shape[-1]
    dt_val = float(probe.dt)
    env_config = dict(probe.env_config)
    solver_name = probe.solver_name
    ic_metas = [dict(probe.metadata.get("ic_meta", {}))]
    event_times = [list(probe.event_times[0])]
    event_types = [list(probe.event_types[0])]
    event_locs = [list(probe.event_locations[0])]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        ds = f.create_dataset(
            "states", shape=(n, T, D), dtype=np.float32,
            chunks=(1, T, D), compression="gzip",
        )
        ds[0] = probe.states[0].astype(np.float32)
        del probe

        for idx in range(1, n):
            set_all_seeds(seeds[idx])
            env = env_module.Env(cfg)
            b = env.rollout(T=T, seed=seeds[idx])
            ds[idx] = b.states[0].astype(np.float32)
            ic_metas.append(dict(b.metadata.get("ic_meta", {})))
            event_times.append(list(b.event_times[0]))
            event_types.append(list(b.event_types[0]))
            event_locs.append(list(b.event_locations[0]))
            del b

        # Aggregate discontinuity density (events per timestep)
        Ds = [len(event_times[i]) / float(T) for i in range(n)]

        f.attrs["dt"] = dt_val
        f.attrs["env_name"] = env_name
        f.attrs["seed"] = int(base_seed)
        f.attrs["solver_name"] = solver_name
        f.attrs["env_config_json"] = json.dumps(_jsonify(env_config))
        f.attrs["metadata_json"] = json.dumps(_jsonify({
            "split": split,
            "n_trajectories": n,
            "trajectory_length": T,
            "seed_base": base_seed,
            "seeds_used": seeds,
            "ic_metas": ic_metas,
            "discontinuity_density_per_traj": Ds,
            "discontinuity_density_mean": float(np.mean(Ds)) if Ds else 0.0,
        }))
        f.attrs["has_actions"] = False
        f.attrs["format_version"] = "1.0"
        _write_events_group(f, event_times, event_types, event_locs)

    h = hash_dataset(out_path)
    return {"path": str(out_path), "sha256": h, "n": n,
            "mean_D": float(np.mean(Ds)) if Ds else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="euler2d_v2")
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--n_val", type=int, required=True)
    ap.add_argument("--n_test", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.env)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        if n <= 0:
            continue
        set_all_seeds(args.seed + {"train": 0, "val": 100_000, "test": 200_000}[split])
        t0 = time.time()
        path = out_dir / f"{args.env}_{split}.h5"
        info = generate_split_streaming(args.env, cfg, split, n, args.seed, path)
        info["wall_s"] = time.time() - t0
        results[split] = info
        print(f"  {split}: {info['path']}  sha256={info['sha256'][:16]}...  "
              f"n={info['n']}  wall={info['wall_s']:.1f}s  mean_D={info['mean_D']:.4f}",
              flush=True)

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
