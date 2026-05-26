"""
Unified data generation pipeline.

Usage:
    python3 generate_data.py --env particles2d \
        --n_train 20 --n_val 5 --n_test 5 --seed 42 --out data/euler2d/mini

Writes three HDF5 files to <out>/<env>_{train,val,test}.h5, each containing
the unified TrajectoryBatch format with per-trajectory events and
discontinuity density.

Determinism: for seed S, train uses seeds [S, S+1, ..., S+n_train-1];
val uses [S+100000, ...]; test uses [S+200000, ...]. Re-running with the
same seed produces byte-identical HDF5 files.

Parallelism: by default serial. Pass --jobs > 1 to parallelize via joblib
(only for envs that are picklable and safe to parallelize; PDE/ODE envs
are, MuJoCo envs are NOT — we force --jobs 1 for mujoco_drop).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_format import TrajectoryBatch, save_hdf5, load_hdf5
from config import load_config
from repro import set_all_seeds, snapshot_env, hash_dataset


MUJOCO_ENVS = {"mujoco_drop"}


def _load_env_module(env_name: str):
    return importlib.import_module(env_name)


def _seeds_for_split(base_seed: int, split: str, n: int) -> list:
    offset = {"train": 0, "val": 100_000, "test": 200_000}[split]
    return [base_seed + offset + i for i in range(n)]


def _rollout_one(env_module, cfg, T: int, seed: int) -> TrajectoryBatch:
    env = env_module.Env(cfg)
    return env.rollout(T=T, seed=seed)


def _generate_split(env_name: str, cfg, split: str, n: int, base_seed: int,
                    jobs: int) -> TrajectoryBatch:
    env_module = _load_env_module(env_name)
    T = int(cfg["env"]["trajectory_length"])
    seeds = _seeds_for_split(base_seed, split, n)

    batches = []
    if jobs <= 1 or env_name in MUJOCO_ENVS:
        # serial
        for s in seeds:
            batches.append(_rollout_one(env_module, cfg, T, s))
    else:
        try:
            from joblib import Parallel, delayed
            batches = Parallel(n_jobs=jobs)(
                delayed(_rollout_one)(env_module, cfg, T, s) for s in seeds
            )
        except ImportError:
            for s in seeds:
                batches.append(_rollout_one(env_module, cfg, T, s))

    # Concatenate into a single TrajectoryBatch
    states = np.concatenate([b.states for b in batches], axis=0)
    if batches[0].actions is not None:
        actions = np.concatenate([b.actions for b in batches], axis=0)
    else:
        actions = None

    event_times = [b.event_times[0] for b in batches]
    event_types = [b.event_types[0] for b in batches]
    event_locs = [b.event_locations[0] for b in batches]

    # Aggregate discontinuity density
    Ds = []
    for b in batches:
        md = b.metadata.get("discontinuity_density", None)
        if md is None:
            Ds.append(0.0)
        else:
            Ds.append(float(md[0]) if isinstance(md, list) else float(md))

    combined = TrajectoryBatch(
        states=states,
        actions=actions,
        dt=batches[0].dt,
        event_times=event_times,
        event_types=event_types,
        event_locations=event_locs,
        env_name=env_name,
        env_config=dict(batches[0].env_config),
        seed=base_seed,
        solver_name=batches[0].solver_name,
        metadata={
            "split": split,
            "n_trajectories": n,
            "trajectory_length": T,
            "seed_base": base_seed,
            "seeds_used": seeds,
            "discontinuity_density_per_traj": Ds,
            "discontinuity_density_mean": float(np.mean(Ds)) if Ds else 0.0,
        },
    )
    return combined


def generate(env: str, n_train: int, n_val: int, n_test: int,
             seed: int, out: str, jobs: int = 1) -> dict:
    cfg = load_config(env)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        if n <= 0:
            continue
        # Deterministic: set all seeds at each split start so python/numpy
        # random draws inside env reset() are reproducible even when envs
        # use module-level rngs.
        set_all_seeds(seed + {"train": 0, "val": 100_000, "test": 200_000}[split])
        t0 = time.time()
        batch = _generate_split(env, cfg, split, n, seed, jobs)
        wall = time.time() - t0

        path = out_dir / f"{env}_{split}.h5"
        save_hdf5(batch, path)
        h = hash_dataset(path)
        results[split] = {
            "path": str(path),
            "sha256": h,
            "n": n,
            "wall_s": wall,
            "mean_D": batch.metadata.get("discontinuity_density_mean", None),
        }
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--n_val", type=int, required=True)
    ap.add_argument("--n_test", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)

    res = generate(args.env, args.n_train, args.n_val, args.n_test,
                   args.seed, args.out, args.jobs)
    for split, info in res.items():
        print(f"  {split}: {info['path']}  sha256={info['sha256'][:16]}...  "
              f"n={info['n']}  wall={info['wall_s']:.2f}s  mean_D={info['mean_D']}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
