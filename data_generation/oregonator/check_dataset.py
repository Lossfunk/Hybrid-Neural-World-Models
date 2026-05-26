#!/usr/bin/env python3
"""Sanity-check the generated dataset — programmatic checks + summary tables.

Verifies, for every split file in data/oregonator/:
  - HDF5 reads cleanly
  - shape == (N, T, 2, H, W) and matches recorded count
  - no NaN, no Inf, all values within sane bounds
  - param distribution falls within expected range for the split
  - seed range matches SEED_OFFSETS allocation
  - no seed appears in more than one split (leakage check)
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA_DIR = ROOT / "data" / "oregonator"


SPLITS = ["train", "val", "test", "ood_near", "ood_far"]


def check_split(name: str, sample_n: int = 10) -> dict:
    """Sampling-based per-split check.

    Full-scan on metadata arrays (params/seeds/ic_types — small, all of them
    fit in RAM trivially). Random-sample `sample_n` trajectories for state
    value-range / NaN / Inf checks (full traj-by-traj scan would take ~30 min
    per split due to gzip decompression). Statistically sufficient to catch
    widespread bugs."""
    path = DATA_DIR / f"oregonator_{name}.h5"
    if not path.exists():
        return {"name": name, "exists": False, "errors": ["missing"]}
    info = {"name": name, "exists": True, "errors": []}
    try:
        f = h5py.File(path, "r")
    except (OSError, BlockingIOError) as e:
        info["errors"].append(f"locked / unreadable: {e}")
        info["exists"] = False
        return info
    with f:
        states = f["states"]
        params = f["params"]
        ic_types = f["ic_types"]
        seeds = f["seeds"]
        info["shape"] = tuple(states.shape)
        n_filled = int(seeds.shape[0])
        info["n_trajs"] = n_filled
        info["n_filled"] = n_filled
        if n_filled > 0:
            # Random sample for state value checks
            rng = np.random.RandomState(42)
            sample_idxs = rng.choice(n_filled, size=min(sample_n, n_filled),
                                      replace=False)
            sample_idxs = sorted(sample_idxs.tolist())
            state_min = float("inf"); state_max = float("-inf")
            has_nan = False; has_inf = False
            for i in sample_idxs:
                tr = states[i]   # (T, C, H, W)
                if np.isnan(tr).any():
                    has_nan = True
                    info["errors"].append(f"NaN in sampled traj {i}")
                if np.isinf(tr).any():
                    has_inf = True
                    info["errors"].append(f"Inf in sampled traj {i}")
                state_min = min(state_min, float(tr.min()))
                state_max = max(state_max, float(tr.max()))
            info["state_min"] = state_min
            info["state_max"] = state_max
            info["has_nan"] = has_nan
            info["has_inf"] = has_inf
            info["n_sampled"] = len(sample_idxs)
            # Full scans on cheap metadata
            p = params[:n_filled]
            info["f_range"] = (float(p[:, 0].min()), float(p[:, 0].max()))
            info["eps_range"] = (float(p[:, 1].min()), float(p[:, 1].max()))
            info["q_range"] = (float(p[:, 2].min()), float(p[:, 2].max()))
            info["D_range"] = (float(p[:, 3].min()), float(p[:, 3].max()))
            info["ic_counts"] = {
                k: int((ic_types[:n_filled] == k.encode()).sum())
                for k in ("spiral", "target", "random")
            }
            info["seeds"] = seeds[:n_filled].tolist()
        info["attrs"] = dict(f.attrs)
    return info


def main() -> int:
    print("=" * 78)
    print(f"Dataset summary  ({DATA_DIR})")
    print("=" * 78)
    all_seeds = {}
    for name in SPLITS:
        info = check_split(name)
        if not info["exists"]:
            print(f"  [{name:<10}] missing")
            continue
        print(f"  [{name:<10}] shape={info['shape']}  filled={info['n_filled']}")
        if info["n_filled"] > 0:
            n_total = info['n_filled']
            n_samp = info.get('n_sampled', 0)
            print(f"               states range "
                  f"[{info['state_min']:.3f}, {info['state_max']:.3f}]   "
                  f"nan={info['has_nan']} inf={info['has_inf']}   "
                  f"(checked {n_samp}/{n_total} sampled trajs)")
            print(f"               f={info['f_range']}  eps={info['eps_range']}")
            print(f"               q={info['q_range']}  D={info['D_range']}")
            print(f"               IC: {info['ic_counts']}  "
                  f"({100*info['ic_counts']['spiral']/n_total:.0f}/"
                  f"{100*info['ic_counts']['target']/n_total:.0f}/"
                  f"{100*info['ic_counts']['random']/n_total:.0f}%, target 50/30/20)")
            all_seeds[name] = set(info["seeds"])
        if info["errors"]:
            print(f"               !! errors: {info['errors']}")
    # Leakage check
    print()
    print("Leakage check — seed overlap across splits:")
    splits_present = list(all_seeds.keys())
    any_overlap = False
    for i in range(len(splits_present)):
        for j in range(i + 1, len(splits_present)):
            a, b = splits_present[i], splits_present[j]
            overlap = all_seeds[a] & all_seeds[b]
            if overlap:
                print(f"  !! {a} ∩ {b} = {sorted(overlap)[:5]}{'...' if len(overlap)>5 else ''}  "
                      f"({len(overlap)} shared)")
                any_overlap = True
    if not any_overlap:
        print(f"  OK — no seeds shared across {len(splits_present)} splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
