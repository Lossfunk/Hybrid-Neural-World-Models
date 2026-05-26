#!/usr/bin/env python3
"""Convert HDF5 train data to numpy fp16 memmap for fast random-access training.

Why: the gzip-compressed HDF5 dataset has per-traj chunks (~52 MB compressed,
~105 MB uncompressed). Default h5py chunk cache (1 MB) means every random
frame access decompresses the whole chunk. Even a 4 GB cache thrashes once
the training set exceeds ~38 trajs.

Memmap solution: store as raw fp16 array on disk. OS handles paging — only
the read frames are in RAM. Random access cost = single 4KB-aligned read.

Output:
  data/oregonator_train_memmap.npy   (~21 GB for 400 trajs)
  data/oregonator_train_memmap.json   metadata

Usage:
  python convert_to_memmap.py --n_trajs 400
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
SRC = ROOT / "data" / "oregonator" / "oregonator_train.h5"
OUT_BASE = ROOT / "data" / "oregonator" / "oregonator_train_memmap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trajs", type=int, default=400)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()

    out_npy = OUT_BASE.with_suffix(".npy")
    out_meta = OUT_BASE.with_suffix(".json")
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    with h5py.File(args.src, "r", rdcc_nbytes=128 * 1024**2) as f:
        n_avail, T, C, H, W = f["states"].shape
        n = min(args.n_trajs, n_avail)
        dt_save = float(f.attrs["dt_save"])
        ram_per_traj = T * C * H * W * np_dtype().itemsize
        total_gb = n * ram_per_traj / 1e9
        print(f"Source: {args.src}  shape=({n_avail}, {T}, {C}, {H}, {W})")
        print(f"Converting first {n} trajs to {args.dtype}  → {total_gb:.1f} GB on disk")
        print(f"Output: {out_npy}", flush=True)

        # Pre-allocate the output memmap on disk
        out_arr = np.lib.format.open_memmap(
            out_npy, mode="w+", dtype=np_dtype,
            shape=(n, T, C, H, W),
        )
        t0 = time.time()
        for i in range(n):
            out_arr[i] = f["states"][i].astype(np_dtype)
            if (i + 1) % 25 == 0 or i == n - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  converted {i+1}/{n}  wall={elapsed:.0f}s  ETA={eta:.0f}s",
                      flush=True)
        out_arr.flush()
        del out_arr      # close memmap

    # Save metadata
    meta = dict(n_trajs=n, T=T, C=C, H=H, W=W,
                  dt_save=dt_save, dtype=args.dtype, src=args.src,
                  size_gb=total_gb)
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Memmap: {out_npy} ({total_gb:.1f} GB)")
    print(f"Metadata: {out_meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
