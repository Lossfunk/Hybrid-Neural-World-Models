"""Dataset & sampler for Oregonator 2D shortcut training.

The Oregonator HDF5 layout (from generate_dataset.py) is channels-first:
  /states     (N, T, C=2, H=256, W=256) float32
  /params     (N, 4) float32 — [f, eps, q, D]
  /ic_types   (N,) S16
  /seeds      (N,) i4
  attrs:      split_name, n_save, dt_save, total_time, L_x, L_y, n_x, n_y

This differs from the Euler 2D layout (N, T, H*W*C flat). We keep a parallel
class hierarchy here rather than reusing data_utils_2d.py.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


def compute_channel_stats(ds, n_trajs: int = 50, n_frames_per_traj: int = 20,
                            seed: int = 0):
    """Welford running mean/M2 over a sample of frames. O(1) RAM, O(N) reads.
    Returns (ch_mean, ch_std) as numpy arrays of shape (C,)."""
    rng = np.random.RandomState(seed)
    ch_mean = np.zeros(ds.C, dtype=np.float64)
    ch_m2 = np.zeros(ds.C, dtype=np.float64)
    total = 0
    n_trajs = min(n_trajs, ds.N)
    for i in range(n_trajs):
        ts = rng.randint(0, ds.T, size=min(n_frames_per_traj, ds.T))
        for t in ts:
            frame = ds.frame(i, t).numpy()        # (C, H, W)
            C, H, W = frame.shape
            flat = frame.reshape(C, -1)            # (C, HW)
            k = flat.shape[1]
            for c in range(C):
                delta = flat[c] - ch_mean[c]
                new_mean = ch_mean[c] + delta.sum() / (total + k)
                ch_m2[c] += ((flat[c] - ch_mean[c]) * (flat[c] - new_mean)).sum()
                ch_mean[c] = new_mean
            total += k
    ch_std = np.sqrt(ch_m2 / max(total - 1, 1))
    ch_std = np.maximum(ch_std, 1e-6)
    return ch_mean.astype(np.float32), ch_std.astype(np.float32)


class Oregonator2DDataset:
    """Lazy h5 reader for Oregonator HDF5 files.

    Critical perf knob: rdcc_nbytes (HDF5 chunk cache size). Our chunks are
    one-traj-per-chunk = 30-50 MB compressed, ~105 MB uncompressed. h5py's
    default cache is 1 MB → every random frame access re-decompresses the
    whole chunk. We bump the cache to 4 GB so up to ~80 trajs stay
    decompressed in RAM simultaneously. Cuts disk-bottlenecked training
    by 5-10× once the cache is warm.
    """

    def __init__(self, path: str, rdcc_nbytes: int = 4 * 1024**3,
                 rdcc_nslots: int = 4001):
        self.path = str(path)
        self.f = h5py.File(self.path, "r",
                            rdcc_nbytes=rdcc_nbytes,
                            rdcc_nslots=rdcc_nslots,
                            rdcc_w0=0.75)
        self.states = self.f["states"]            # (N, T, C, H, W)
        self.N, self.T, self.C, self.H, self.W = self.states.shape
        self.dt = float(self.f.attrs["dt_save"])

    def frame(self, i: int, t: int) -> torch.Tensor:
        """Return (C, H, W) tensor for trajectory i at frame t."""
        arr = self.states[i, t]                   # (C, H, W) float32
        return torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))

    def close(self):
        self.f.close()


class InMemoryOregonator2DDataset:
    """In-memory dataset. Holds (N, T, C, H, W) in RAM as float32. Used for
    production training: avoids the per-step gzip-decompression overhead that
    cripples h5py-backed reads."""

    def __init__(self, states: np.ndarray, dt_save: float):
        # Keep storage dtype as-is (fp16 or fp32); promote to fp32 in frame().
        self.states_array = states
        self.N, self.T, self.C, self.H, self.W = self.states_array.shape
        self.dt = float(dt_save)
        self._is_fp16 = (states.dtype == np.float16)

    def frame(self, i: int, t: int) -> torch.Tensor:
        arr = self.states_array[i, t]
        if self._is_fp16:
            arr = arr.astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr))

    def close(self):
        pass

    @classmethod
    def from_h5(cls, path: str, n_trajs: int, dtype: str = "float32"):
        """Load first n_trajs trajectories from an HDF5 dataset into RAM.

        dtype="float32" — full precision, ~105 MB/traj. 80 trajs ≈ 8.4 GB.
        dtype="float16" — half precision, ~53 MB/traj. 150 trajs ≈ 7.9 GB.
                          Conversion to float32 happens on-the-fly per frame.

        For main-track accuracy we want more training trajs; fp16 is the
        cheapest way to fit ~2× more data into the same RAM budget.
        """
        import h5py
        import time as _t
        t0 = _t.time()
        np_dtype = np.float32 if dtype == "float32" else np.float16
        # Use a much smaller chunk cache during load (just enough to hold 1
        # chunk at a time). Reading sequentially so a large cache wastes RAM
        # during the load and trips OOM on tight-RAM systems.
        with h5py.File(path, "r", rdcc_nbytes=128 * 1024**2) as f:
            n_avail = f["states"].shape[0]
            n = min(n_trajs, n_avail)
            print(f"[InMemoryOregonator2DDataset] loading {n} trajs from {path} "
                  f"as {dtype}...", flush=True)
            # Chunk-by-chunk load → astype conversion → minimal peak RAM
            arr = np.empty((n,) + f["states"].shape[1:], dtype=np_dtype)
            for i in range(n):
                arr[i] = f["states"][i].astype(np_dtype)
                if (i + 1) % 25 == 0:
                    print(f"  loaded {i+1}/{n}", flush=True)
            dt_save = float(f.attrs["dt_save"])
        ram_gb = arr.nbytes / 1e9
        print(f"[InMemoryOregonator2DDataset] loaded {n} trajs in "
              f"{_t.time()-t0:.1f}s ({ram_gb:.1f} GB RAM, {dtype})", flush=True)
        return cls(arr, dt_save)


class MemmapOregonator2DDataset:
    """Memmap-backed dataset. Random access from on-disk fp16/fp32 array via
    numpy memory-mapped file. OS handles paging — only the read frames are
    in RAM. Cost per random frame access: one 4 KB-aligned disk read.

    Created by scripts/convert_to_memmap.py — produces .npy + .json metadata.
    """

    def __init__(self, npy_path: str, dt_save: float | None = None):
        import json as _json
        import numpy as _np
        npy_path = str(npy_path)
        meta_path = Path(npy_path).with_suffix(".json")
        if meta_path.exists() and dt_save is None:
            meta = _json.loads(meta_path.read_text())
            dt_save = float(meta["dt_save"])
        self.path = npy_path
        # mmap_mode='r' → read-only, no writes possible from training
        self.states_array = _np.lib.format.open_memmap(npy_path, mode="r")
        self.N, self.T, self.C, self.H, self.W = self.states_array.shape
        self.dt = float(dt_save) if dt_save is not None else 0.05
        self._is_fp16 = (self.states_array.dtype == _np.float16)
        print(f"[MemmapOregonator2DDataset] {npy_path}  N={self.N} T={self.T} "
              f"CxHxW={self.C}x{self.H}x{self.W}  dtype={self.states_array.dtype}",
              flush=True)

    def frame(self, i: int, t: int) -> torch.Tensor:
        arr = self.states_array[i, t]
        if self._is_fp16:
            arr = arr.astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr))

    def close(self):
        # numpy memmap auto-closes on GC; nothing explicit to do
        pass


class OregonatorShortcutSampler:
    """Yields (u0, u_target, dt_horizon) triples for shortcut training.

    horizon_set is a list of integer strides; the continuous dt is
    horizon * base_dt.
    """

    def __init__(self, dataset, horizon_set, seed: int,
                 samples_per_epoch: int):
        self.ds = dataset
        self.horizons = sorted(int(h) for h in horizon_set)
        self.rng = np.random.RandomState(int(seed))
        self.samples_per_epoch = int(samples_per_epoch)
        self.dt_base = dataset.dt

    def epoch_iter(self, batch_size: int):
        n_samples = self.samples_per_epoch
        steps = (n_samples + batch_size - 1) // batch_size
        N, T = self.ds.N, self.ds.T
        for _ in range(steps):
            hs = self.rng.choice(self.horizons, size=batch_size)
            idxs = self.rng.randint(0, N, size=batch_size)
            u0s, uts, dts = [], [], []
            for i, h in zip(idxs, hs):
                t0 = self.rng.randint(0, T - h)
                u0s.append(self.ds.frame(i, t0))
                uts.append(self.ds.frame(i, t0 + h))
                dts.append(h * self.dt_base)
            yield (
                torch.stack(u0s, dim=0),
                torch.stack(uts, dim=0),
                torch.tensor(dts, dtype=torch.float32),
            )
