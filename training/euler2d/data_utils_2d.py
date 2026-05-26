"""Dataset helpers for 2D PDE training (Euler v2).

Reshapes the flattened-per-frame state (N, T, H*W*C) → (N, T, C, H, W) in
a per-sample transform so we don't hold the reshaped 4D tensor in RAM.

The underlying states on disk are stored as float32, shape (N, T, H*W*C)
where C=4 (conserved variables for Euler) and H=W=128 for the default
grid. A 500-traj train set is ~13 GB raw; we keep it on disk via h5py and
only stream the frames we sample per batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data_generation" / "euler2d"))


def compute_channel_stats(ds, n_trajs: int = 50, n_frames_per_traj: int = 20,
                            seed: int = 0):
    """Sample `n_trajs * n_frames_per_traj` frames and return per-channel
    (mean, std) over all spatial positions. O(1) RAM, O(N) reads.

    Returns two numpy arrays of shape (C,).
    """
    rng = np.random.RandomState(seed)
    means = []
    m2s = []       # running sum of squared deviations
    counts = 0
    ch_mean = np.zeros(ds.C, dtype=np.float64)
    ch_m2 = np.zeros(ds.C, dtype=np.float64)
    total = 0
    n_trajs = min(n_trajs, ds.N)
    for i in range(n_trajs):
        ts = rng.randint(0, ds.T, size=min(n_frames_per_traj, ds.T))
        for t in ts:
            frame = ds.frame(i, t).numpy()     # (C, H, W)
            C, H, W = frame.shape
            flat = frame.reshape(C, -1)        # (C, HW)
            k = flat.shape[1]
            for c in range(C):
                delta = flat[c] - ch_mean[c]
                new_mean = ch_mean[c] + delta.sum() / (total + k)
                ch_m2[c] += ((flat[c] - ch_mean[c]) * (flat[c] - new_mean)).sum()
                ch_mean[c] = new_mean
            total += k
    ch_std = np.sqrt(ch_m2 / max(total - 1, 1))
    # guard against zero-variance channel
    ch_std = np.maximum(ch_std, 1e-6)
    return ch_mean.astype(np.float32), ch_std.astype(np.float32)


class Euler2DDataset:
    """Lazy h5 reader. Holds file open for the life of the dataset and
    indexes into the `states` dataset on demand. Frames returned as
    torch.float32 (C, H, W) tensors."""

    def __init__(self, path: str, H: int = 128, W: int = 128, C: int = 4):
        self.path = str(path)
        self.f = h5py.File(self.path, "r")
        self.states = self.f["states"]           # h5py Dataset, shape (N, T, H*W*C)
        self.N, self.T, D = self.states.shape
        assert D == H * W * C, f"state dim {D} != {H}*{W}*{C}"
        self.H = H
        self.W = W
        self.C = C
        self.dt = float(self.f.attrs["dt"])

    def frame(self, i: int, t: int) -> torch.Tensor:
        """Return (C, H, W) tensor for trajectory i at time t."""
        arr = self.states[i, t].reshape(self.H, self.W, self.C)
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1).astype(np.float32)))

    def close(self):
        self.f.close()


class ShortcutSampler2D:
    """Yields (u0, u_target, dt_horizon) triples. u0, u_target shape (B, C, H, W).
    horizon_set is a list of integer strides; the continuous dt is
    horizon * base_dt."""

    def __init__(self, dataset: Euler2DDataset, horizon_set, seed: int,
                 samples_per_epoch: int):
        self.ds = dataset
        self.horizons = list(horizon_set)
        self.rng = np.random.RandomState(seed)
        self.samples_per_epoch = samples_per_epoch
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


class ShuffledShortcutSampler2D(ShortcutSampler2D):
    """Ablation: Δt input is a random horizon unrelated to the true gap."""
    def epoch_iter(self, batch_size: int):
        n_samples = self.samples_per_epoch
        steps = (n_samples + batch_size - 1) // batch_size
        N, T = self.ds.N, self.ds.T
        for _ in range(steps):
            hs_true = self.rng.choice(self.horizons, size=batch_size)
            hs_fake = self.rng.choice(self.horizons, size=batch_size)
            idxs = self.rng.randint(0, N, size=batch_size)
            u0s, uts, dts = [], [], []
            for i, h_true, h_fake in zip(idxs, hs_true, hs_fake):
                t0 = self.rng.randint(0, T - h_true)
                u0s.append(self.ds.frame(i, t0))
                uts.append(self.ds.frame(i, t0 + h_true))
                dts.append(h_fake * self.dt_base)
            yield (
                torch.stack(u0s, dim=0),
                torch.stack(uts, dim=0),
                torch.tensor(dts, dtype=torch.float32),
            )


class SequentialSampler2D:
    """Yields (u_t, u_{t+1}) adjacent-frame pairs."""
    def __init__(self, dataset: Euler2DDataset, seed: int, samples_per_epoch: int):
        self.ds = dataset
        self.rng = np.random.RandomState(seed)
        self.samples_per_epoch = samples_per_epoch

    def epoch_iter(self, batch_size: int):
        n_samples = self.samples_per_epoch
        steps = (n_samples + batch_size - 1) // batch_size
        N, T = self.ds.N, self.ds.T
        for _ in range(steps):
            idxs = self.rng.randint(0, N, size=batch_size)
            t0 = self.rng.randint(0, T - 1, size=batch_size)
            u0s = torch.stack([self.ds.frame(i, t) for i, t in zip(idxs, t0)], dim=0)
            uts = torch.stack([self.ds.frame(i, t + 1) for i, t in zip(idxs, t0)], dim=0)
            yield u0s, uts
