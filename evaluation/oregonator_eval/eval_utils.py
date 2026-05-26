"""Shared evaluation helpers for the C1-C5 claim verification scripts.

Common operations across C1 (e-map at fronts), C2 (calibration in smooth
regions), C3 (step-doubling estimator), C5 (OOD calibration of C3):
  - Load trained ShortcutOregonator2D from checkpoint
  - Compute true error e(s, Δt) = ‖f(s, Δt) − Φ(s, Δt)‖ given ground truth
  - Compute step-doubling estimator ê(s, Δt) = ‖f(s, Δt) − f(f(s, Δt/2), Δt/2)‖
  - Compute spatial e-map (per-cell error) at a given horizon
  - Detect fronts (peak |∇u|)
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "training"))

from shortcut_oregonator_2d import ShortcutOregonator2D    # noqa: E402


def load_model(ckpt_path: str, device: str = "cuda") -> ShortcutOregonator2D:
    """Load a trained checkpoint into a ShortcutOregonator2D."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m_cfg = cfg["model"]
    state = ckpt["model_state_dict"]
    # ch_mean/ch_std are buffers in state — model construction uses default
    # then state_dict load overwrites them
    model = ShortcutOregonator2D(
        channels=int(m_cfg.get("channels", 2)),
        base_ch=int(m_cfg["base_ch"]),
        emb_dim=int(m_cfg["emb_dim"]),
        ch_mults=tuple(m_cfg["ch_mults"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: ShortcutOregonator2D, state: torch.Tensor,
             dt: float | torch.Tensor) -> torch.Tensor:
    """Convenience: handles dt as scalar or tensor; returns (B, C, H, W)."""
    device = state.device
    if state.dim() == 3:
        state = state.unsqueeze(0)
    B = state.shape[0]
    if isinstance(dt, (float, int)):
        dt = torch.full((B,), float(dt), device=device, dtype=torch.float32)
    elif dt.dim() == 0:
        dt = dt.expand(B)
    return model(state, dt)


@torch.no_grad()
def step_doubling_estimator(model: ShortcutOregonator2D, state: torch.Tensor,
                              dt: float) -> tuple:
    """Compute ê(s, Δt) per cell.

    Returns (e_hat_map, pred_full) where:
      e_hat_map: (B, H, W) per-cell norm of difference (across channels)
      pred_full: (B, C, H, W) f(s, Δt) — needed downstream for true error
    """
    pred_full = predict(model, state, dt)
    pred_mid = predict(model, state, dt * 0.5)
    pred_chain = predict(model, pred_mid, dt * 0.5)
    diff = pred_full - pred_chain                     # (B, C, H, W)
    e_hat_map = torch.sqrt((diff ** 2).sum(dim=1))    # (B, H, W) per-cell
    return e_hat_map, pred_full


@torch.no_grad()
def true_error(pred_full: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-cell ‖f(s, Δt) − Φ(s, Δt)‖ across channels.
    Returns (B, H, W)."""
    diff = pred_full - target
    return torch.sqrt((diff ** 2).sum(dim=1))


def front_mask(u: np.ndarray, threshold_pct: float = 0.9) -> np.ndarray:
    """Boolean mask of cells where |∇u| exceeds the percentile threshold.

    u: (H, W) numpy. Returns boolean (H, W).
    threshold_pct: e.g. 0.9 → cells in top 10% of |∇u| are flagged as 'front'.
    """
    gy, gx = np.gradient(u)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    thr = np.quantile(mag, threshold_pct)
    return mag >= thr


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, robust to constant inputs."""
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = x.std(); sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def sample_pairs(ds_path: str, horizons: list, n_pairs_per_horizon: int = 200,
                 seed: int = 0) -> list:
    """Sample (traj_i, t0, h) tuples from an HDF5 dataset such that
    t0 + h < T. One sublist per horizon."""
    import h5py
    out = {h: [] for h in horizons}
    rng = np.random.RandomState(int(seed))
    with h5py.File(ds_path, "r") as f:
        N, T = f["states"].shape[:2]
    for h in horizons:
        for _ in range(n_pairs_per_horizon):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - h))
            out[h].append((i, t0, h))
    return out


def load_pair(ds_path: str, i: int, t0: int, h: int) -> tuple:
    """Load (state at t0, state at t0+h, dt) from an HDF5 dataset."""
    with h5py.File(ds_path, "r") as f:
        u0 = np.array(f["states"][i, t0])         # (C, H, W)
        ut = np.array(f["states"][i, t0 + h])     # (C, H, W)
        dt = float(f.attrs["dt_save"]) * h
    return u0, ut, dt
