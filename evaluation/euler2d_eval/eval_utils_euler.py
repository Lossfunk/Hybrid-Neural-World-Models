"""Euler-specific evaluation helpers.

Mirrors evaluation/oregonator_eval/eval_utils.py but for the Euler v2
ShortcutPDE2D model. Key differences:
  - Model class: ShortcutPDE2D (4-channel)
  - Dataset format: Euler stores flat (T, nx*ny*4); Euler2DDataset.frame() handles reshape
  - Front detector: |∇p| (pressure gradient) instead of |∇u|

Common helpers identical:
  - load_model(ckpt_path)
  - predict(model, state, dt)
  - step_doubling_estimator(model, state, dt) → ê-map (per-cell)
  - true_error(pred_full, target) → e per-cell
  - front_mask_pressure(state) → boolean mask of top-10% |∇p|
  - sample_pairs / load_pair from Euler HDF5
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
NEURIPS = HERE.parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))

from shortcut_pde_2d import ShortcutPDE2D                  # noqa: E402
from data_utils_2d import Euler2DDataset                   # noqa: E402

GAMMA = 1.4


def load_model(ckpt_path: str, device: str = "cuda") -> ShortcutPDE2D:
    """Load a trained ShortcutPDE2D from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m_cfg = cfg["model"]
    state = ckpt["model_state_dict"]
    model = ShortcutPDE2D(
        channels=int(m_cfg.get("channels", 4)),
        base_ch=int(m_cfg["base_ch"]),
        emb_dim=int(m_cfg["emb_dim"]),
        ch_mults=tuple(m_cfg["ch_mults"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: ShortcutPDE2D, state: torch.Tensor,
             dt) -> torch.Tensor:
    """Forward pass with handling for scalar dt."""
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
def step_doubling_estimator(model: ShortcutPDE2D, state: torch.Tensor,
                              dt: float) -> tuple:
    """Per-cell ê = ‖f(s, dt) − f(f(s, dt/2), dt/2)‖ over 4 channels.

    Returns (e_hat_map, pred_full):
      e_hat_map: (B, H, W)
      pred_full: (B, 4, H, W)
    """
    pred_full = predict(model, state, dt)
    pred_mid = predict(model, state, dt * 0.5)
    pred_chain = predict(model, pred_mid, dt * 0.5)
    diff = pred_full - pred_chain
    e_hat_map = torch.sqrt((diff ** 2).sum(dim=1) + 1e-12)
    return e_hat_map, pred_full


@torch.no_grad()
def true_error(pred_full: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-cell ‖f(s,Δt) − Φ(s,Δt)‖ across 4 channels. Returns (B, H, W)."""
    diff = pred_full - target
    return torch.sqrt((diff ** 2).sum(dim=1) + 1e-12)


def pressure_from_cons(q: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Compute pressure from conserved state q = (rho, rho*u, rho*v, E).
    q can be (C, H, W) or (B, C, H, W). Returns p with shape minus the C-axis."""
    if q.ndim == 4:
        rho = q[:, 0]
        rhou = q[:, 1]
        rhov = q[:, 2]
        E = q[:, 3]
    elif q.ndim == 3:
        rho = q[0]
        rhou = q[1]
        rhov = q[2]
        E = q[3]
    else:
        raise ValueError(f"unexpected q shape {q.shape}")
    rho_safe = np.maximum(rho, 1e-8)
    u = rhou / rho_safe
    v = rhov / rho_safe
    kinetic = 0.5 * rho * (u * u + v * v)
    p = (gamma - 1.0) * (E - kinetic)
    return np.maximum(p, 1e-8)


def front_mask_pressure(state: np.ndarray, threshold_pct: float = 0.9) -> np.ndarray:
    """Front cells = top (1-threshold_pct) by |∇p|. Shock detector.

    state: (4, H, W) numpy. Returns boolean (H, W).

    Uses strict > comparison and a small floor to avoid the degenerate case
    where the quantile is 0 (e.g., piecewise-constant initial conditions
    where most cells have gradient 0).
    """
    p = pressure_from_cons(state)
    gy, gx = np.gradient(p)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    thr = float(np.quantile(mag, threshold_pct))
    # Floor at a small epsilon so a degenerate q-value of 0 doesn't flag everything
    thr = max(thr, 1e-6 * float(mag.max() + 1e-12))
    return mag > thr


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = x.std(); sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def sample_pairs(ds: Euler2DDataset, horizons: list,
                  n_pairs_per_horizon: int = 200, seed: int = 0) -> dict:
    """Sample (i, t0, h) tuples per horizon."""
    out = {h: [] for h in horizons}
    rng = np.random.RandomState(int(seed))
    N, T = ds.N, ds.T
    for h in horizons:
        for _ in range(n_pairs_per_horizon):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - h))
            out[h].append((i, t0, h))
    return out


def load_pair(ds: Euler2DDataset, i: int, t0: int, h: int) -> tuple:
    """Return (state at t0, state at t0+h, dt) tensors."""
    u0 = ds.frame(i, t0)            # (4, H, W)
    ut = ds.frame(i, t0 + h)        # (4, H, W)
    dt = ds.dt * h
    return u0, ut, dt
