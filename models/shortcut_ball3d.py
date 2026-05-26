"""Shortcut model for 3D ball state prediction.

Input:  state (B, 9)  [pos 3, linvel 3, angvel 3]
       dt    (B,)    horizon in seconds
Output: state (B, 9)  predicted state at t + dt

Architecture: small MLP with FiLM conditioning on dt. Same recipe as
oregonator/euler but adapted for low-dim state. 9-dim → MLP → 9-dim
delta added to input (residual prediction in normalized space).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_dt_embedding(dt: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=dt.device, dtype=dt.dtype) / half
    )
    ang = dt[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMBlock(nn.Module):
    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.emb_proj = nn.Linear(emb_dim, dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x)))
        scale_shift = self.emb_proj(emb)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        h = h * (1.0 + scale) + shift
        h = self.act(self.norm2(self.fc2(h)))
        return x + h         # residual


class ShortcutBall3D(nn.Module):
    def __init__(self, state_dim: int = 9, hidden_dim: int = 256,
                 emb_dim: int = 64, n_blocks: int = 4,
                 ch_mean=None, ch_std=None):
        super().__init__()
        self.state_dim = state_dim
        self.emb_dim = emb_dim

        mean = torch.zeros(state_dim) if ch_mean is None else torch.as_tensor(ch_mean, dtype=torch.float32)
        std  = torch.ones(state_dim)  if ch_std  is None else torch.as_tensor(ch_std,  dtype=torch.float32)
        assert mean.shape == (state_dim,) and std.shape == (state_dim,)
        self.register_buffer("ch_mean", mean)
        self.register_buffer("ch_std", std)

        self.dt_embed = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMBlock(hidden_dim, emb_dim) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def normalize(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.ch_mean) / self.ch_std

    def denormalize(self, s: torch.Tensor) -> torch.Tensor:
        return s * self.ch_std + self.ch_mean

    def forward(self, s: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() > 1:
            dt = dt.squeeze(-1)
        dt_raw = sinusoidal_dt_embedding(dt.to(s.dtype), self.emb_dim)
        emb = self.dt_embed(dt_raw)

        s_norm = self.normalize(s)
        h = self.input_proj(s_norm)
        for blk in self.blocks:
            h = blk(h, emb)
        delta_norm = self.output_proj(h)
        return self.denormalize(s_norm + delta_norm)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = ShortcutBall3D()
    s = torch.randn(4, 9)
    dt = torch.tensor([0.01, 0.05, 0.1, 0.5])
    y = m(s, dt)
    print(f"ShortcutBall3D params: {param_count(m):,}")
    print(f"in={tuple(s.shape)}, dt={tuple(dt.shape)}, out={tuple(y.shape)}")
