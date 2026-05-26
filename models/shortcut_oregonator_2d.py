"""Shortcut world model for 2D Oregonator state — periodic-BC fork of
models/shortcut_pde_2d.py.

Differences from the Euler 2D model:
  - default channels = 2 (u, v) instead of 4 (ρ, ρu, ρv, E)
  - all Conv2d layers use padding_mode="circular" (Oregonator has periodic BCs;
    Euler's IC types — Schulz-Rinne / Sedov / Double Mach — are non-periodic
    and use zero-padding)

Architecture, FiLM-Δt conditioning, residual head, and per-channel
normalization are otherwise identical.

Input:
  u:  (B, C, H, W) float32   state with C=2 channels (u, v)
  dt: (B,)         float32   horizon in non-dim time units
Output:
  u_pred: (B, C, H, W) float32 predicted state at t + dt
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


class ConvBlock2dCircular(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="circular")
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2) if emb_dim > 0 else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        if self.emb_proj is not None and emb is not None:
            scale_shift = self.emb_proj(emb)[:, :, None, None]
            scale, shift = torch.chunk(scale_shift, 2, dim=1)
            h = h * (1.0 + scale) + shift
        h = self.act(self.norm2(self.conv2(h)))
        return h


class ShortcutOregonator2D(nn.Module):
    def __init__(self, channels: int = 2, base_ch: int = 32, emb_dim: int = 64,
                 ch_mults=(1, 2, 4), ch_mean=None, ch_std=None):
        super().__init__()
        self.channels = channels
        self.emb_dim = emb_dim

        mean = torch.zeros(channels) if ch_mean is None else torch.as_tensor(ch_mean, dtype=torch.float32)
        std = torch.ones(channels) if ch_std is None else torch.as_tensor(ch_std, dtype=torch.float32)
        assert mean.shape == (channels,) and std.shape == (channels,)
        self.register_buffer("ch_mean", mean)
        self.register_buffer("ch_std", std)

        self.dt_embed = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        chs = [base_ch * m for m in ch_mults]
        self.stem = nn.Conv2d(channels, chs[0], 3, padding=1, padding_mode="circular")

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        for i, c in enumerate(chs):
            self.enc_blocks.append(ConvBlock2dCircular(prev, c, emb_dim))
            if i < len(chs) - 1:
                self.downs.append(nn.AvgPool2d(2))
            prev = c

        self.bot = ConvBlock2dCircular(chs[-1], chs[-1], emb_dim)

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.dec_blocks.append(ConvBlock2dCircular(chs[i] + chs[i - 1], chs[i - 1], emb_dim))

        self.head = nn.Sequential(
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], channels, 3, padding=1, padding_mode="circular"),
        )

    def normalize(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.ch_mean.view(1, -1, 1, 1)) / self.ch_std.view(1, -1, 1, 1)

    def denormalize(self, u: torch.Tensor) -> torch.Tensor:
        return u * self.ch_std.view(1, -1, 1, 1) + self.ch_mean.view(1, -1, 1, 1)

    def forward(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Accepts u in physical units, returns predicted next state in
        physical units. Internal U-Net operates in normalized space.
        Output is u + delta (residual prediction)."""
        if dt.dim() > 1:
            dt = dt.squeeze(-1)
        dt_raw = sinusoidal_dt_embedding(dt.to(u.dtype), self.emb_dim)
        emb = self.dt_embed(dt_raw)

        u_norm = self.normalize(u)
        h = self.stem(u_norm)
        skips = []
        for i, blk in enumerate(self.enc_blocks):
            h = blk(h, emb)
            if i < len(self.enc_blocks) - 1:
                skips.append(h)
                h = self.downs[i](h)

        h = self.bot(h, emb)

        for up, dblk, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dblk(h, emb)

        delta_norm = self.head(h)
        return self.denormalize(u_norm + delta_norm)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = ShortcutOregonator2D(channels=2, base_ch=32, ch_mults=(1, 2, 4))
    u = torch.randn(2, 2, 64, 64)
    dt = torch.tensor([0.05, 0.4])
    y = m(u, dt)
    print(f"ShortcutOregonator2D (base_ch=32, 3 levels) params: {param_count(m):,}")
    print(f"in={tuple(u.shape)} out={tuple(y.shape)}")
