"""
Shortcut world model for 2D PDE state (Euler v2).

Input:
  u:  (B, C, H, W) float32   state with C=4 conserved channels (rho, rhou, rhov, E)
  dt: (B,)          float32  horizon in seconds
Output:
  u_pred: (B, C, H, W) float32 predicted state at t + dt

Architecture: 2D U-Net with FiLM-style sinusoidal Δt conditioning. Output
is a delta added to the input; residual prediction matches the 1D model.
Zero-padding on boundaries (Euler test ICs are non-periodic: Schulz-Rinne,
Sedov, Double Mach; outflow/wall BCs handled by the solver, not the net).
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


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
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


class ShortcutPDE2D(nn.Module):
    def __init__(self, channels: int = 4, base_ch: int = 32, emb_dim: int = 64,
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
        self.stem = nn.Conv2d(channels, chs[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        for i, c in enumerate(chs):
            self.enc_blocks.append(ConvBlock2d(prev, c, emb_dim))
            if i < len(chs) - 1:
                self.downs.append(nn.AvgPool2d(2))
            prev = c

        self.bot = ConvBlock2d(chs[-1], chs[-1], emb_dim)

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.dec_blocks.append(ConvBlock2d(chs[i] + chs[i - 1], chs[i - 1], emb_dim))

        self.head = nn.Sequential(
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], channels, 3, padding=1),
        )

    def normalize(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.ch_mean.view(1, -1, 1, 1)) / self.ch_std.view(1, -1, 1, 1)

    def denormalize(self, u: torch.Tensor) -> torch.Tensor:
        return u * self.ch_std.view(1, -1, 1, 1) + self.ch_mean.view(1, -1, 1, 1)

    def forward(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Accepts u in physical units and returns predicted next state in
        physical units. Network interior operates in normalized space."""
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
    m = ShortcutPDE2D(channels=4, base_ch=32, ch_mults=(1, 2, 4))
    u = torch.randn(2, 4, 128, 128)
    dt = torch.tensor([0.002, 0.01])
    y = m(u, dt)
    print(f"ShortcutPDE2D (base_ch=32, 3 levels) params: {param_count(m):,}")
    print(f"in={tuple(u.shape)} out={tuple(y.shape)}")

    m4 = ShortcutPDE2D(channels=4, base_ch=32, ch_mults=(1, 2, 4, 4))
    y4 = m4(u, dt)
    print(f"ShortcutPDE2D (base_ch=32, 4 levels) params: {param_count(m4):,}")
