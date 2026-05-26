"""2D Oregonator (Tyson 1985 / Tyson-Fife 2-variable reduction), periodic BC.

Equations (non-dimensional, no time/length units — D set to 1):
    ∂u/∂t = D·∇²u + (1/ε) [u(1−u) − f·v·(u−q)/(u+q)]
    ∂v/∂t = u − v                       (v doesn't diffuse in the standard Tyson form)

State per cell: (u, v). Two channels.

This is the canonical numerical Oregonator used in Jahnke 1989, Tyson 1985,
Barkley's spiral-wave studies — derived from the 3-variable Field-Noyes
Oregonator by setting w to its quasi-steady-state value f·v/(q+u) under the
limit ε' → 0. The chemistry is identical to BZ; the simplification is
purely numerical.

Parameters (Tyson 1985 / Jahnke 1989, give T ≈ 1.7-2 in these units):
    ε = 0.05
    q = 0.002
    f = 2.0
    D = 1.0
On a domain of size L × L with L ≈ 100, dx ≈ 0.4, the spiral wavelength
λ = c·T ≈ √(D/ε)·T ≈ 4.5·1.7 ≈ 7.6 covers ~20 grid cells — well-resolved.

Discretization
--------------
- Spatial:   2D periodic 5-point Laplacian via np.roll.
- Time:      Strang split — react(dt/2) ∘ diffuse(dt) ∘ react(dt/2).
- Reaction:  vectorized implicit Euler with closed-form 2×2 Jacobian.
- Diffusion: explicit FTCS, internal CFL substepping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class TysonParams:
    eps: float = 0.05      # excitation timescale
    q: float = 0.002       # rate-constant ratio
    f: float = 2.0         # excitability (sweepable)
    D: float = 1.0         # uniform diffusivity (non-dim)


CHANNEL_NAMES = ("u", "v")


class OregonatorTyson2D:
    def __init__(self, n_x: int = 256, n_y: int = 256,
                 L_x: float = 100.0, L_y: float = 100.0,
                 params: TysonParams | None = None,
                 fixed_substep: bool = False):
        """fixed_substep=False (default): production CFL-stability-adaptive
        solver. Internal substepping: explicit FTCS diffusion uses
        n_sub = ceil(dt / 0.4·dx²/(4D)); implicit-Euler reaction uses
        n_sub = ceil(dt / 4ε'). Effectively exact at any requested macro-dt
        (truncation error is below numerical precision).

        fixed_substep=True: NO internal substepping. Single explicit
        diffusion step + single implicit-Euler reaction step at the
        requested macro-dt. Has O(dt²) truncation error — meaningfully
        non-zero, suitable as a Richardson extrapolation baseline. NB:
        diffusion is FTCS-unstable for dt > dx²/(4D) ≈ 0.038 on our 256²
        grid; choose macro-dt accordingly (eg dt=0.025).
        """
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.L_x = float(L_x)
        self.L_y = float(L_y)
        self.dx = self.L_x / self.n_x
        self.dy = self.L_y / self.n_y
        self.x = np.linspace(self.dx / 2, self.L_x - self.dx / 2, self.n_x)
        self.y = np.linspace(self.dy / 2, self.L_y - self.dy / 2, self.n_y)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")
        self.params = params if params is not None else TysonParams()
        self.fixed_substep = bool(fixed_substep)
        self.u = np.zeros((self.n_y, self.n_x), dtype=np.float64)
        self.v = np.zeros((self.n_y, self.n_x), dtype=np.float64)
        self.t_sim = 0.0

    # ── Initial conditions ────────────────────────────────────────────────

    def reset_spiral(self, v_refractory: float = 0.5) -> None:
        """Barkley broken-front spiral seed (Barkley 1991, Eqn 5).

        Seed = a single excited "strip" in the upper half (y > L/2,
        x < L/2) — i.e., the left quarter of the upper half. v is set to
        the refractory level in the LOWER half — but only on the LEFT
        half (x < L/2) — so the recovery variable trails the wave at
        the broken edge.

        This produces a single phase singularity at the corner (L/2, L/2)
        rather than the periodic-BC artifacts of a full half-plane IC.
        v in the right half stays at 0, so the periodic boundary at
        x=L_x has matching values on both sides (no spurious wave source).
        """
        cx = self.L_x / 2
        cy = self.L_y / 2
        self.u[:] = 0.0
        self.v[:] = 0.0
        # excited region: upper-LEFT quadrant only
        excited = (self.X < cx) & (self.Y > cy)
        self.u[excited] = 1.0
        # refractory v: lower-LEFT quadrant only (the wave's "trail")
        refrac = (self.X < cx) & (self.Y < cy)
        self.v[refrac] = float(v_refractory)
        self.t_sim = 0.0

    def reset_target(self, x0_frac: float = 0.5, y0_frac: float = 0.5,
                     r_frac: float = 0.05) -> None:
        """Small high-u disk in quiescent background — emits target waves."""
        x0 = x0_frac * self.L_x
        y0 = y0_frac * self.L_y
        r = r_frac * min(self.L_x, self.L_y)
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.u[(self.X - x0) ** 2 + (self.Y - y0) ** 2 < r ** 2] = 1.0
        self.t_sim = 0.0

    def reset_random(self, seed: int = 0, burn_in_steps: int = 200,
                     burn_in_dt: float = 0.05) -> None:
        rng = np.random.RandomState(int(seed))
        self.u[:] = rng.uniform(0.0, 0.5, size=(self.n_y, self.n_x))
        self.v[:] = rng.uniform(0.0, 0.5, size=(self.n_y, self.n_x))
        self.t_sim = 0.0
        for _ in range(int(burn_in_steps)):
            self.step(burn_in_dt)
        self.t_sim = 0.0

    # ── Reaction ──────────────────────────────────────────────────────────

    def _reaction_rates(self, u: np.ndarray, v: np.ndarray) -> tuple:
        eps = self.params.eps
        q = self.params.q
        f = self.params.f
        # Guarded denominator (u+q is never 0 in practice; floor for safety)
        denom = u + q
        R_u = (u * (1.0 - u) - f * v * (u - q) / denom) / eps
        R_v = u - v
        return R_u, R_v

    def _reaction_jacobian(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Closed-form 2×2 Jacobian, returned as (n_cells, 2, 2)."""
        eps = self.params.eps
        q = self.params.q
        f = self.params.f
        u_f = u.ravel()
        v_f = v.ravel()
        denom = u_f + q
        denom2 = denom * denom
        n_cells = u.size
        J = np.empty((n_cells, 2, 2), dtype=np.float64)
        # ∂R_u/∂u = (1/ε) [(1-2u) - f·v·2q/(u+q)²]
        J[:, 0, 0] = ((1.0 - 2.0 * u_f) - f * v_f * 2.0 * q / denom2) / eps
        # ∂R_u/∂v = -(1/ε) f·(u-q)/(u+q)
        J[:, 0, 1] = -f * (u_f - q) / denom / eps
        # ∂R_v/∂u = 1
        J[:, 1, 0] = 1.0
        # ∂R_v/∂v = -1
        J[:, 1, 1] = -1.0
        return J

    def _reaction_step(self, dt: float, n_newton: int = 8,
                        tol: float = 1e-10) -> None:
        """Vectorized implicit Euler:  y_{n+1} = y_n + dt · R(y_{n+1}).

        Newton iterates are clipped to physical bounds at each step to
        prevent unphysical excursions at sharp IC discontinuities. The
        physical Tyson Oregonator has u ∈ [0, 1] and v ≈ u (so v ∈ [0, 1])
        in steady-state spiral motion; we use generous bounds [0, 1.5]
        and [0, 2.0] to allow brief overshoot while preventing runaway.
        """
        u_old = self.u
        v_old = self.v
        u_n = self.u.copy()
        v_n = self.v.copy()
        eye2 = np.eye(2)[None, :, :]
        n_cells = self.u.size
        F = np.empty((n_cells, 2), dtype=np.float64)
        for _it in range(int(n_newton)):
            R_u, R_v = self._reaction_rates(u_n, v_n)
            F[:, 0] = (u_n - u_old - dt * R_u).ravel()
            F[:, 1] = (v_n - v_old - dt * R_v).ravel()
            J_R = self._reaction_jacobian(u_n, v_n)
            J = eye2 - dt * J_R
            delta = np.linalg.solve(J, -F)
            u_n += delta[:, 0].reshape(self.u.shape)
            v_n += delta[:, 1].reshape(self.v.shape)
            # Per-iteration clip prevents Newton excursions from leaving the
            # physical manifold and getting stuck at unphysical fixed points.
            np.clip(u_n, 0.0, 1.5, out=u_n)
            np.clip(v_n, 0.0, 2.0, out=v_n)
            if np.max(np.abs(delta)) < tol:
                break
        self.u, self.v = u_n, v_n

    # ── Diffusion ─────────────────────────────────────────────────────────

    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0)
                + np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1)
                - 4.0 * f) / (self.dx * self.dx)

    def _diffusion_step(self, dt: float) -> None:
        D = self.params.D
        if D <= 0.0:
            return
        # Standard Tyson form: only u diffuses
        if self.fixed_substep:
            # Single explicit FTCS step. CFL-violating for dt > dx²/(4D);
            # caller's responsibility to choose dt accordingly.
            self.u = self.u + dt * D * self._laplacian(self.u)
            return
        dt_max = 0.4 * self.dx * self.dx / (4.0 * D)
        n_sub = max(1, int(np.ceil(dt / dt_max)))
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            self.u = self.u + dt_sub * D * self._laplacian(self.u)

    # ── Outer step ────────────────────────────────────────────────────────

    def step(self, dt: float, scheme: str = "strang") -> None:
        if scheme == "strang":
            self._reaction_step(dt * 0.5)
            self._diffusion_step(dt)
            self._reaction_step(dt * 0.5)
        else:
            self._reaction_step(dt)
            self._diffusion_step(dt)
        self.t_sim += dt

    # ── Diagnostics ───────────────────────────────────────────────────────

    def saved_state(self) -> np.ndarray:
        return np.stack([self.u, self.v], axis=0)

    def rollout(self, n_save_frames: int, dt_save: float,
                progress_every: int = 0) -> dict:
        n_channels = 2
        frames = np.empty((n_save_frames, n_channels, self.n_y, self.n_x),
                           dtype=np.float32)
        times = np.empty(n_save_frames, dtype=np.float64)
        t_start = time.time()
        for k in range(n_save_frames):
            frames[k] = self.saved_state().astype(np.float32)
            times[k] = self.t_sim
            if progress_every and k % progress_every == 0:
                print(f"  frame {k:4d}/{n_save_frames}  t={self.t_sim:.3f}  "
                      f"u_max={self.u.max():.3f}  v_max={self.v.max():.3f}  "
                      f"wall={time.time() - t_start:.1f}s", flush=True)
            if k == n_save_frames - 1:
                break
            self.step(dt_save)
        return dict(states=frames, times=times)


if __name__ == "__main__":
    sim = OregonatorTyson2D(n_x=128, n_y=128, L_x=100.0, L_y=100.0)
    sim.reset_spiral()
    print(f"Pre-rollout: u in [{sim.u.min():.3f}, {sim.u.max():.3f}]  "
          f"v in [{sim.v.min():.3f}, {sim.v.max():.3f}]")
    # dt=0.05 keeps dt/ε = 1, which Newton handles cleanly. dt=0.1 (dt/ε=2)
    # makes Newton oscillate and v can shoot to large unphysical values.
    out = sim.rollout(n_save_frames=40, dt_save=0.05, progress_every=10)
    print(f"Final: u_max={sim.u.max():.3f} v_max={sim.v.max():.3f}")
