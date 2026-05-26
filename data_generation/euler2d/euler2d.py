"""
euler2d: 2D compressible Euler on a 128x128 grid with HLL fluxes.

Ground truth: first-order Godunov-type finite-volume with HLL (Harten-Lax-
van Leer) numerical flux. Initial condition: classic four-quadrant 2D
Riemann problem (Schulz-Rinne configuration 3: top-right/top-left/
bottom-left/bottom-right states with shock/contact/rarefaction combinations).

Event detector: undivided-difference shock sensor applied to the pressure
field. A cell (i,j) is flagged when
    (|p_{i+1,j} - p_{i,j}| + |p_{i,j+1} - p_{i,j}|) / p_{i,j}  >  threshold.

We chose the undivided-difference sensor rather than Ducros because it only
needs pressure (Ducros needs velocity divergence and curl, which adds cost
and numerical noise at first order). Documented in SETUP_LOG.md.

Trajectory state stored: the density field, flattened to D=16384 floats per
step. Other channels are kept internal.
"""
from __future__ import annotations

import sys
import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_format import TrajectoryBatch
from config import load_config


# ── Conservative variables: q = [rho, rho*u, rho*v, E] ─────────────────────
def prim_from_cons(q, gamma):
    rho = q[..., 0]
    u = q[..., 1] / rho
    v = q[..., 2] / rho
    E = q[..., 3]
    p = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    return rho, u, v, p


def flux_x(q, gamma):
    rho, u, v, p = prim_from_cons(q, gamma)
    f = np.empty_like(q)
    f[..., 0] = rho * u
    f[..., 1] = rho * u * u + p
    f[..., 2] = rho * u * v
    f[..., 3] = u * (q[..., 3] + p)
    return f


def flux_y(q, gamma):
    rho, u, v, p = prim_from_cons(q, gamma)
    f = np.empty_like(q)
    f[..., 0] = rho * v
    f[..., 1] = rho * u * v
    f[..., 2] = rho * v * v + p
    f[..., 3] = v * (q[..., 3] + p)
    return f


def hll_flux(qL, qR, axis, gamma):
    rhoL, uL, vL, pL = prim_from_cons(qL, gamma)
    rhoR, uR, vR, pR = prim_from_cons(qR, gamma)
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)

    if axis == 0:  # x
        FL = flux_x(qL, gamma)
        FR = flux_x(qR, gamma)
        sL = np.minimum(uL - aL, uR - aR)
        sR = np.maximum(uL + aL, uR + aR)
    else:          # y
        FL = flux_y(qL, gamma)
        FR = flux_y(qR, gamma)
        sL = np.minimum(vL - aL, vR - aR)
        sR = np.maximum(vL + aL, vR + aR)

    sL_b = sL[..., None]
    sR_b = sR[..., None]
    F = np.where(
        sL_b >= 0, FL,
        np.where(sR_b <= 0, FR,
                 (sR_b * FL - sL_b * FR + sL_b * sR_b * (qR - qL)) / (sR_b - sL_b + 1e-12)),
    )
    return F


def _four_quadrant_ic(nx, ny, domain, gamma):
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    xc = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
    yc = np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    # Schulz-Rinne Config 3: four states separated by x=0, y=0
    # (rho, u, v, p) per quadrant
    q1 = (1.5, 0.0, 0.0, 1.5)          # top-right  (x>0, y>0)
    q2 = (0.5323, 1.206, 0.0, 0.3)     # top-left
    q3 = (0.138, 1.206, 1.206, 0.029)  # bot-left
    q4 = (0.5323, 0.0, 1.206, 0.3)     # bot-right

    rho = np.zeros((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))
    mask = [
        (X > 0) & (Y > 0), q1,
        (X <= 0) & (Y > 0), q2,
        (X <= 0) & (Y <= 0), q3,
        (X > 0) & (Y <= 0), q4,
    ]
    for m, st in zip(mask[0::2], mask[1::2]):
        rho[m], u[m], v[m], p[m] = st

    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    q = np.stack([rho, rho * u, rho * v, E], axis=-1)
    return q, dx, dy


def _step_hll(q, dx, dy, gamma, cfl):
    rho, u, v, p = prim_from_cons(q, gamma)
    a = np.sqrt(gamma * np.maximum(p, 1e-8) / np.maximum(rho, 1e-8))
    max_speed = np.max(np.abs(u) + a) + np.max(np.abs(v) + a)
    dt = cfl * min(dx, dy) / (max_speed + 1e-12)

    # x-fluxes at interfaces i+1/2
    qL = q[:-1, :, :]
    qR = q[1:, :, :]
    Fx = hll_flux(qL, qR, axis=0, gamma=gamma)   # shape (nx-1, ny, 4)

    # y-fluxes at interfaces j+1/2
    qD = q[:, :-1, :]
    qU = q[:, 1:, :]
    Fy = hll_flux(qD, qU, axis=1, gamma=gamma)

    qnew = q.copy()
    qnew[1:-1, :, :] -= (dt / dx) * (Fx[1:, :, :] - Fx[:-1, :, :])
    qnew[:, 1:-1, :] -= (dt / dy) * (Fy[:, 1:, :] - Fy[:, :-1, :])
    # outflow (zero-gradient) BCs
    qnew[0, :, :] = qnew[1, :, :]
    qnew[-1, :, :] = qnew[-2, :, :]
    qnew[:, 0, :] = qnew[:, 1, :]
    qnew[:, -1, :] = qnew[:, -2, :]
    return qnew, dt


class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        p = cfg["env"]["solver"]["params"]
        self.grid = tuple(p["grid"])
        self.domain = tuple(p["domain"])
        self.gamma = float(p["gamma"])
        self.cfl = float(p["cfl"])
        self.final_time = float(p.get("final_time", 0.25))
        self.base_dt = float(cfg["env"]["base_dt"])
        self._rng = None
        self._q = None
        self._dx = self._dy = None

    def reset(self, seed: int):
        self._rng = np.random.RandomState(seed)
        q, dx, dy = _four_quadrant_ic(self.grid[0], self.grid[1], self.domain, self.gamma)
        self._q = q
        self._dx = dx
        self._dy = dy
        return q[..., 0].astype(np.float32)   # density field

    def step(self, action=None):
        self._q, sub_dt = _step_hll(self._q, self._dx, self._dy, self.gamma, self.cfl)
        return self._q[..., 0].astype(np.float32), sub_dt

    def rollout(self, T: int, actions=None, seed: int = 0) -> TrajectoryBatch:
        self.reset(seed)
        nx, ny = self.grid
        densities = [self._q[..., 0].astype(np.float32).reshape(-1)]
        pressures = [prim_from_cons(self._q, self.gamma)[3].astype(np.float32)]
        real_times = [0.0]

        t_sim = 0.0
        for t in range(T):
            self._q, sub_dt = _step_hll(self._q, self._dx, self._dy, self.gamma, self.cfl)
            t_sim += float(sub_dt)
            densities.append(self._q[..., 0].astype(np.float32).reshape(-1))
            pressures.append(prim_from_cons(self._q, self.gamma)[3].astype(np.float32))
            real_times.append(t_sim)

        states_arr = np.stack(densities[:T], axis=0)[None, :, :]   # (1, T, nx*ny)

        # Event detection: undivided-difference sensor on pressure, per step
        threshold = float(self.cfg["env"]["event_detector"]["params"]["threshold"])
        event_times_this: List[float] = []
        event_types_this: List[str] = []
        event_locs_this: List[np.ndarray] = []
        for t in range(T):
            P = pressures[t]
            dpx = np.abs(np.diff(P, axis=0))
            dpy = np.abs(np.diff(P, axis=1))
            # Pad to full grid size
            dpx_full = np.zeros_like(P)
            dpx_full[:-1, :] = dpx
            dpy_full = np.zeros_like(P)
            dpy_full[:, :-1] = dpy
            sensor = (dpx_full + dpy_full) / np.maximum(P, 1e-6)
            flagged = np.argwhere(sensor > threshold)
            if flagged.size > 0:
                # Emit one event-per-step tagged with the first flagged cell
                # (we could emit all, but that explodes storage).
                event_times_this.append(real_times[t + 1])
                event_types_this.append("shock")
                event_locs_this.append(flagged[0].astype(np.float64))

        batch = TrajectoryBatch(
            states=states_arr,
            actions=None,
            dt=float(self.base_dt),   # nominal; real time varies via CFL
            event_times=[event_times_this],
            event_types=[event_types_this],
            event_locations=[event_locs_this],
            env_name="euler2d",
            env_config={
                "grid": list(self.grid), "domain": list(self.domain),
                "gamma": self.gamma, "cfl": self.cfl,
                "ic": "four_quadrant_schulz_rinne_3",
            },
            seed=seed,
            solver_name="finite_volume_hll_v1",
            metadata={
                "gen_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "real_sim_times": real_times[:T],
                "discontinuity_density": [float(len(event_times_this)) / T],
                "discontinuity_density_units": "events_per_timestep",
                "detector": "undivided_difference_pressure",
            },
        )
        return batch


def detect_events(trajectory: TrajectoryBatch, cfg=None):
    return trajectory.event_times, trajectory.event_types, trajectory.event_locations


def load_env_from_config(name: str = "euler2d") -> Env:
    return Env(load_config(name))
