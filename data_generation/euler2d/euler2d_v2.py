"""
euler2d_v2: 2D compressible Euler with 4-channel storage and multiple IC
types. Reuses the HLL solver machinery from euler2d.

Conservative state stored: q[..., 4] = (rho, rho*u, rho*v, E).

ICs supported (by config key ic_name):
  - "schulz_rinne_3" : the same as euler2d v1
  - "schulz_rinne_4" : Kurganov-Tadmor 2002, config 4
  - "schulz_rinne_6" : config 6
  - "schulz_rinne_12": config 12
  - "sedov"          : point-source energy deposition
  - "random_mixture" : per-seed choice among the above with random
                       energy perturbation for sedov

Event detector: undivided-difference on pressure, same as v1.
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
from euler2d import (
    prim_from_cons, flux_x, flux_y, hll_flux, _step_hll,
)


def _sr_config_states(config_id: int):
    """Return (q1, q2, q3, q4) primitive states (rho, u, v, p) per quadrant
    for Schulz-Rinne 2D Riemann configurations, following Kurganov-Tadmor 2002.
    Quadrant layout: q1=top-right, q2=top-left, q3=bot-left, q4=bot-right.
    """
    if config_id == 3:
        return (
            (1.5,    0.0,   0.0,   1.5),
            (0.5323, 1.206, 0.0,   0.3),
            (0.138,  1.206, 1.206, 0.029),
            (0.5323, 0.0,   1.206, 0.3),
        )
    if config_id == 4:
        return (
            (1.1,   0.0,     0.0,     1.1),
            (0.5065, 0.8939, 0.0,     0.35),
            (1.1,   0.8939,  0.8939,  1.1),
            (0.5065, 0.0,    0.8939,  0.35),
        )
    if config_id == 6:
        return (
            (1.0,   0.75,   -0.5,    1.0),
            (2.0,   0.75,    0.5,    1.0),
            (1.0,  -0.75,    0.5,    1.0),
            (3.0,  -0.75,   -0.5,    1.0),
        )
    if config_id == 12:
        return (
            (0.5313, 0.0,    0.0,    0.4),
            (1.0,    0.7276, 0.0,    1.0),
            (0.8,    0.0,    0.0,    1.0),
            (1.0,    0.0,    0.7276, 1.0),
        )
    raise ValueError(f"Unknown SR config: {config_id}")


def _schulz_rinne_ic(nx, ny, domain, gamma, config_id=3):
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    xc = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
    yc = np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    q1, q2, q3, q4 = _sr_config_states(config_id)
    # Separator at domain midpoint
    xm = 0.5 * (xmin + xmax)
    ym = 0.5 * (ymin + ymax)
    rho = np.zeros((nx, ny)); u = np.zeros((nx, ny))
    v = np.zeros((nx, ny));   p = np.zeros((nx, ny))
    mask = [
        (X >  xm) & (Y >  ym), q1,
        (X <= xm) & (Y >  ym), q2,
        (X <= xm) & (Y <= ym), q3,
        (X >  xm) & (Y <= ym), q4,
    ]
    for m, st in zip(mask[0::2], mask[1::2]):
        rho[m], u[m], v[m], p[m] = st
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E], axis=-1), dx, dy


def _sedov_ic(nx, ny, domain, gamma, energy=1.0, bg_rho=1.0,
              center=(0.5, 0.5), radius_cells=2):
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    xc = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
    yc = np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    rho = np.full((nx, ny), bg_rho)
    u = np.zeros_like(rho)
    v = np.zeros_like(rho)
    p = np.full((nx, ny), 1e-5)

    # Deposit energy in a small disk of radius_cells
    cx = center[0] * (xmax - xmin) + xmin
    cy = center[1] * (ymax - ymin) + ymin
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    blast_mask = r2 <= (radius_cells * max(dx, dy)) ** 2
    n_blast = int(blast_mask.sum())
    vol_cell = dx * dy
    # Distribute the explosion energy uniformly among blast cells
    p_blast = energy / (vol_cell * n_blast) * (gamma - 1.0)
    p[blast_mask] = p_blast

    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E], axis=-1), dx, dy


def _init_state(ic_name, nx, ny, domain, gamma, rng):
    if ic_name in ("schulz_rinne_3", "schulz_rinne_4", "schulz_rinne_6",
                    "schulz_rinne_12"):
        cid = int(ic_name.split("_")[-1])
        return _schulz_rinne_ic(nx, ny, domain, gamma, config_id=cid), cid, None
    if ic_name == "sedov":
        E0 = float(rng.uniform(0.5, 2.0))
        bg = float(rng.uniform(0.8, 1.2))
        cx = 0.5 + float(rng.uniform(-0.02, 0.02))
        cy = 0.5 + float(rng.uniform(-0.02, 0.02))
        q, dx, dy = _sedov_ic(nx, ny, domain, gamma, energy=E0,
                              bg_rho=bg, center=(cx, cy), radius_cells=2)
        meta = {"ic": "sedov", "E0": E0, "bg_rho": bg, "center": [cx, cy]}
        return (q, dx, dy), None, meta
    if ic_name == "random_mixture":
        # Sample one of the supported ICs per seed
        pool = ["schulz_rinne_3", "schulz_rinne_4", "schulz_rinne_6",
                "schulz_rinne_12", "sedov", "sedov", "sedov"]
        choice = str(rng.choice(pool))
        return _init_state(choice, nx, ny, domain, gamma, rng)
    raise ValueError(f"Unknown ic_name: {ic_name}")


class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        p = cfg["env"]["solver"]["params"]
        self.grid = tuple(p["grid"])
        self.domain = tuple(p["domain"])
        self.gamma = float(p["gamma"])
        self.cfl = float(p["cfl"])
        self.ic_name = str(p.get("ic_name", "random_mixture"))
        self.fixed_frame_dt = bool(p.get("fixed_frame_dt", True))
        self.base_dt = float(cfg["env"]["base_dt"])
        self._rng = None
        self._q = None
        self._dx = self._dy = None
        self._ic_meta = {}

    def reset(self, seed: int):
        self._rng = np.random.RandomState(seed)
        result = _init_state(self.ic_name, self.grid[0], self.grid[1],
                             self.domain, self.gamma, self._rng)
        (q, dx, dy), cid, ic_extra = result
        self._q = q
        self._dx = dx
        self._dy = dy
        self._ic_meta = {"ic_name": self.ic_name, "config_id": cid}
        if ic_extra is not None:
            self._ic_meta.update(ic_extra)
        return self._q.astype(np.float32)

    def step(self, action=None):
        self._q, sub_dt = _step_hll(self._q, self._dx, self._dy, self.gamma, self.cfl)
        return self._q.astype(np.float32), sub_dt

    def _step_fixed_dt(self, target_dt):
        """Advance exactly target_dt seconds by taking variable-number CFL
        substeps, truncating the last one."""
        t_acc = 0.0
        while t_acc < target_dt - 1e-12:
            rho, u, v, p = prim_from_cons(self._q, self.gamma)
            a = np.sqrt(self.gamma * np.maximum(p, 1e-8) / np.maximum(rho, 1e-8))
            max_speed = np.max(np.abs(u) + a) + np.max(np.abs(v) + a)
            sub = self.cfl * min(self._dx, self._dy) / (max_speed + 1e-12)
            sub = min(sub, target_dt - t_acc)
            # Inline HLL step with given sub
            qL = self._q[:-1, :, :]; qR = self._q[1:, :, :]
            Fx = hll_flux(qL, qR, axis=0, gamma=self.gamma)
            qD = self._q[:, :-1, :]; qU = self._q[:, 1:, :]
            Fy = hll_flux(qD, qU, axis=1, gamma=self.gamma)
            q = self._q.copy()
            q[1:-1, :, :] -= (sub / self._dx) * (Fx[1:, :, :] - Fx[:-1, :, :])
            q[:, 1:-1, :] -= (sub / self._dy) * (Fy[:, 1:, :] - Fy[:, :-1, :])
            q[0, :, :] = q[1, :, :]; q[-1, :, :] = q[-2, :, :]
            q[:, 0, :] = q[:, 1, :]; q[:, -1, :] = q[:, -2, :]
            self._q = q
            t_acc += sub

    def rollout(self, T: int, actions=None, seed: int = 0) -> TrajectoryBatch:
        self.reset(seed)
        nx, ny = self.grid
        frames = [self._q.astype(np.float32)]    # (nx, ny, 4)
        pressures = [prim_from_cons(self._q, self.gamma)[3].astype(np.float32)]
        real_times = [0.0]

        t_sim = 0.0
        for t in range(T):
            if self.fixed_frame_dt:
                self._step_fixed_dt(self.base_dt)
                t_sim += self.base_dt
                frames.append(self._q.astype(np.float32))
                pressures.append(prim_from_cons(self._q, self.gamma)[3].astype(np.float32))
                real_times.append(t_sim)
            else:
                self._q, sub_dt = _step_hll(self._q, self._dx, self._dy, self.gamma, self.cfl)
                t_sim += float(sub_dt)
                frames.append(self._q.astype(np.float32))
                pressures.append(prim_from_cons(self._q, self.gamma)[3].astype(np.float32))
                real_times.append(t_sim)

        # Store as (T, nx, ny, 4) flattened to (T, nx*ny*4) to fit in
        # the (N, T, D) TrajectoryBatch convention
        arr = np.stack(frames[:T], axis=0)    # (T, nx, ny, 4)
        states_flat = arr.reshape(T, nx * ny * 4)[None, :, :]   # (1, T, D)

        # Event detection (same sensor as v1)
        threshold = float(self.cfg["env"]["event_detector"]["params"]["threshold"])
        event_times = []; event_types = []; event_locs = []
        for t in range(T):
            P = pressures[t]
            dpx = np.abs(np.diff(P, axis=0))
            dpy = np.abs(np.diff(P, axis=1))
            dpx_full = np.zeros_like(P); dpx_full[:-1, :] = dpx
            dpy_full = np.zeros_like(P); dpy_full[:, :-1] = dpy
            sensor = (dpx_full + dpy_full) / np.maximum(P, 1e-6)
            flagged = np.argwhere(sensor > threshold)
            if flagged.size > 0:
                event_times.append(real_times[t + 1])
                event_types.append("shock")
                event_locs.append(flagged[0].astype(np.float64))

        batch = TrajectoryBatch(
            states=states_flat,
            actions=None,
            dt=float(self.base_dt),
            event_times=[event_times],
            event_types=[event_types],
            event_locations=[event_locs],
            env_name="euler2d_v2",
            env_config={
                "grid": list(self.grid), "domain": list(self.domain),
                "gamma": self.gamma, "cfl": self.cfl,
                "ic_name": self.ic_name,
                "fixed_frame_dt": self.fixed_frame_dt,
            },
            seed=seed,
            solver_name="finite_volume_hll_v2_4ch",
            metadata={
                "gen_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "real_sim_times": real_times[:T],
                "ic_meta": self._ic_meta,
                "discontinuity_density": [float(len(event_times)) / T],
                "discontinuity_density_units": "events_per_timestep",
                "detector": "undivided_difference_pressure",
                "channels": 4,
                "channel_order": ["rho", "rho*u", "rho*v", "E"],
                "nx": nx, "ny": ny,
            },
        )
        return batch


def detect_events(trajectory: TrajectoryBatch, cfg=None):
    return trajectory.event_times, trajectory.event_types, trajectory.event_locations


def load_env_from_config(name: str = "euler2d_v2") -> Env:
    return Env(load_config(name))


def unpack_state(flat_state: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Reshape (..., nx*ny*4) back to (..., nx, ny, 4)."""
    lead = flat_state.shape[:-1]
    return flat_state.reshape(*lead, nx, ny, 4)
