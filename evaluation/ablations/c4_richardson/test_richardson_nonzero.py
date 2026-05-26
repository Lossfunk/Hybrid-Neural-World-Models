#!/usr/bin/env python3
"""Verify Richardson disagreement is meaningfully non-zero on the
fixed-stencil solver, but ~0 on the production CFL-adaptive solver.

Picks 5 random states from val.h5, computes Richardson disagreement at
several horizons under both solver configurations.

Production solver: dt_save=0.05, reads dataset's stored states.
Fixed-stencil solver: re-integrates state from a synthetic IC at base_dt=0.025
to confirm CFL stability + non-zero Richardson disagreement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "data_generation" / "oregonator"))

from oregonator2d_tyson import OregonatorTyson2D, TysonParams    # noqa: E402

DATA = ROOT / "data" / "oregonator" / "oregonator_val.h5"


def run_solver(state0, n_steps, params, dt, fixed):
    sim = OregonatorTyson2D(n_x=state0.shape[2], n_y=state0.shape[1],
                              L_x=100.0, L_y=100.0,
                              params=TysonParams(**params),
                              fixed_substep=fixed)
    sim.u[:] = state0[0]
    sim.v[:] = state0[1]
    for _ in range(n_steps):
        sim.step(dt)
    return np.stack([sim.u, sim.v], axis=0).astype(np.float32)


def main():
    rng = np.random.RandomState(7)
    with h5py.File(DATA, "r") as f:
        N, T = f["states"].shape[:2]
        for trial in range(5):
            i = int(rng.randint(0, N))
            t0 = int(rng.randint(0, T - 65))
            state0 = np.array(f["states"][i, t0])
            params = dict(eps=float(f["params"][i, 1]),
                           q=float(f["params"][i, 2]),
                           f=float(f["params"][i, 0]),
                           D=float(f["params"][i, 3]))
            print(f"\n[trial {trial+1}] traj={i} t0={t0}  f={params['f']:.2f}  eps={params['eps']:.3f}")
            for label, dt, fixed in [
                ("PROD adaptive (dt=0.05)", 0.05, False),
                ("FIXED-STENCIL (dt=0.025)", 0.025, True),
            ]:
                print(f"  {label}:")
                for h in [2, 4, 8, 16, 32]:
                    big = run_solver(state0, h, params, dt, fixed)
                    mid = run_solver(state0, h // 2, params, dt, fixed)
                    chain = run_solver(mid, h // 2, params, dt, fixed)
                    e_rich = float(np.sqrt(((big - chain) ** 2).sum(axis=0)).mean())
                    print(f"    h={h:>2d}  Richardson disagreement = {e_rich:.6e}")


if __name__ == "__main__":
    main()
