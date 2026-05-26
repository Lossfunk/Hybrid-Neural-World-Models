"""3D ball-in-box environment, pure numpy (no mujoco contact solver).

State: (x, y, z, vx, vy, vz, wx, wy, wz) = 9-dim
   3-pos + 3-linvel + 3-angvel.

Note: we drop quaternion since the ball is symmetric and orientation doesn't
affect free-flight dynamics. Angular velocity is preserved (no friction
torque on bounces).

Physics:
  - Free flight: classical kinematics under gravity
  - Wall collision: reflection of velocity component normal to wall, multiplied
    by restitution. Tangential velocity unchanged (frictionless).
  - Box: [-L, L] x [-L, L] x [0, H], default L=0.5, H=1.0

Per-trajectory parameters:
  - restitution e ∈ [0, 1] (energy retained on bounce)
  - gravity g
  - initial position, velocity, angular velocity
"""
from __future__ import annotations

import numpy as np


class Ball3DEnv:
    def __init__(self, L: float = 0.5, H: float = 1.0,
                  ball_radius: float = 0.05):
        self.L = L
        self.H = H
        self.r = ball_radius
        # Walls: each is (axis, side) where side is +1 or -1
        # axis: 0=x, 1=y, 2=z. side: +1 means upper wall (e.g., x=+L), -1 lower.
        # For z: only the floor (z=0) gives lower-side bounce; ceiling at z=H gives upper-side
        self.walls = [
            (0, +1, +L),   # x = +L
            (0, -1, -L),
            (1, +1, +L),
            (1, -1, -L),
            (2, +1, +H),
            (2, -1, 0.0),
        ]

    def reset(self, seed: int,
              v_min: float = 1.0, v_max: float = 3.0,
              w_max: float = 5.0,
              restitution: float = 0.85,
              gravity: float = -9.81) -> np.ndarray:
        rng = np.random.RandomState(int(seed))
        # Initial position: random within box (with margin = ball_radius)
        margin = self.r * 1.5
        x = float(rng.uniform(-self.L + margin, self.L - margin))
        y = float(rng.uniform(-self.L + margin, self.L - margin))
        z = float(rng.uniform(margin, self.H - margin))
        # Initial linear velocity: uniform on sphere * magnitude
        v_dir = rng.normal(size=3); v_dir /= np.linalg.norm(v_dir) + 1e-12
        v_mag = float(rng.uniform(v_min, v_max))
        vx, vy, vz = v_dir * v_mag
        # Initial angular velocity
        wx = float(rng.uniform(-w_max, w_max))
        wy = float(rng.uniform(-w_max, w_max))
        wz = float(rng.uniform(-w_max, w_max))
        # Store
        self._state = np.array([x, y, z, vx, vy, vz, wx, wy, wz], dtype=np.float64)
        self._restitution = float(restitution)
        self._gravity_vec = np.array([0.0, 0.0, float(gravity)])
        return self._state.astype(np.float32).copy()

    def step(self, dt: float, n_internal: int = 50) -> np.ndarray:
        """Advance physics by dt seconds using n_internal sub-steps + collisions."""
        sub_dt = dt / n_internal
        s = self._state.copy()
        for _ in range(n_internal):
            # Free flight: semi-implicit Euler
            s[3:6] += self._gravity_vec * sub_dt
            s[0:3] += s[3:6] * sub_dt
            # Wall collisions (loop until no wall is penetrated; usually 0 or 1 iteration)
            for _wall_iter in range(4):
                bounced = False
                for axis, side, position in self.walls:
                    if side > 0:
                        # Penetration when s[axis] + r > position, with v[axis] > 0 (moving outward)
                        if s[axis] + self.r > position and s[axis + 3] > 0:
                            s[axis] = position - self.r
                            s[axis + 3] = -s[axis + 3] * self._restitution
                            bounced = True
                    else:
                        if s[axis] - self.r < position and s[axis + 3] < 0:
                            s[axis] = position + self.r
                            s[axis + 3] = -s[axis + 3] * self._restitution
                            bounced = True
                if not bounced:
                    break
        self._state = s
        return self._state.astype(np.float32).copy()

    def rollout(self, n_frames: int, dt_save: float) -> np.ndarray:
        out = np.empty((n_frames + 1, 9), dtype=np.float32)
        out[0] = self._state.astype(np.float32)
        for k in range(n_frames):
            out[k + 1] = self.step(dt_save)
        return out


if __name__ == "__main__":
    env = Ball3DEnv()
    s0 = env.reset(seed=0, restitution=0.85)
    print(f"Initial state: pos={s0[:3]}, linvel={s0[3:6]}, angvel={s0[6:9]}")
    states = env.rollout(n_frames=200, dt_save=0.01)
    print(f"\nRollout shape: {states.shape}")
    print(f"z range: [{states[:,2].min():.3f}, {states[:,2].max():.3f}]")
    print(f"x range: [{states[:,0].min():.3f}, {states[:,0].max():.3f}]")
    print(f"y range: [{states[:,1].min():.3f}, {states[:,1].max():.3f}]")
    vmag = np.linalg.norm(states[:, 3:6], axis=1)
    print(f"\nVelocity magnitudes: init={vmag[0]:.3f}, max={vmag.max():.3f}, "
          f"final={vmag[-1]:.3f}")
    # Bouncing pattern via z peaks
    print(f"\nz peaks (local maxima with z > 0.06):")
    for i in range(1, len(states) - 1):
        if states[i, 2] > states[i-1, 2] and states[i, 2] > states[i+1, 2] and states[i, 2] > 0.06:
            print(f"  t={i*0.01:.2f}s  z={states[i,2]:.3f}")
    # Sanity: stays in box
    pos = states[:, :3]
    in_box = ((np.abs(pos[:, 0]) < env.L + 0.01) & (np.abs(pos[:, 1]) < env.L + 0.01) &
                (pos[:, 2] > -0.01) & (pos[:, 2] < env.H + 0.01))
    print(f"\nFraction inside box: {in_box.mean():.3f}")
    print(f"NaN count: {int(np.sum(np.isnan(states)))}")
