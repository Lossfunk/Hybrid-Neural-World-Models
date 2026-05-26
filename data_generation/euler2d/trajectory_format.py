"""
Unified trajectory format for all NeurIPS envs.

TrajectoryBatch holds a batch of trajectories from one env with ground-truth
event labels (collisions, shocks, contacts, stiffness drops, pericenter passages).

Serialization is HDF5 (not pickle) so the format is language-agnostic and
inspectable with h5dump.
"""
from __future__ import annotations

import json
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np


@dataclass
class TrajectoryBatch:
    # Core arrays
    states: np.ndarray                       # (N, T, D) float32
    actions: Optional[np.ndarray] = None     # (N, T, A) float32 or None
    dt: float = 0.01                         # base timestep (seconds / sim units)

    # Event annotations — each outer list has length N
    event_times: Optional[list] = None       # list[ list[float] ]
    event_types: Optional[list] = None       # list[ list[str] ]
    event_locations: Optional[list] = None   # list[ list[np.ndarray] ]  (e.g. grid indices for PDEs)

    # Provenance
    env_name: str = ""
    env_config: dict = field(default_factory=dict)
    seed: int = 0
    solver_name: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.states.dtype != np.float32:
            self.states = self.states.astype(np.float32)
        if self.states.ndim != 3:
            raise ValueError(f"states must be (N,T,D), got shape {self.states.shape}")
        N = self.states.shape[0]
        T = self.states.shape[1]
        if self.actions is not None:
            if self.actions.dtype != np.float32:
                self.actions = self.actions.astype(np.float32)
            if self.actions.ndim != 3 or self.actions.shape[0] != N or self.actions.shape[1] != T:
                raise ValueError(
                    f"actions shape must be (N,T,A) matching states; got {self.actions.shape}"
                )
        for name in ("event_times", "event_types", "event_locations"):
            v = getattr(self, name)
            if v is not None and len(v) != N:
                raise ValueError(f"{name} has length {len(v)}, expected {N}")

    @property
    def N(self) -> int:
        return int(self.states.shape[0])

    @property
    def T(self) -> int:
        return int(self.states.shape[1])

    @property
    def D(self) -> int:
        return int(self.states.shape[2])


# ── HDF5 serialization ────────────────────────────────────────────────────

def _jsonify(obj):
    """Convert numpy scalars / arrays into JSON-safe primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def save_hdf5(batch: TrajectoryBatch, path: Union[str, Path]) -> Path:
    """Save a TrajectoryBatch to HDF5. Overwrites existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("states", data=batch.states, compression="gzip")
        if batch.actions is not None:
            f.create_dataset("actions", data=batch.actions, compression="gzip")

        f.attrs["dt"] = float(batch.dt)
        f.attrs["env_name"] = batch.env_name
        f.attrs["seed"] = int(batch.seed)
        f.attrs["solver_name"] = batch.solver_name
        f.attrs["env_config_json"] = json.dumps(_jsonify(batch.env_config))
        f.attrs["metadata_json"] = json.dumps(_jsonify(batch.metadata))
        f.attrs["has_actions"] = batch.actions is not None
        f.attrs["format_version"] = "1.0"

        # Events — stored as a group with variable-length per trajectory
        if batch.event_times is not None:
            ev = f.create_group("events")
            ev.attrs["present"] = True
            for i in range(batch.N):
                g = ev.create_group(f"traj_{i:05d}")
                times = np.asarray(batch.event_times[i], dtype=np.float64) \
                        if batch.event_times[i] is not None else np.zeros(0, dtype=np.float64)
                g.create_dataset("times", data=times)

                if batch.event_types is not None and batch.event_types[i] is not None:
                    types = [str(t).encode("utf-8") for t in batch.event_types[i]]
                    g.create_dataset("types",
                                     data=np.asarray(types, dtype="S64") if types else np.zeros(0, dtype="S64"))

                if batch.event_locations is not None and batch.event_locations[i] is not None:
                    locs = batch.event_locations[i]
                    if len(locs) > 0:
                        arr = np.stack([np.asarray(loc, dtype=np.float64).reshape(-1) for loc in locs])
                        g.create_dataset("locations", data=arr)
                        g.attrs["location_shape"] = np.asarray(locs[0]).shape
                    else:
                        g.create_dataset("locations", data=np.zeros((0, 0), dtype=np.float64))
        else:
            f.create_group("events").attrs["present"] = False

    return path


def load_hdf5(path: Union[str, Path]) -> TrajectoryBatch:
    path = Path(path)
    with h5py.File(path, "r") as f:
        states = f["states"][...]
        actions = f["actions"][...] if bool(f.attrs.get("has_actions", False)) else None

        env_config = json.loads(f.attrs["env_config_json"])
        metadata = json.loads(f.attrs["metadata_json"])

        event_times = None
        event_types = None
        event_locations = None
        if "events" in f and bool(f["events"].attrs.get("present", False)):
            ev = f["events"]
            N = states.shape[0]
            event_times = []
            event_types = []
            event_locations = []
            for i in range(N):
                key = f"traj_{i:05d}"
                if key not in ev:
                    event_times.append([])
                    event_types.append([])
                    event_locations.append([])
                    continue
                g = ev[key]
                event_times.append(g["times"][...].tolist())
                if "types" in g:
                    event_types.append([t.decode("utf-8") for t in g["types"][...]])
                else:
                    event_types.append(None)
                if "locations" in g:
                    arr = g["locations"][...]
                    event_locations.append([arr[j] for j in range(arr.shape[0])] if arr.size else [])
                else:
                    event_locations.append(None)

        return TrajectoryBatch(
            states=states,
            actions=actions,
            dt=float(f.attrs["dt"]),
            event_times=event_times,
            event_types=event_types,
            event_locations=event_locations,
            env_name=str(f.attrs["env_name"]),
            env_config=env_config,
            seed=int(f.attrs["seed"]),
            solver_name=str(f.attrs["solver_name"]),
            metadata=metadata,
        )
