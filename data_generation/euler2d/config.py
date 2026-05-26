"""
Config loader. Merges base.yaml with an env-specific override and returns
a frozen mapping (read-only) so downstream code can't mutate it.

Required fields checked:
  base   : device, num_workers, wandb.project, data_root, seed_default
  env    : name, state_dim, action_dim, base_dt, trajectory_length,
           solver.name, solver.params (dict), event_detector.name,
           event_detector.params (dict)
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Union

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS_DIR = HERE / "configs"
BASE_PATH = CONFIGS_DIR / "base.yaml"


def _freeze(obj):
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _ensure(path, cfg, dotted_key, expected_type=None):
    cur = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise ValueError(f"{path}: missing required key '{dotted_key}'")
        cur = cur[part]
    if expected_type is not None and not isinstance(cur, expected_type):
        # Reject bool when int is expected (bool is-a int in Python, which is wrong here)
        if expected_type is int and isinstance(cur, bool):
            raise ValueError(
                f"{path}: key '{dotted_key}' should be int, got bool"
            )
        type_str = (
            "/".join(t.__name__ for t in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise ValueError(
            f"{path}: key '{dotted_key}' should be {type_str}, "
            f"got {type(cur).__name__}"
        )


# Allowed keys per scope. Unknown keys cause an error so typos get caught.
_ALLOWED_TOP_LEVEL = {
    "device", "num_workers", "wandb", "data_root", "seed_default", "env",
}
_ALLOWED_WANDB = {"project", "mode"}
_ALLOWED_ENV = {
    "name", "state_dim", "action_dim", "base_dt", "trajectory_length",
    "solver", "event_detector",
}
_ALLOWED_SOLVER = {"name", "params"}
_ALLOWED_DETECTOR = {"name", "params"}


def _reject_unknown(path, obj, allowed, scope_name):
    if not isinstance(obj, dict):
        return
    unknown = set(obj.keys()) - allowed
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) in {scope_name}: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )


def load_config(env_path: Union[str, Path]) -> Mapping:
    """
    Load and validate a merged config.

    env_path: either a name like 'particles2d' or a Path to the env yaml.
    Returns a frozen mapping.
    """
    env_path = Path(env_path)
    if not env_path.is_absolute():
        # allow "particles2d" or "env/particles2d.yaml" or actual path
        candidates = [
            CONFIGS_DIR / "env" / f"{env_path.name}.yaml",
            CONFIGS_DIR / env_path,
            HERE.parent / env_path,
            Path.cwd() / env_path,
        ]
        for c in candidates:
            if c.exists():
                env_path = c
                break

    if not env_path.exists():
        raise FileNotFoundError(f"config file not found: {env_path}")

    with open(BASE_PATH) as f:
        base = yaml.safe_load(f) or {}
    with open(env_path) as f:
        env = yaml.safe_load(f) or {}

    merged = _deep_merge(base, {"env": env})

    # Validate required fields
    _ensure(env_path, merged, "device", str)
    _ensure(env_path, merged, "num_workers", int)
    _ensure(env_path, merged, "wandb.project", str)
    _ensure(env_path, merged, "data_root", str)
    _ensure(env_path, merged, "seed_default", int)

    _ensure(env_path, merged, "env.name", str)
    _ensure(env_path, merged, "env.state_dim", int)
    _ensure(env_path, merged, "env.action_dim", int)
    _ensure(env_path, merged, "env.base_dt", (int, float))
    _ensure(env_path, merged, "env.trajectory_length", int)
    _ensure(env_path, merged, "env.solver.name", str)
    _ensure(env_path, merged, "env.solver.params", dict)
    _ensure(env_path, merged, "env.event_detector.name", str)
    _ensure(env_path, merged, "env.event_detector.params", dict)

    # Reject unknown top-level, wandb, env, solver, detector keys.
    _reject_unknown(env_path, merged, _ALLOWED_TOP_LEVEL, "top-level")
    _reject_unknown(env_path, merged.get("wandb", {}), _ALLOWED_WANDB, "wandb")
    _reject_unknown(env_path, merged["env"], _ALLOWED_ENV, "env")
    _reject_unknown(env_path, merged["env"]["solver"], _ALLOWED_SOLVER, "env.solver")
    _reject_unknown(env_path, merged["env"]["event_detector"], _ALLOWED_DETECTOR, "env.event_detector")

    return _freeze(merged)


if __name__ == "__main__":
    # Smoke: load every env config.
    import sys
    env_dir = CONFIGS_DIR / "env"
    ok = True
    for yml in sorted(env_dir.glob("*.yaml")):
        try:
            cfg = load_config(yml)
            assert cfg["env"]["name"]  # read back
            print(f"OK  {yml.name}: name={cfg['env']['name']} "
                  f"state_dim={cfg['env']['state_dim']} "
                  f"solver={cfg['env']['solver']['name']}")
        except Exception as e:
            ok = False
            print(f"FAIL {yml.name}: {type(e).__name__}: {e}")
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
