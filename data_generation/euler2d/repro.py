"""
Reproducibility harness.

- snapshot_env()       : returns a dict describing the runtime (git, pip, CUDA).
- set_all_seeds(seed)  : sets python, numpy, torch, CUDA seeds. Also
                         tightens torch CUDA determinism flags.
- hash_dataset(path)   : sha256 of a file (streaming).
- @reproducible(seed)  : decorator that snapshots env, sets seeds, writes a
                         run record to a run directory (default ./runs/<ts>/).

Every experiment script should use @reproducible.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional


# ── Env snapshot ──────────────────────────────────────────────────────────

def _run(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def _git_hash() -> str:
    h = _run("git rev-parse HEAD")
    dirty = _run("git status --porcelain")
    return (h + ("-dirty" if dirty else "")) if h else "no-git"


def _cuda_info() -> dict:
    info = {"cuda_available": False}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["num_devices"] = torch.cuda.device_count()
    except Exception as e:
        info["error"] = str(e)
    return info


def snapshot_env() -> dict:
    pip_freeze = _run(f"{sys.executable} -m pip freeze")
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cuda": _cuda_info(),
        "pip_freeze": pip_freeze.splitlines(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }


# ── Seeding ───────────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Tighten determinism. Users can relax for perf.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ── Dataset hash ──────────────────────────────────────────────────────────

def hash_dataset(path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── @reproducible decorator ───────────────────────────────────────────────

def reproducible(seed: Optional[int] = None, run_root: Optional[str] = None):
    """
    Wrap an entrypoint:

        @reproducible(seed=42)
        def main(...):
            ...

    The wrapper:
      1. Resolves seed (uses `seed` kwarg on call if not fixed here).
      2. Calls set_all_seeds.
      3. Writes a run_record.json with snapshot_env() + args/kwargs +
         return value to <run_root>/<timestamp>/.
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            s = kwargs.pop("_seed_override", seed)
            if s is None:
                raise ValueError("@reproducible needs a seed (at decorator or via _seed_override kwarg)")
            set_all_seeds(int(s))
            rroot = Path(run_root) if run_root else Path("runs")
            rdir = rroot / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            rdir.mkdir(parents=True, exist_ok=True)

            env = snapshot_env()
            record = {
                "fn": fn.__name__,
                "module": fn.__module__,
                "seed": int(s),
                "env": env,
                "args_repr": [repr(a) for a in args],
                "kwargs_repr": {k: repr(v) for k, v in kwargs.items()},
            }
            out = fn(*args, **kwargs)
            try:
                json.dumps(out)
                record["return"] = out
            except Exception:
                record["return_repr"] = repr(out)
            (rdir / "run_record.json").write_text(json.dumps(record, indent=2, default=str))
            return out
        wrapper.__wrapped_run_dir_parent__ = run_root
        return wrapper
    return decorator
