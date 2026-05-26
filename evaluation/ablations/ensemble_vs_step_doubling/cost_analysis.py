#!/usr/bin/env python3
"""Cost analysis: how step-doubling and K-ensemble scale with K.
Simple deterministic numbers from training history files + parameter counts.
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent

def runtime_for(env: str):
    """Sum elapsed_s across the seed=0 history if available."""
    if env == "oregonator":
        p = ROOT / "checkpoints" / \
              "shortcut_oregonator_v3" / "seed0" / "history.json"
    elif env == "euler":
        p = ROOT / "training" / "checkpoints" / \
              "shortcut_euler2d_v2_dagger" / "seed0" / "history.json"
    else:
        return None
    if not p.exists():
        return None
    h = json.load(open(p))
    if isinstance(h, list):
        # last epoch's elapsed_s is the cumulative wall time
        if h and "elapsed_s" in h[-1]:
            return float(h[-1]["elapsed_s"])
        # otherwise sum per-epoch deltas if any
    return None


def param_count(env: str):
    if env == "oregonator":
        ck = torch.load(ROOT / "checkpoints" / "oregonator" / "best.pt",
                          map_location="cpu", weights_only=False)
    elif env == "euler":
        ck = torch.load(ROOT / "checkpoints" / "euler2d" / "best.pt",
                          map_location="cpu", weights_only=False)
    else:
        return None
    return sum(v.numel() for v in ck["model_state_dict"].values())


def storage_mb(env: str):
    if env == "oregonator":
        p = ROOT / "checkpoints" / \
              "shortcut_oregonator_v3" / "seed0" / "best.pt"
    elif env == "euler":
        p = ROOT / "training" / "checkpoints" / \
              "shortcut_euler2d_v2_dagger" / "seed0" / "best.pt"
    else:
        return None
    return p.stat().st_size / (1024 ** 2)


def main():
    print("=== Per-env single-seed costs ===\n")
    print(f"{'Env':>10}  {'params':>12}  {'storage MB':>12}  {'training s':>14}  {'training h':>10}")
    rows = {}
    for env in ("oregonator", "euler"):
        params = param_count(env)
        st_mb = storage_mb(env)
        rt_s = runtime_for(env)
        rt_h = rt_s / 3600 if rt_s else None
        rows[env] = (params, st_mb, rt_s, rt_h)
        print(f"{env:>10}  {params:>12,}  {st_mb:>12.2f}  {rt_s:>14.1f}  {rt_h:>10.2f}")
    print()

    print("=== K-ensemble cost scaling ===\n")
    print(f"{'K':>4}  {'env':>10}  {'storage':>10}  {'training time':>16}  "
            f"{'inference passes':>18}  {'GPU mem multiplier':>20}")
    print("-" * 90)
    for K in [1, 3, 5, 10, 20]:
        for env in ("oregonator", "euler"):
            params, st_mb, rt_s, rt_h = rows[env]
            mem_mult = K  # each model needs its own param + activation memory
            print(f"{K:>4}  {env:>10}  {K*st_mb:>9.1f}MB  {K*rt_h:>15.2f}h  "
                    f"{K:>18d}  {mem_mult:>17d}x")
    print()

    print("=== Step-doubling cost (constant in K) ===\n")
    for env in ("oregonator", "euler"):
        params, st_mb, rt_s, rt_h = rows[env]
        print(f"{env:>10}  storage={st_mb:.2f}MB  training={rt_h:.2f}h  "
                f"inference passes per pair = 3  GPU mem = 1x")
    print()

    print("=== Inference passes per pair (K-dependent) ===\n")
    print(f"{'K':>4}  {'SD passes':>12}  {'ensemble passes':>16}  {'cost ratio en/sd':>20}")
    sd_passes = 3
    for K in [3, 5, 10, 20]:
        en_passes = K
        ratio = en_passes / sd_passes
        print(f"{K:>4}  {sd_passes:>12d}  {en_passes:>16d}  {ratio:>17.2f}x")


if __name__ == "__main__":
    main()
