#!/usr/bin/env python3
"""Train a ShortcutPDE2D model on the Euler v2 dataset with DAgger augmentation.

90% supervised loss + 10% DAgger loss. DAgger: predict the same target via a
2-step chain f(f(s, dt/2), dt/2) and supervise the chained prediction
against GT. The first call is no_grad (treated as rollout). Matches the
recipe used for the Oregonator v3 model.

For horizons of 1 step (the smallest), DAgger is skipped because chaining
at dt/2 would land at 0.5x base_dt which the model was not supervised on
through the supervised stream. The DAgger fraction is averaged only over
items where chaining is meaningful (h >= 2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "data_generation" / "euler2d"))

from shortcut_pde_2d import ShortcutPDE2D, param_count   # noqa: E402
from data_utils_2d import (Euler2DDataset, ShortcutSampler2D,            # noqa: E402
                            ShuffledShortcutSampler2D, compute_channel_stats)
from repro import set_all_seeds, snapshot_env              # noqa: E402


def validate(model, val_ds, horizons, device, n_batches: int = 8, batch_size: int = 16):
    model.eval()
    rng = np.random.RandomState(12345)
    N, T = val_ds.N, val_ds.T
    total = 0.0; count = 0
    with torch.no_grad():
        for _ in range(n_batches):
            hs = rng.choice(horizons, size=batch_size)
            idxs = rng.randint(0, N, size=batch_size)
            u0s, uts, dts = [], [], []
            for i, h in zip(idxs, hs):
                t0 = rng.randint(0, T - h)
                u0s.append(val_ds.frame(i, t0))
                uts.append(val_ds.frame(i, t0 + h))
                dts.append(h * val_ds.dt)
            u0 = torch.stack(u0s, dim=0).to(device)
            ut = torch.stack(uts, dim=0).to(device)
            dt = torch.tensor(dts, dtype=torch.float32, device=device)
            pred = model(u0, dt)
            loss = F.mse_loss(model.normalize(pred), model.normalize(ut))
            total += loss.item() * batch_size
            count += batch_size
    model.train()
    return total / count


def compute_loss(model, u0: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor,
                  base_dt: float, dagger_weight: float):
    """Total loss = (1-w) * supervised + w * DAgger.

    Supervised: pred_direct = model(u0, dt); MSE(pred_direct, ut) in normalized space.
    DAgger:     pred_chain  = model(no_grad(model(u0, dt/2)), dt/2);
                MSE(pred_chain, ut) in normalized space, averaged only over items where
                chaining is well-defined (h >= 2 implying dt >= 2*base_dt).

    Returns (total_loss, components_dict).
    """
    pred_direct = model(u0, dt)
    loss_sup = F.mse_loss(model.normalize(pred_direct), model.normalize(ut))

    if dagger_weight <= 0.0:
        return loss_sup, {"sup": float(loss_sup.item()), "dagger": float("nan"),
                          "dagger_n": 0}

    # Identify items where DAgger chain makes sense: dt >= 2 * base_dt
    # (small epsilon for float compare)
    valid = dt > (1.5 * base_dt)
    if not valid.any():
        return loss_sup, {"sup": float(loss_sup.item()), "dagger": float("nan"),
                          "dagger_n": 0}

    u0_valid = u0[valid]
    ut_valid = ut[valid]
    dt_valid = dt[valid]
    dt_half = dt_valid * 0.5
    with torch.no_grad():
        pred_mid = model(u0_valid, dt_half)
    pred_chain = model(pred_mid, dt_half)
    loss_dagger = F.mse_loss(
        model.normalize(pred_chain), model.normalize(ut_valid),
    )

    total = (1.0 - dagger_weight) * loss_sup + dagger_weight * loss_dagger
    return total, {
        "sup": float(loss_sup.item()),
        "dagger": float(loss_dagger.item()),
        "dagger_n": int(valid.sum().item()),
    }


def train(cfg_path: str, seed_override=None, shuffle_dt=False, smoke=False,
            dagger_weight_override=None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    runtime_seed = int(seed_override if seed_override is not None else cfg["training"]["seed"])
    set_all_seeds(runtime_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dagger_weight = (
        float(dagger_weight_override)
        if dagger_weight_override is not None
        else float(cfg["training"].get("dagger_weight", 0.1))
    )
    print(f"[dagger] device={device} seed={runtime_seed} smoke={smoke}  "
          f"dagger_weight={dagger_weight}", flush=True)

    train_ds = Euler2DDataset(cfg["dataset"]["train"])
    val_ds = Euler2DDataset(cfg["dataset"]["val"])
    print(f"[dagger] train N={train_ds.N} T={train_ds.T} "
          f"HxW={train_ds.H}x{train_ds.W}  val N={val_ds.N}  base_dt={train_ds.dt}",
          flush=True)

    if smoke:
        train_ds.N = min(50, train_ds.N)
    subset_N = int(cfg["training"].get("train_subset_N", 0) or 0)
    if subset_N > 0:
        train_ds.N = min(subset_N, train_ds.N)
        print(f"[dagger] train_subset_N={train_ds.N}", flush=True)

    horizons = list(cfg["horizon_set"])

    sampler_cls = ShuffledShortcutSampler2D if shuffle_dt else ShortcutSampler2D
    sampler = sampler_cls(
        train_ds, horizon_set=horizons, seed=runtime_seed,
        samples_per_epoch=cfg["training"]["samples_per_epoch"],
    )

    ch_mean, ch_std = compute_channel_stats(train_ds, n_trajs=min(50, train_ds.N))
    print(f"[dagger] channel_stats mean={ch_mean} std={ch_std}", flush=True)

    m_cfg = cfg["model"]
    model = ShortcutPDE2D(
        channels=int(m_cfg.get("channels", 4)),
        base_ch=int(m_cfg["base_ch"]),
        emb_dim=int(m_cfg["emb_dim"]),
        ch_mults=tuple(m_cfg["ch_mults"]),
        ch_mean=ch_mean, ch_std=ch_std,
    ).to(device)
    print(f"[dagger] params: {param_count(model):,}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    output_dir = Path(cfg["output_dir"])
    if shuffle_dt:
        output_dir = output_dir.parent / (output_dir.name + "_shuffled")
    output_dir = output_dir / f"seed{runtime_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_blob = {**cfg, "runtime_seed": runtime_seed, "shuffle_dt": shuffle_dt,
                   "smoke": smoke, "dagger_weight": dagger_weight,
                   "env_snapshot": snapshot_env()}
    (output_dir / "run_config.json").write_text(json.dumps(config_blob, default=str, indent=2))

    epochs = cfg["training"]["epochs"] if not smoke else 3
    ck_every = int(cfg["training"].get("checkpoint_every", 10))
    bs = int(cfg["training"]["batch_size"])
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    base_dt = float(train_ds.dt)

    history = []
    step = 0
    t_start = time.time()
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_sup = []
        epoch_dagger = []
        for u0, ut, dt in sampler.epoch_iter(bs):
            u0 = u0.to(device, non_blocking=True)
            ut = ut.to(device, non_blocking=True)
            dt = dt.to(device, non_blocking=True)

            loss, comps = compute_loss(model, u0, ut, dt,
                                          base_dt=base_dt,
                                          dagger_weight=dagger_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss NaN at step {step}: {loss}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_losses.append(float(loss.item()))
            epoch_sup.append(comps["sup"])
            if not np.isnan(comps["dagger"]):
                epoch_dagger.append(comps["dagger"])
            step += 1

        train_total = float(np.mean(epoch_losses))
        train_sup = float(np.mean(epoch_sup))
        train_dagger = float(np.mean(epoch_dagger)) if epoch_dagger else float("nan")
        val_mse = validate(model, val_ds, horizons, device)
        history.append({"epoch": epoch, "train_total": train_total,
                        "train_sup": train_sup, "train_dagger": train_dagger,
                        "val_mse": val_mse, "step": step,
                        "elapsed_s": time.time() - t_start})
        print(f"  epoch {epoch:3d}/{epochs}  total={train_total:.6f}  "
              f"sup={train_sup:.6f}  dagger={train_dagger:.6f}  "
              f"val_mse={val_mse:.6f}  step={step}", flush=True)

        if val_mse < best_val:
            best_val = val_mse
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_mse": val_mse,
                        "runtime_seed": runtime_seed, "shuffle_dt": shuffle_dt,
                        "dagger_weight": dagger_weight},
                       output_dir / "best.pt")
        if epoch % ck_every == 0 or epoch == epochs:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "epoch": epoch, "runtime_seed": runtime_seed,
                        "shuffle_dt": shuffle_dt,
                        "dagger_weight": dagger_weight},
                       output_dir / f"epoch{epoch:03d}.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    train_ds.close(); val_ds.close()
    print(f"[dagger] done {time.time() - t_start:.1f}s  -> {output_dir}", flush=True)
    return {"best_val": best_val, "output_dir": str(output_dir),
            "history": history, "runtime_seed": runtime_seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shuffle_dt", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dagger_weight", type=float, default=None,
                     help="override the config dagger_weight (default 0.1)")
    args = ap.parse_args()
    out = train(args.config, args.seed, args.shuffle_dt, args.smoke,
                  dagger_weight_override=args.dagger_weight)
    print(f"OK  best_val={out['best_val']:.6f}  seed={out['runtime_seed']}  dir={out['output_dir']}")


if __name__ == "__main__":
    main()
