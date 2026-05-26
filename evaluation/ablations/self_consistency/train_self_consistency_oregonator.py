#!/usr/bin/env python3
"""Train ShortcutOregonator with Frans-style self-consistency loss in
place of GT supervision.

Loss: L = ‖f(s, T) − f(f(s, T/2), T/2)‖²    (no ground truth)
Goal: demonstrate the user's observation that this collapses to the
trivial fixed point f(s, T) ≈ s in physical state space.

Reduced 20 epochs — failure mode appears within ~5-10 epochs typically.
We track val MSE *against ground truth* per epoch to show the model
diverges from the true dynamics even as L_SC goes to zero.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "training" / "oregonator"))

from shortcut_oregonator_2d import ShortcutOregonator2D
from data_utils_oregonator import (Oregonator2DDataset,
                                       OregonatorShortcutSampler)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalized_mse_loss(model, pred, target):
    return F.mse_loss(model.normalize(pred), model.normalize(target))


def train_step_self_consistency(model, u0, dt, base_dt, min_horizon=1):
    """Self-consistency only: L = ||f(s,T) - f(f(s,T/2), T/2)||^2 (NO GT).

    Only applied to samples where T can be split (T >= 2 * min_horizon * base_dt).
    """
    device = u0.device
    splittable = dt > (2 * min_horizon * base_dt + 1e-9)
    if splittable.sum() == 0:
        return None
    u0_s = u0[splittable]
    dt_s = dt[splittable]
    pred_full = model(u0_s, dt_s)
    pred_mid = model(u0_s, dt_s * 0.5)
    pred_chain = model(pred_mid, dt_s * 0.5)
    loss = normalized_mse_loss(model, pred_full, pred_chain)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_horizon", type=int, default=1)
    ap.add_argument("--data_path", default=str(ROOT / "data" / "oregonator" / "oregonator_train.h5"))
    ap.add_argument("--val_path", default=str(ROOT / "data" / "oregonator" / "oregonator_val.h5"))
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "checkpoints" / "self_consistency_oreg"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Oregonator2DDataset(args.data_path)
    val_ds = Oregonator2DDataset(args.val_path)

    horizons = [1, 2, 4, 8, 16, 32, 64]
    train_sampler = OregonatorShortcutSampler(train_ds, horizons, seed=args.seed,
                                  samples_per_epoch=args.batch_size * args.steps_per_epoch)
    val_sampler = OregonatorShortcutSampler(val_ds, horizons, seed=args.seed + 1000,
                                  samples_per_epoch=args.batch_size * 30)
    base_dt = float(train_ds.dt)

    # Channel stats for normalizer
    print("Computing channel stats...", flush=True)
    sample_states = train_ds.states[:50, ::20]            # (N_seg, T, 2, H, W)
    sample = torch.from_numpy(sample_states.reshape(-1, sample_states.shape[2], sample_states.shape[3], sample_states.shape[4]))
    ch_mean = sample.mean(dim=(0, 2, 3))
    ch_std = sample.std(dim=(0, 2, 3))
    print(f"  ch_mean={ch_mean.tolist()}, ch_std={ch_std.tolist()}", flush=True)

    model = ShortcutOregonator2D(channels=2, base_ch=32, emb_dim=64,
                                       ch_mults=(1, 2, 2, 4),
                                       ch_mean=ch_mean, ch_std=ch_std).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    t0_run = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_arr = []
        for step, (u0_b, ut_b, dt_b) in enumerate(train_sampler.epoch_iter(args.batch_size)):
            if step >= args.steps_per_epoch: break
            u0_b = u0_b.to(DEVICE); dt_b = dt_b.to(DEVICE)
            loss = train_step_self_consistency(model, u0_b, dt_b, base_dt,
                                                  min_horizon=args.min_horizon)
            if loss is None: continue
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_arr.append(loss.item())

        # Validate against GT
        model.eval()
        val_losses = []
        first_batch = None
        with torch.no_grad():
            for k, (u0_v, ut_v, dt_v) in enumerate(val_sampler.epoch_iter(args.batch_size)):
                if k >= 20: break
                u0_v = u0_v.to(DEVICE); ut_v = ut_v.to(DEVICE); dt_v = dt_v.to(DEVICE)
                pred = model(u0_v, dt_v)
                v = F.mse_loss(pred, ut_v).item()
                val_losses.append(v)
                if first_batch is None:
                    first_batch = (u0_v, pred)
        # trivial fixed point check
        u0_v, pred = first_batch
        trivial_dist = F.mse_loss(pred, u0_v).item()       # if ≈0 → model output ≈ input

        train_loss = float(np.mean(loss_arr))
        val_loss = float(np.mean(val_losses))
        elapsed = time.time() - t0_run
        history.append({"epoch": epoch, "train_sc_loss": train_loss,
                          "val_mse_vs_GT": val_loss,
                          "trivial_fixed_point_dist": trivial_dist,
                          "elapsed_s": elapsed})
        print(f"  epoch {epoch:3d}/{args.epochs}  L_SC={train_loss:.6f}  "
                f"val_MSE_vs_GT={val_loss:.6f}  trivial_FP_dist={trivial_dist:.6f}  "
                f"({elapsed:.0f}s)", flush=True)

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"channels": 2, "base_ch": 32, "emb_dim": 64, "ch_mults": [1,2,2,4]},
        "ch_mean": ch_mean, "ch_std": ch_std,
    }, out_dir / "best.pt")
    Path(out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
