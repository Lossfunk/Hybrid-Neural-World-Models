#!/usr/bin/env python3
"""Train ShortcutBall3D with multi-horizon supervision + 10% DAgger.

Mirrors the Oregonator/Euler recipe but for the 9-dim ball state:
  - 90% supervised loss at GT horizons in {1, 2, 4, 8, 16, 32, 64} steps
  - 10% DAgger loss (chained-prediction supervised against GT, no_grad on first call)
  - AdamW + cosine LR + grad clip
  - Per-channel mean/std normalization
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "models"))

from shortcut_ball3d import ShortcutBall3D, param_count       # noqa: E402


HORIZONS = [1, 2, 4, 8, 16, 32, 64]


class Ball3DDataset:
    """Loads the (N, T, 9) array fully into RAM (small dataset)."""
    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            self.states = np.array(f["states"], dtype=np.float32)
            self.dt = float(f.attrs["dt_save"])
        self.N, self.T, self.D = self.states.shape

    def frame(self, i: int, t: int) -> torch.Tensor:
        return torch.from_numpy(self.states[i, t])


def compute_channel_stats(ds: Ball3DDataset) -> tuple:
    arr = ds.states.reshape(-1, ds.D)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    return mean, std


def sample_batch(ds: Ball3DDataset, horizons: list, batch_size: int,
                  rng: np.random.RandomState) -> tuple:
    hs = rng.choice(horizons, size=batch_size)
    idxs = rng.randint(0, ds.N, size=batch_size)
    s0s = np.empty((batch_size, ds.D), dtype=np.float32)
    sts = np.empty((batch_size, ds.D), dtype=np.float32)
    dts = np.empty((batch_size,), dtype=np.float32)
    for k, (i, h) in enumerate(zip(idxs, hs)):
        t0 = rng.randint(0, ds.T - h)
        s0s[k] = ds.states[i, t0]
        sts[k] = ds.states[i, t0 + h]
        dts[k] = h * ds.dt
    return torch.from_numpy(s0s), torch.from_numpy(sts), torch.from_numpy(dts)


def compute_loss(model, s0, st, dt, base_dt, dagger_weight: float):
    pred = model(s0, dt)
    sup = F.mse_loss(model.normalize(pred), model.normalize(st))
    if dagger_weight <= 0:
        return sup, {"sup": float(sup.item()), "dagger": float("nan"), "dagger_n": 0}
    valid = dt > 1.5 * base_dt
    if not valid.any():
        return sup, {"sup": float(sup.item()), "dagger": float("nan"), "dagger_n": 0}
    s0_v = s0[valid]; st_v = st[valid]; dt_v = dt[valid]
    dt_half = dt_v * 0.5
    with torch.no_grad():
        s_mid = model(s0_v, dt_half)
    s_chain = model(s_mid, dt_half)
    dag = F.mse_loss(model.normalize(s_chain), model.normalize(st_v))
    total = (1 - dagger_weight) * sup + dagger_weight * dag
    return total, {"sup": float(sup.item()), "dagger": float(dag.item()),
                   "dagger_n": int(valid.sum().item())}


def validate(model, val_ds, device, n_batches=8, batch_size=64,
              rng_seed=12345):
    model.eval()
    rng = np.random.RandomState(rng_seed)
    total = 0.0; count = 0
    with torch.no_grad():
        for _ in range(n_batches):
            s0, st, dt = sample_batch(val_ds, HORIZONS, batch_size, rng)
            s0 = s0.to(device); st = st.to(device); dt = dt.to(device)
            pred = model(s0, dt)
            loss = F.mse_loss(model.normalize(pred), model.normalize(st))
            total += loss.item() * batch_size
            count += batch_size
    model.train()
    return total / count


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    train_ds = Ball3DDataset(args.train_path)
    val_ds = Ball3DDataset(args.val_path)
    print(f"[ball3d] train N={train_ds.N} T={train_ds.T} D={train_ds.D}", flush=True)
    print(f"[ball3d] val   N={val_ds.N}", flush=True)

    ch_mean, ch_std = compute_channel_stats(train_ds)
    print(f"[ball3d] channel_mean={ch_mean}", flush=True)
    print(f"[ball3d] channel_std ={ch_std}", flush=True)

    model = ShortcutBall3D(
        state_dim=train_ds.D,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        n_blocks=args.n_blocks,
        ch_mean=ch_mean, ch_std=ch_std,
    ).to(device)
    print(f"[ball3d] params: {param_count(model):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * args.steps_per_epoch
    )

    out_dir = Path(args.output_dir) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    history = []
    best_val = float("inf")
    base_dt = float(train_ds.dt)
    t_start = time.time()
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = []
        epoch_sup = []
        epoch_dag = []
        for _ in range(args.steps_per_epoch):
            s0, st, dt = sample_batch(train_ds, HORIZONS, args.batch_size, rng)
            s0 = s0.to(device); st = st.to(device); dt = dt.to(device)
            loss, comps = compute_loss(model, s0, st, dt, base_dt,
                                          dagger_weight=args.dagger_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            epoch_total.append(float(loss.item()))
            epoch_sup.append(comps["sup"])
            if not np.isnan(comps["dagger"]):
                epoch_dag.append(comps["dagger"])
            step += 1

        train_total = float(np.mean(epoch_total))
        train_sup = float(np.mean(epoch_sup))
        train_dag = float(np.mean(epoch_dag)) if epoch_dag else float("nan")
        val_mse = validate(model, val_ds, device)
        history.append({"epoch": epoch, "train_total": train_total,
                        "train_sup": train_sup, "train_dagger": train_dag,
                        "val_mse": val_mse, "step": step,
                        "elapsed_s": time.time() - t_start,
                        "lr": float(sched.get_last_lr()[0])})
        print(f"  epoch {epoch:3d}/{args.epochs}  total={train_total:.6f}  "
              f"sup={train_sup:.6f}  dag={train_dag:.6f}  "
              f"val={val_mse:.6f}  lr={sched.get_last_lr()[0]:.2e}  "
              f"({time.time()-t_start:.0f}s)", flush=True)

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "ch_mean": ch_mean.tolist(), "ch_std": ch_std.tolist(),
                "epoch": epoch, "val_mse": val_mse,
            }, out_dir / "best.pt")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "ch_mean": ch_mean.tolist(), "ch_std": ch_std.tolist(),
                "epoch": epoch,
            }, out_dir / f"epoch{epoch:03d}.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[ball3d] done {time.time()-t_start:.1f}s, best_val={best_val:.6f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default=str(ROOT / "data" / "ball3d" / "ball3d_train.h5"))
    ap.add_argument("--val_path",   default=str(ROOT / "data" / "ball3d" / "ball3d_val.h5"))
    ap.add_argument("--output_dir", default=str(ROOT / "checkpoints" / "ball3d"))
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--emb_dim",    type=int, default=64)
    ap.add_argument("--n_blocks",   type=int, default=4)
    ap.add_argument("--epochs",     type=int, default=80)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size",   type=int, default=128)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--grad_clip",     type=float, default=1.0)
    ap.add_argument("--dagger_weight", type=float, default=0.1)
    ap.add_argument("--checkpoint_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
