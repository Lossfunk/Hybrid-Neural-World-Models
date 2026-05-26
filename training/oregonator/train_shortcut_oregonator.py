#!/usr/bin/env python3
"""Train a ShortcutOregonator2D model on the Oregonator dataset.

Loss formulation matches the workshop paper "Shortcut World Models: Learning
to Leap, Not Step" (ICLR 2026 World Models Workshop, Section 4):

    L = 0.9 · L_sup + 0.1 · L_DAgger

  L_sup     — every horizon supervised against ground truth from the solver
  L_DAgger  — chain two predictions at half the step size; supervise the
              chained result against ground truth; gradients flow through
              both passes. 10% of samples per batch (the user-tunable
              dagger_prob), gated to samples whose dt is splittable
              (dt > min_horizon * dt_save + ε).

Forked from training/euler2d/train_shortcut_2d.py with the DAgger
pattern ported from training/train_shortcut_dagger_10pct.py. Uses the
ShortcutOregonator2D model (periodic-BC Conv2d) and the
Oregonator2DDataset.

NOTE: Frans-style self-consistency (compare two model outputs to each other
without GT grounding) is explicitly NOT used here — see the audit in this
project.
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

from shortcut_oregonator_2d import ShortcutOregonator2D, param_count   # noqa: E402
from data_utils_oregonator import (Oregonator2DDataset,                # noqa: E402
                                     InMemoryOregonator2DDataset,
                                     MemmapOregonator2DDataset,
                                     OregonatorShortcutSampler,
                                     compute_channel_stats)
from repro import set_all_seeds, snapshot_env                          # noqa: E402

import math


def normalized_mse_loss(model, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE in normalized state space — channels weighted equally."""
    return F.mse_loss(model.normalize(pred), model.normalize(target))


def train_step(model: ShortcutOregonator2D, u0: torch.Tensor, ut: torch.Tensor,
               dt: torch.Tensor, dagger_prob: float, min_dt: float) -> dict:
    """One gradient step with 90% L_sup + 10% L_DAgger.

    Returns dict with: loss, sup_loss, dagger_loss, n_sup, n_dag, dagger_min_dt.
    """
    device = u0.device
    batch_size = u0.shape[0]

    # 10% of samples flagged for DAgger; gating: only samples where dt > min_dt
    use_dagger = torch.rand(batch_size, device=device) < dagger_prob
    can_split = dt > (min_dt + 1e-9)
    use_dagger = use_dagger & can_split

    sup_idx = (~use_dagger).nonzero(as_tuple=True)[0]
    dag_idx = use_dagger.nonzero(as_tuple=True)[0]

    # Supervised partition (90% in expectation)
    if sup_idx.numel() > 0:
        pred_sup = model(u0[sup_idx], dt[sup_idx])
        sup_loss = normalized_mse_loss(model, pred_sup, ut[sup_idx])
    else:
        sup_loss = torch.tensor(0.0, device=device)

    # DAgger partition (10% in expectation)
    if dag_idx.numel() > 0:
        u0_d = u0[dag_idx]
        ut_d = ut[dag_idx]
        dt_d = dt[dag_idx]
        half_dt = dt_d * 0.5
        # First chained pass — physical-unit input, physical-unit output
        mid = model(u0_d, half_dt)
        # Second chained pass — feed model's own prediction; NO detach.
        # Gradients flow through both passes per the workshop paper.
        final = model(mid, half_dt)
        dagger_loss = normalized_mse_loss(model, final, ut_d)
        dag_min_dt = float(dt_d.min().item())
    else:
        dagger_loss = torch.tensor(0.0, device=device)
        dag_min_dt = float("nan")

    # Combined loss weighted by actual sample fractions (matches reference impl)
    n_sup = int(sup_idx.numel())
    n_dag = int(dag_idx.numel())
    if n_sup > 0 and n_dag > 0:
        w_sup = n_sup / batch_size
        w_dag = n_dag / batch_size
        loss = w_sup * sup_loss + w_dag * dagger_loss
    elif n_sup > 0:
        loss = sup_loss
    else:
        loss = dagger_loss

    return dict(loss=loss, sup_loss=sup_loss, dagger_loss=dagger_loss,
                n_sup=n_sup, n_dag=n_dag, dagger_min_dt=dag_min_dt)


def validate(model, val_ds, horizons, device, n_batches: int = 8,
             batch_size: int = 16) -> dict:
    """Validation that also computes the C3-proxy: Pearson R² between
    step-doubling estimator ê and true error e, pooled across val batches.

    Returns dict with: mse, c3_R2 (overall), c3_R2_per_h (per horizon),
    n_samples_for_c3.
    """
    model.eval()
    rng = np.random.RandomState(12345)
    N, T = val_ds.N, val_ds.T
    total_mse = 0.0
    count = 0
    # accumulators for C3 proxy: pair-level (ê, e)
    e_hat_per_h = {h: [] for h in horizons if h >= 2}
    e_true_per_h = {h: [] for h in horizons if h >= 2}
    with torch.no_grad():
        for _ in range(n_batches):
            hs = rng.choice(horizons, size=batch_size)
            idxs = rng.randint(0, N, size=batch_size)
            u0s, uts, dts, hs_used = [], [], [], []
            for i, h in zip(idxs, hs):
                t0 = rng.randint(0, T - h)
                u0s.append(val_ds.frame(i, t0))
                uts.append(val_ds.frame(i, t0 + h))
                dts.append(h * val_ds.dt)
                hs_used.append(int(h))
            u0 = torch.stack(u0s, dim=0).to(device)
            ut = torch.stack(uts, dim=0).to(device)
            dt = torch.tensor(dts, dtype=torch.float32, device=device)
            pred = model(u0, dt)
            loss = normalized_mse_loss(model, pred, ut)
            total_mse += loss.item() * batch_size
            count += batch_size
            # C3-proxy: chain-prediction for samples with h >= 2
            mask = torch.tensor([h >= 2 for h in hs_used], device=device)
            if mask.any():
                idx_split = mask.nonzero(as_tuple=True)[0]
                u0_s = u0[idx_split]
                dt_s = dt[idx_split]
                ut_s = ut[idx_split]
                pred_s = pred[idx_split]
                pred_mid = model(u0_s, dt_s * 0.5)
                pred_chain = model(pred_mid, dt_s * 0.5)
                # per-sample mean over cells of ‖.‖₂ across channels
                e_hat = torch.sqrt(((pred_s - pred_chain) ** 2).sum(dim=1))
                e_true = torch.sqrt(((pred_s - ut_s) ** 2).sum(dim=1))
                e_hat_pair = e_hat.mean(dim=(1, 2)).cpu().numpy()
                e_true_pair = e_true.mean(dim=(1, 2)).cpu().numpy()
                for j_local, j_full in enumerate(idx_split.cpu().tolist()):
                    h_used = hs_used[j_full]
                    e_hat_per_h[h_used].append(float(e_hat_pair[j_local]))
                    e_true_per_h[h_used].append(float(e_true_pair[j_local]))
    model.train()

    # Pearson R² per horizon and pooled
    def r2(a, b):
        if len(a) < 2: return float("nan")
        a = np.asarray(a); b = np.asarray(b)
        if a.std() < 1e-12 or b.std() < 1e-12: return float("nan")
        return float(np.corrcoef(a, b)[0, 1] ** 2)

    c3_per_h = {h: r2(e_hat_per_h[h], e_true_per_h[h]) for h in e_hat_per_h}
    pooled_e_hat = sum(e_hat_per_h.values(), [])
    pooled_e_true = sum(e_true_per_h.values(), [])
    c3_pooled = r2(pooled_e_hat, pooled_e_true)
    return dict(
        mse=total_mse / max(count, 1),
        c3_R2=c3_pooled,
        c3_R2_per_h=c3_per_h,
        n_samples_for_c3=len(pooled_e_hat),
    )


def train(cfg_path: str, seed_override=None, smoke: bool = False,
          resume_from: str | None = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    runtime_seed = int(seed_override if seed_override is not None
                       else cfg["training"]["seed"])
    set_all_seeds(runtime_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_shortcut_oregonator] device={device} seed={runtime_seed} "
          f"smoke={smoke}", flush=True)

    inmem_n = int(cfg["training"].get("inmemory_train_N", 0) or 0)
    memmap_path = cfg["training"].get("memmap_train_path", None)
    if memmap_path:
        train_ds = MemmapOregonator2DDataset(memmap_path)
    elif inmem_n > 0:
        inmem_dtype = cfg["training"].get("inmemory_dtype", "float32")
        train_ds = InMemoryOregonator2DDataset.from_h5(
            cfg["dataset"]["train"], n_trajs=inmem_n, dtype=inmem_dtype)
    else:
        train_ds = Oregonator2DDataset(cfg["dataset"]["train"])
    val_ds = Oregonator2DDataset(cfg["dataset"]["val"])
    print(f"[train_shortcut_oregonator] train N={train_ds.N} T={train_ds.T} "
          f"CxHxW={train_ds.C}x{train_ds.H}x{train_ds.W}  val N={val_ds.N}",
          flush=True)

    if smoke:
        if hasattr(train_ds, 'states_array'):
            # in-memory: just rebind N
            train_ds.N = min(50, train_ds.N)
        else:
            train_ds.N = min(50, train_ds.N)
    subset_N = int(cfg["training"].get("train_subset_N", 0) or 0)
    if subset_N > 0 and inmem_n == 0:
        train_ds.N = min(subset_N, train_ds.N)
        print(f"[train_shortcut_oregonator] train_subset_N={train_ds.N}", flush=True)

    horizons = list(cfg["horizon_set"])
    min_horizon = min(horizons)
    min_dt_for_dagger = min_horizon * train_ds.dt
    print(f"[train_shortcut_oregonator] horizons={horizons}  "
          f"min_horizon={min_horizon}  min_dt_for_dagger={min_dt_for_dagger:.4f}",
          flush=True)

    sampler = OregonatorShortcutSampler(
        train_ds, horizon_set=horizons, seed=runtime_seed,
        samples_per_epoch=cfg["training"]["samples_per_epoch"],
    )

    ch_mean, ch_std = compute_channel_stats(train_ds, n_trajs=min(50, train_ds.N))
    print(f"[train_shortcut_oregonator] channel_stats mean={ch_mean} std={ch_std}",
          flush=True)

    m_cfg = cfg["model"]
    model = ShortcutOregonator2D(
        channels=int(m_cfg.get("channels", 2)),
        base_ch=int(m_cfg["base_ch"]),
        emb_dim=int(m_cfg["emb_dim"]),
        ch_mults=tuple(m_cfg["ch_mults"]),
        ch_mean=ch_mean, ch_std=ch_std,
    ).to(device)
    print(f"[train_shortcut_oregonator] params: {param_count(model):,}",
          flush=True)

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[train_shortcut_oregonator] resumed model weights from {resume_from} "
              f"(optimizer + scheduler reset)", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    # Cosine LR schedule with optional warmup. Smooths convergence and
    # reduces LR as training plateaus, which helps the C3 R² climb later
    # when sup loss is already converging.
    warmup_epochs = int(cfg["training"].get("warmup_epochs", 0) or 0)
    epochs_for_cosine = int(cfg["training"].get("epochs", 80))
    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        # Cosine from 1.0 → 0.1 over the rest
        progress = (epoch_idx - warmup_epochs) / max(1, epochs_for_cosine - warmup_epochs)
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    output_dir = Path(cfg["output_dir"]) / f"seed{runtime_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_blob = {**cfg, "runtime_seed": runtime_seed, "smoke": smoke,
                    "env_snapshot": snapshot_env()}
    (output_dir / "run_config.json").write_text(
        json.dumps(config_blob, default=str, indent=2))

    epochs = cfg["training"]["epochs"] if not smoke else 5
    ck_every = int(cfg["training"].get("checkpoint_every", 10))
    bs = int(cfg["training"]["batch_size"])
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    dagger_prob = float(cfg["training"].get("dagger_prob", 0.1))

    history = []
    step = 0
    t_start = time.time()
    best_val = float("inf")
    best_c3 = -float("inf")
    # Early stopping: stop when neither val_mse nor c3_R² has improved for
    # `patience` epochs. "Improved" means strictly better than best so far.
    patience = int(cfg["training"].get("patience", 0) or 0)
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total, epoch_sup, epoch_dag = [], [], []
        epoch_n_sup, epoch_n_dag = 0, 0
        epoch_dag_min_dt = []
        for u0, ut, dt in sampler.epoch_iter(bs):
            u0 = u0.to(device, non_blocking=True)
            ut = ut.to(device, non_blocking=True)
            dt = dt.to(device, non_blocking=True)

            out = train_step(model, u0, ut, dt,
                              dagger_prob=dagger_prob,
                              min_dt=min_dt_for_dagger)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"loss NaN at step {step}: total={loss.item()} "
                    f"sup={out['sup_loss'].item()} dag={out['dagger_loss'].item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_total.append(loss.item())
            epoch_sup.append(out["sup_loss"].item())
            epoch_dag.append(out["dagger_loss"].item())
            epoch_n_sup += out["n_sup"]
            epoch_n_dag += out["n_dag"]
            if not np.isnan(out["dagger_min_dt"]):
                epoch_dag_min_dt.append(out["dagger_min_dt"])
            step += 1

        train_total = float(np.mean(epoch_total))
        train_sup = float(np.mean(epoch_sup))
        train_dag = float(np.mean([x for x in epoch_dag if x > 0]) if epoch_n_dag > 0 else 0.0)
        val_out = validate(model, val_ds, horizons, device)
        val_mse = val_out["mse"]
        val_c3 = val_out["c3_R2"]

        record = {
            "epoch": epoch, "train_total": train_total,
            "train_sup": train_sup, "train_dag": train_dag,
            "val_mse": val_mse,
            "val_c3_R2": val_c3,
            "val_c3_R2_per_h": val_out["c3_R2_per_h"],
            "step": step,
            "n_sup": epoch_n_sup, "n_dag": epoch_n_dag,
            "dag_frac": epoch_n_dag / max(epoch_n_sup + epoch_n_dag, 1),
            "elapsed_s": time.time() - t_start,
            "dag_min_dt_seen": (min(epoch_dag_min_dt) if epoch_dag_min_dt else None),
        }
        history.append(record)
        c3_str = f"{val_c3:.3f}" if not (val_c3 != val_c3) else "nan"
        print(f"  epoch {epoch:3d}/{epochs}  total={train_total:.6f}  "
              f"sup={train_sup:.6f}  dag={train_dag:.6f}  "
              f"val={val_mse:.6f}  c3_R²={c3_str}  "
              f"n_sup/n_dag={epoch_n_sup}/{epoch_n_dag} "
              f"({record['dag_frac']*100:.1f}%)", flush=True)

        improved = False
        if val_mse < best_val:
            best_val = val_mse
            improved = True
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_mse": val_mse,
                        "runtime_seed": runtime_seed},
                       output_dir / "best.pt")
        # also count C3 R² improvements (smoothed by 0.01 to ignore noise)
        if (not (val_c3 != val_c3)) and val_c3 > best_c3 + 0.01:
            best_c3 = val_c3
            improved = True
        if improved:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if epoch % ck_every == 0 or epoch == epochs:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "epoch": epoch, "runtime_seed": runtime_seed},
                       output_dir / f"epoch{epoch:03d}.pt")
        scheduler.step()
        # Early stopping
        if patience > 0 and no_improve_epochs >= patience:
            print(f"[early-stop] no improvement in val_mse OR c3_R² for "
                  f"{patience} epochs; stopping at epoch {epoch}/{epochs}",
                  flush=True)
            break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    train_ds.close(); val_ds.close()
    print(f"[train_shortcut_oregonator] done {time.time() - t_start:.1f}s "
          f"-> {output_dir}", flush=True)
    return {"best_val": best_val, "output_dir": str(output_dir),
            "history": history, "runtime_seed": runtime_seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="Path to a checkpoint (best.pt / epochNNN.pt). Loads "
                         "model weights as init; optimizer + scheduler reset.")
    args = ap.parse_args()
    out = train(args.config, args.seed, args.smoke, args.resume)
    print(f"OK  best_val={out['best_val']:.6f}  seed={out['runtime_seed']}  "
          f"dir={out['output_dir']}")


if __name__ == "__main__":
    main()
