#!/usr/bin/env python3
"""P0.2 — Learned error head ablation (Option B: post-hoc on frozen f_θ).

Trains a small CNN head that taps the U-Net's last decoder features and
predicts the per-cell error magnitude ‖f_θ(s, Δt) − Φ(s, Δt)‖. Compares
its AUROC against the existing step-doubling ê AUROC.

Speed-tuned version: pre-loads all training pairs into GPU memory upfront,
trains end-to-end through frozen U-Net (no feature cache).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT_AB = HERE.parent
ROOT_ALL = ROOT_AB.parent
sys.path.insert(0, str(ROOT_ALL / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT_ALL / "models"))

from eval_utils import load_model      # noqa: E402
from shortcut_oregonator_2d import sinusoidal_dt_embedding  # noqa: E402

CKPT = ROOT_ALL / "checkpoints" / "oregonator" / "best.pt"
DATA_DIR = ROOT_ALL / "data" / "oregonator"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = ["test", "ood_near", "ood_far"]
HORIZONS = [2, 4, 8, 16, 32, 64]
DT_BASE = 0.05
N_TRAIN_SAMPLES = 3000
N_EVAL_PER_H = 80
N_EPOCHS = 12
BATCH_SIZE = 32


# ─────────────────────────────────────────────────────────────────────────
def forward_with_features(model, u: torch.Tensor, dt: torch.Tensor):
    """Run the frozen U-Net; return (prediction, last decoder features)."""
    if dt.dim() > 1:
        dt = dt.squeeze(-1)
    dt_raw = sinusoidal_dt_embedding(dt.to(u.dtype), model.emb_dim)
    emb = model.dt_embed(dt_raw)

    u_norm = model.normalize(u)
    h = model.stem(u_norm)
    skips = []
    for i, blk in enumerate(model.enc_blocks):
        h = blk(h, emb)
        if i < len(model.enc_blocks) - 1:
            skips.append(h)
            h = model.downs[i](h)
    h = model.bot(h, emb)
    for up, dblk, skip in zip(model.ups, model.dec_blocks, reversed(skips)):
        h = up(h)
        h = torch.cat([h, skip], dim=1)
        h = dblk(h, emb)
    features = h
    delta_norm = model.head(h)
    pred = model.denormalize(u_norm + delta_norm)
    return pred, features


# ─────────────────────────────────────────────────────────────────────────
class ErrorHead(nn.Module):
    """Per-cell error magnitude prediction. Input: U-Net last decoder
    features (B, base_ch, H, W). Output: (B, 1, H, W) ≥ 0."""
    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x))


# ─────────────────────────────────────────────────────────────────────────
def preload_data(ds_path: Path, n: int, horizons: list, seed: int = 0):
    """Sample (i, t0, h) and read all pairs into RAM as numpy arrays."""
    rng = np.random.RandomState(seed)
    with h5py.File(ds_path, "r") as f:
        N, T = f["states"].shape[:2]
    triplets = []
    for _ in range(n):
        i = int(rng.randint(0, N))
        h = int(horizons[rng.randint(len(horizons))])
        t0 = int(rng.randint(0, T - h))
        triplets.append((i, t0, h))
    # Sort by traj index to get sequential HDF5 reads (much faster)
    order = np.argsort([t[0] * 10000 + t[1] for t in triplets])
    triplets = [triplets[k] for k in order]

    states = np.empty((n, 2, 256, 256), dtype=np.float32)
    targets = np.empty((n, 2, 256, 256), dtype=np.float32)
    dts = np.empty(n, dtype=np.float32)
    print(f"  preloading {n} pairs from {ds_path.name} ...", flush=True)
    t0_ = time.time()
    with h5py.File(ds_path, "r") as f:
        s = f["states"]
        for k, (i, t0, h) in enumerate(triplets):
            states[k] = s[i, t0]
            targets[k] = s[i, t0 + h]
            dts[k] = h * DT_BASE
            if k % 500 == 0 and k > 0:
                print(f"    {k}/{n}  ({time.time()-t0_:.1f}s)", flush=True)
    print(f"  preloaded in {time.time()-t0_:.1f}s", flush=True)
    return states, targets, dts


# ─────────────────────────────────────────────────────────────────────────
def train_head_inline(model, head, states_np, targets_np, dts_np,
                       n_epochs: int, batch_size: int, lr: float = 1e-3):
    """End-to-end training: forward through frozen U-Net (no_grad) to get
    features+pred, compute true error label, train head with backprop on
    head-only params."""
    n = len(states_np)
    states_t = torch.from_numpy(states_np)     # CPU; pin if helpful
    targets_t = torch.from_numpy(targets_np)
    dts_t = torch.from_numpy(dts_np)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    print(f"  training: {n} samples × {n_epochs} epochs, batch={batch_size}",
          flush=True)
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total = 0.0; nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s = states_t[idx].to(DEVICE, non_blocking=True)
            tgt = targets_t[idx].to(DEVICE, non_blocking=True)
            dt = dts_t[idx].to(DEVICE, non_blocking=True)
            with torch.no_grad():
                pred, feats = forward_with_features(model, s, dt)
                err_label = torch.sqrt(((pred - tgt) ** 2).sum(dim=1, keepdim=True) + 1e-12)
            err_pred = head(feats)
            loss = F.l1_loss(err_pred, err_label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
        if epoch % 2 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch:3d}  L1={total/nb:.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)


# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_auroc(model, head, split_path: Path, horizons: list,
               n_per_h: int, seed: int = 42):
    out = {}
    rng = np.random.RandomState(seed)
    with h5py.File(split_path, "r") as f:
        N, T = f["states"].shape[:2]
        for h in horizons:
            scores_all = []; labels_all = []
            for _ in range(n_per_h):
                i = int(rng.randint(0, N))
                t0 = int(rng.randint(0, T - h))
                s = torch.from_numpy(np.array(f["states"][i, t0])
                                       ).unsqueeze(0).to(DEVICE)
                tgt = torch.from_numpy(np.array(f["states"][i, t0 + h])
                                         ).unsqueeze(0).to(DEVICE)
                dt = torch.tensor([h * DT_BASE], dtype=torch.float32, device=DEVICE)
                pred, feats = forward_with_features(model, s, dt)
                pred_err = head(feats)
                true_err = torch.sqrt(((pred - tgt) ** 2).sum(dim=1, keepdim=True) + 1e-12)
                te = true_err.flatten().cpu().numpy()
                pe = pred_err.flatten().cpu().numpy()
                thr = float(np.quantile(te, 0.75))
                lbl = (te > thr).astype(int)
                if lbl.sum() == 0 or lbl.sum() == len(lbl):
                    continue
                scores_all.append(pe); labels_all.append(lbl)
            if not scores_all:
                out[h] = float("nan")
                continue
            sc = np.concatenate(scores_all); lb = np.concatenate(labels_all)
            out[h] = float(roc_auc_score(lb, sc))
    return out


def load_step_doubling_auroc():
    out = {}
    base = ROOT_ALL / "ablations" / "auroc_recompute" / "results"
    for split in SPLITS:
        p = base / f"{split}_per_horizon.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        m = j["metrics"]
        out[split] = {int(h): float(m[str(h)]["auroc_q75"]) for h in HORIZONS}
    return out


def main():
    t_total = time.time()
    print(f"[error_head] device={DEVICE}", flush=True)
    print(f"[error_head] N_TRAIN={N_TRAIN_SAMPLES}  N_EVAL_PER_H={N_EVAL_PER_H}  "
          f"epochs={N_EPOCHS}  batch={BATCH_SIZE}", flush=True)
    model = load_model(str(CKPT), device=DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    print("\n=== Stage 1: preload training data ===", flush=True)
    train_path = DATA_DIR / "oregonator_train.h5"
    states, targets, dts = preload_data(train_path, N_TRAIN_SAMPLES, HORIZONS,
                                          seed=0)

    print("\n=== Stage 2: train error head (end-to-end through frozen U-Net) ===",
          flush=True)
    # Probe a single forward to learn the feature channel count
    with torch.no_grad():
        dummy_s = torch.from_numpy(states[:1]).to(DEVICE)
        dummy_dt = torch.from_numpy(dts[:1]).to(DEVICE)
        _, feats0 = forward_with_features(model, dummy_s, dummy_dt)
    in_ch = feats0.shape[1]
    print(f"  feature channels = {in_ch}", flush=True)
    head = ErrorHead(in_ch=in_ch).to(DEVICE)
    print(f"  head params = {sum(p.numel() for p in head.parameters()):,}",
          flush=True)
    train_head_inline(model, head, states, targets, dts,
                       n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    head.eval()

    # Save head weights
    torch.save({"head_state": head.state_dict(), "in_ch": in_ch},
                RESULTS_DIR / "error_head.pt")

    print("\n=== Stage 3: eval AUROC per (split, horizon) ===", flush=True)
    auroc_eh = {}
    for split in SPLITS:
        sp = DATA_DIR / f"oregonator_{split}.h5"
        if not sp.exists():
            print(f"  WARN: missing {sp}")
            continue
        ts = time.time()
        auroc_eh[split] = eval_auroc(model, head, sp, HORIZONS, N_EVAL_PER_H)
        print(f"  {split}: {auroc_eh[split]} ({time.time()-ts:.1f}s)",
              flush=True)

    auroc_sd = load_step_doubling_auroc()

    out = {
        "config": {
            "ckpt": str(CKPT),
            "n_train_samples": N_TRAIN_SAMPLES,
            "n_eval_per_h": N_EVAL_PER_H,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "horizons": HORIZONS,
            "splits": SPLITS,
            "threshold_quantile": 0.75,
        },
        "auroc_error_head": auroc_eh,
        "auroc_step_doubling": auroc_sd,
        "wall_seconds_total": time.time() - t_total,
    }
    out_path = RESULTS_DIR / "auroc_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[error_head] wrote {out_path}  total={out['wall_seconds_total']:.1f}s",
          flush=True)

    print("\n=== Comparison: AUROC@q75 (per-cell) ===", flush=True)
    print(f"{'split':<10} {'h':>4} | {'step-doubling':>14} | {'error head':>11} | "
          f"{'Δ':>7}", flush=True)
    print("-" * 60)
    for split in SPLITS:
        for h in HORIZONS:
            sd = auroc_sd.get(split, {}).get(h, float("nan"))
            eh = auroc_eh.get(split, {}).get(h, float("nan"))
            if np.isnan(sd) or np.isnan(eh):
                d = float("nan")
            else:
                d = eh - sd
            print(f"{split:<10} {h:>4} | {sd:>14.4f} | {eh:>11.4f} | {d:>+7.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
