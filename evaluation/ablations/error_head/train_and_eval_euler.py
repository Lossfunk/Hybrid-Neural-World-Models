#!/usr/bin/env python3
"""Learned error head for Euler 2D — same approach as Oregonator script."""
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
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "training" / "euler2d"))
sys.path.insert(0, str(ROOT / "models"))

from eval_utils_euler import load_model    # noqa: E402
from data_utils_2d import Euler2DDataset    # noqa: E402

CKPT = ROOT / "checkpoints" / "euler2d" / "best.pt"
DATA_DIR = ROOT / "data" / "euler2d"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = ["test", "ood_near", "ood_far"]
HORIZONS = [16, 32, 64]
N_TRAIN_SAMPLES = 1500
N_EVAL_PER_H = 80
N_EPOCHS = 8
BATCH_SIZE = 16


def forward_with_features(model, u, dt):
    """Tap the U-Net's last decoder block features."""
    from shortcut_oregonator_2d import sinusoidal_dt_embedding
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


class ErrorHead(nn.Module):
    def __init__(self, in_ch, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        return F.softplus(self.net(x))


def main():
    t_total = time.time()
    print(f"[error_head_euler] device={DEVICE}", flush=True)
    sys.path.insert(0, str(ROOT / "models"))
    model = load_model(str(CKPT), device=DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    train_ds = Euler2DDataset(str(DATA_DIR / "euler2d_v2_train.h5"))

    print(f"\n=== Stage 1: collect training pairs ({N_TRAIN_SAMPLES}) ===", flush=True)
    rng = np.random.RandomState(0)
    states_list, targets_list, dts_list = [], [], []
    for k in range(N_TRAIN_SAMPLES):
        i = int(rng.randint(0, train_ds.N))
        h = int(HORIZONS[rng.randint(len(HORIZONS))])
        t0 = int(rng.randint(0, train_ds.T - h))
        states_list.append(train_ds.frame(i, t0).numpy())
        targets_list.append(train_ds.frame(i, t0 + h).numpy())
        dts_list.append(h * train_ds.dt)
    states_np = np.stack(states_list)
    targets_np = np.stack(targets_list)
    dts_np = np.array(dts_list, dtype=np.float32)
    print(f"  collected {len(states_np)} pairs, shape {states_np.shape}", flush=True)

    print(f"\n=== Stage 2: train error head ===", flush=True)
    with torch.no_grad():
        s = torch.from_numpy(states_np[:1]).to(DEVICE)
        d = torch.from_numpy(dts_np[:1]).to(DEVICE)
        _, feats0 = forward_with_features(model, s, d)
    in_ch = feats0.shape[1]
    print(f"  feature channels = {in_ch}", flush=True)
    head = ErrorHead(in_ch=in_ch).to(DEVICE)

    states_t = torch.from_numpy(states_np)
    targets_t = torch.from_numpy(targets_np)
    dts_t = torch.from_numpy(dts_np)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(states_np)
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        total = 0.0; nb = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            s = states_t[idx].to(DEVICE)
            tgt = targets_t[idx].to(DEVICE)
            dt = dts_t[idx].to(DEVICE)
            with torch.no_grad():
                pred, feats = forward_with_features(model, s, dt)
                err_label = torch.sqrt(((pred - tgt) ** 2).sum(dim=1, keepdim=True) + 1e-12)
            err_pred = head(feats)
            loss = F.l1_loss(err_pred, err_label)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        print(f"    epoch {epoch:2d}  L1={total/nb:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    head.eval()

    print(f"\n=== Stage 3: eval AUROC per (split, horizon) ===", flush=True)
    auroc_eh = {}
    for split in SPLITS:
        ds = Euler2DDataset(str(DATA_DIR / f"euler2d_v2_{split}.h5"))
        rng = np.random.RandomState(42)
        out = {}
        for h in HORIZONS:
            scores_all, labels_all = [], []
            for _ in range(N_EVAL_PER_H):
                i = int(rng.randint(0, ds.N))
                t0 = int(rng.randint(0, ds.T - h))
                s = ds.frame(i, t0).to(DEVICE).unsqueeze(0)
                tgt = ds.frame(i, t0 + h).to(DEVICE).unsqueeze(0)
                dt = torch.tensor([h * ds.dt], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    pred, feats = forward_with_features(model, s, dt)
                    pred_err = head(feats)
                te = torch.sqrt(((pred - tgt) ** 2).sum(dim=1, keepdim=True) + 1e-12)
                te_flat = te.flatten().cpu().numpy()
                pe_flat = pred_err.flatten().cpu().numpy()
                thr = float(np.quantile(te_flat, 0.75))
                lbl = (te_flat > thr).astype(int)
                if lbl.sum() == 0 or lbl.sum() == len(lbl): continue
                scores_all.append(pe_flat); labels_all.append(lbl)
            if not scores_all:
                out[h] = float("nan"); continue
            sc = np.concatenate(scores_all); lb = np.concatenate(labels_all)
            out[h] = float(roc_auc_score(lb, sc))
            print(f"  {split} h={h:2d}  AUROC_EH={out[h]:.4f}", flush=True)
        auroc_eh[split] = out

    out = {
        "config": {
            "ckpt": str(CKPT), "n_train": N_TRAIN_SAMPLES,
            "n_eval_per_h": N_EVAL_PER_H, "n_epochs": N_EPOCHS,
            "horizons": HORIZONS, "splits": SPLITS,
        },
        "auroc_error_head": auroc_eh,
        "wall_seconds": time.time() - t_total,
    }
    out_path = RESULTS / "auroc_compare_euler.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[error_head_euler] wrote {out_path}  total={out['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
