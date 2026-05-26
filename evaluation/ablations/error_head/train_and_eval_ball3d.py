#!/usr/bin/env python3
"""Learned error head for Ball 3D — small MLP that predicts scalar
trajectory error from (state, dt) directly, trained against true error
of the frozen surrogate."""
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
sys.path.insert(0, str(ROOT / "training" / "ball3d"))
from shortcut_ball3d import ShortcutBall3D, sinusoidal_dt_embedding   # noqa: E402

CKPT = ROOT / "checkpoints" / "ball3d" / "best.pt"
DATA_DIR = ROOT / "data" / "ball3d"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = ["test", "ood_near", "ood_far"]
HORIZONS = [16, 32, 64]
DT_BASE = 0.01
N_TRAIN_SAMPLES = 4000
N_EVAL_PER_H = 80
N_EPOCHS = 30
BATCH_SIZE = 64


def load_model():
    ck = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    m = ShortcutBall3D(state_dim=9, hidden_dim=cfg["hidden_dim"],
                          emb_dim=cfg["emb_dim"], n_blocks=cfg["n_blocks"],
                          ch_mean=ck["ch_mean"], ch_std=ck["ch_std"]).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


class ErrorHeadMLP(nn.Module):
    """MLP that takes (state, dt_embedding) and outputs scalar error magnitude."""
    def __init__(self, state_dim=9, emb_dim=64, hidden=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, dt):
        emb = sinusoidal_dt_embedding(dt, self.emb_dim)
        x = torch.cat([state, emb], dim=-1)
        return F.softplus(self.net(x)).squeeze(-1)


def main():
    t_total = time.time()
    print(f"[error_head_ball3d] device={DEVICE}", flush=True)
    model = load_model()

    print(f"\n=== Stage 1: collect training pairs ({N_TRAIN_SAMPLES}) ===", flush=True)
    with h5py.File(DATA_DIR / "ball3d_train.h5", "r") as f:
        states_h5 = np.array(f["states"], dtype=np.float32)
    N_traj, T_traj = states_h5.shape[:2]
    rng = np.random.RandomState(0)
    s_list, tgt_list, dt_list = [], [], []
    for _ in range(N_TRAIN_SAMPLES):
        i = int(rng.randint(0, N_traj))
        h = int(HORIZONS[rng.randint(len(HORIZONS))])
        t0 = int(rng.randint(0, T_traj - h))
        s_list.append(states_h5[i, t0])
        tgt_list.append(states_h5[i, t0 + h])
        dt_list.append(h * DT_BASE)
    states_np = np.stack(s_list).astype(np.float32)
    targets_np = np.stack(tgt_list).astype(np.float32)
    dts_np = np.array(dt_list, dtype=np.float32)

    print(f"  collected {len(states_np)} pairs", flush=True)

    head = ErrorHeadMLP(state_dim=9, emb_dim=64, hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    states_t = torch.from_numpy(states_np)
    targets_t = torch.from_numpy(targets_np)
    dts_t = torch.from_numpy(dts_np)
    n = len(states_t)

    print(f"\n=== Stage 2: train error head ===", flush=True)
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
                pred = model(s, dt)
                err_label = torch.sqrt(((pred[:, :3] - tgt[:, :3]) ** 2).sum(dim=1) + 1e-12)
            err_pred = head(s, dt)
            loss = F.l1_loss(err_pred, err_label)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"    epoch {epoch:3d}  L1={total/nb:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    head.eval()

    print(f"\n=== Stage 3: eval AUROC per (split, horizon) ===", flush=True)
    auroc_eh = {}
    for split in SPLITS:
        with h5py.File(DATA_DIR / f"ball3d_{split}.h5", "r") as f:
            states_split = np.array(f["states"], dtype=np.float32)
        N_s, T_s = states_split.shape[:2]
        rng = np.random.RandomState(42)
        out = {}
        for h in HORIZONS:
            scores, labels = [], []
            for _ in range(N_EVAL_PER_H):
                i = int(rng.randint(0, N_s))
                t0 = int(rng.randint(0, T_s - h))
                s = torch.from_numpy(states_split[i, t0]).unsqueeze(0).to(DEVICE)
                tgt = torch.from_numpy(states_split[i, t0 + h]).unsqueeze(0).to(DEVICE)
                dt = torch.tensor([h * DT_BASE], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    pred = model(s, dt)
                    pred_err = head(s, dt)
                true_err = torch.sqrt(((pred[:, :3] - tgt[:, :3]) ** 2).sum(dim=1) + 1e-12)
                scores.append(float(pred_err.cpu().item()))
                labels.append(float(true_err.cpu().item()))
            sc = np.array(scores); lb = np.array(labels)
            thr = float(np.quantile(lb, 0.75))
            bin_lbl = (lb > thr).astype(int)
            if bin_lbl.sum() == 0 or bin_lbl.sum() == len(bin_lbl):
                out[h] = float("nan")
            else:
                out[h] = float(roc_auc_score(bin_lbl, sc))
            print(f"  {split} h={h:2d}  AUROC_EH={out[h]:.4f}", flush=True)
        auroc_eh[split] = out

    res = {
        "config": {"ckpt": str(CKPT), "n_train": N_TRAIN_SAMPLES,
                     "n_eval_per_h": N_EVAL_PER_H, "n_epochs": N_EPOCHS,
                     "horizons": HORIZONS, "splits": SPLITS},
        "auroc_error_head": auroc_eh,
        "wall_seconds": time.time() - t_total,
    }
    out_path = RESULTS / "auroc_compare_ball3d.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\n[error_head_ball3d] wrote {out_path}  total={res['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
