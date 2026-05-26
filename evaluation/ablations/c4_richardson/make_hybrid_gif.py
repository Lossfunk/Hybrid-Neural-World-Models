#!/usr/bin/env python3
"""Render a GIF showing the adaptive horizon descent in action.

For a chosen test trajectory:
  - top panel:    solver-only ground truth u(x,y) at each macro-step
  - bottom panel: hybrid-ours u(x,y) at each macro-step
  - title:        current sim time, last horizon used, decision (surrogate/solver),
                  cumulative speedup-so-far
  - colored border on bottom panel: green=big surrogate jump (h≥16),
                                     yellow=medium (h=4,8), orange=small (h=2),
                                     red=solver fallback

Each macro-step in hybrid is one frame. Solver-only frames are aligned to
those macro boundaries. Annotations show "leaping ahead" by big-h jumps.

Args: --traj <test_idx>  --out <name>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "oregonator_eval"))
sys.path.insert(0, str(ROOT / "data_generation" / "oregonator"))
sys.path.insert(0, str(ROOT / "models"))

from eval_utils import load_model    # noqa: E402
from oregonator2d_tyson import OregonatorTyson2D, TysonParams    # noqa: E402

CKPT = ROOT / "checkpoints" / "oregonator" / "best.pt"
DATA_DIR = ROOT / "data" / "oregonator"
GIF_DIR = HERE / "gifs"
GIF_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_DESC = [64, 32, 16, 8, 4, 2, 1]


def horizon_color(h):
    if h is None:
        return "#777777"
    if h >= 32: return "#1a9850"      # dark green — big jump
    if h >= 16: return "#66bd63"      # mid green
    if h >= 8:  return "#fdae61"      # orange
    if h >= 4:  return "#f46d43"      # red-orange
    if h >= 2:  return "#d73027"      # red — small step
    return "#5c5c5c"                  # solver fallback gray


@torch.no_grad()
def hybrid_with_capture(model, state0_np, total_steps, params, dt_save,
                          tau, device):
    """Run hybrid-ours, capturing the full state at every macro-step boundary.
    Returns dict with per-macro state, horizon, time, surrogate-vs-solver."""
    state = torch.from_numpy(state0_np.copy()).to(device)
    steps_remaining = total_steps
    states_per_macro = [state.cpu().numpy().copy()]
    decisions = []   # list of (horizon, "surrogate"/"solver", e_hat, wall_after_macro)
    t_start = time.perf_counter()
    while steps_remaining > 0:
        chosen = None
        for h in HORIZONS_DESC:
            if h > steps_remaining: continue
            dt_t = torch.tensor([h * dt_save], device=device, dtype=torch.float32)
            pred_full = model(state.unsqueeze(0), dt_t)[0]
            if h >= 2:
                pred_mid = model(state.unsqueeze(0), dt_t * 0.5)[0]
                pred_chain = model(pred_mid.unsqueeze(0), dt_t * 0.5)[0]
                e_hat = float(torch.sqrt(((pred_full - pred_chain) ** 2).sum(dim=0)).mean().item())
            else:
                e_hat = 0.0
            if e_hat < tau:
                chosen = (h, pred_full, e_hat); break
        if chosen is not None:
            h, pred_full, e_hat = chosen
            state = pred_full
            steps_remaining -= h
            decisions.append((h, "surrogate", e_hat, time.perf_counter() - t_start))
            states_per_macro.append(state.cpu().numpy().copy())
        else:
            # solver fallback — for visualization, run solver for h=1
            sim = OregonatorTyson2D(n_x=state.shape[2], n_y=state.shape[1],
                                      L_x=100.0, L_y=100.0,
                                      params=TysonParams(**params))
            state_np = state.cpu().numpy()
            sim.u[:] = state_np[0]; sim.v[:] = state_np[1]
            sim.step(dt_save)
            state = torch.from_numpy(np.stack([sim.u, sim.v], axis=0).astype(np.float32)).to(device)
            steps_remaining -= 1
            decisions.append((1, "solver", float("inf"), time.perf_counter() - t_start))
            states_per_macro.append(state.cpu().numpy().copy())
    return dict(states=states_per_macro, decisions=decisions,
                wall_total=time.perf_counter() - t_start)


def solver_with_capture(state0_np, total_steps, params, dt_save):
    """Run solver-only, capturing state at every macro-step boundary."""
    sim = OregonatorTyson2D(n_x=state0_np.shape[2], n_y=state0_np.shape[1],
                              L_x=100.0, L_y=100.0,
                              params=TysonParams(**params))
    sim.u[:] = state0_np[0]; sim.v[:] = state0_np[1]
    states = [state0_np.copy()]
    times = [0.0]
    t_start = time.perf_counter()
    for k in range(total_steps):
        sim.step(dt_save)
        states.append(np.stack([sim.u, sim.v], axis=0).astype(np.float32).copy())
        times.append(time.perf_counter() - t_start)
    return dict(states=states, times=times, wall_total=time.perf_counter() - t_start)


def render_frame(u_solver: np.ndarray, u_hybrid: np.ndarray,
                  step_idx: int, total_steps: int, dt_save: float,
                  decisions: list, last_horizon: int | None,
                  wall_solver: float, wall_hybrid: float) -> Image.Image:
    """Render one frame: 1×2 panel + bottom info bar."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                              gridspec_kw={"width_ratios": [1, 1]})
    # top: solver
    axes[0].imshow(u_solver, origin="lower", cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(f"Solver-only ground truth", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    # bottom: hybrid with colored border indicating last decision
    axes[1].imshow(u_hybrid, origin="lower", cmap="inferno", vmin=0, vmax=1)
    border_color = horizon_color(last_horizon)
    for spine in axes[1].spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(4)
    if last_horizon is not None:
        if last_horizon == 1 and decisions and decisions[-1][1] == "solver":
            label = "solver fallback (h=1)"
        else:
            label = f"surrogate jump h={last_horizon}"
    else:
        label = "init"
    axes[1].set_title(f"Hybrid prediction — {label}", fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # speedup so far
    speedup = wall_solver / max(wall_hybrid, 1e-6) if wall_hybrid > 0 else 0
    sim_time = step_idx * dt_save
    fig.suptitle(
        f"t = {sim_time:.2f} s   "
        f"step {step_idx}/{total_steps}   "
        f"wall: solver={wall_solver:.2f}s  hybrid={wall_hybrid:.2f}s   "
        f"speedup so far = {speedup:.1f}×",
        fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=int, default=0,
                     help="test split traj index")
    ap.add_argument("--total_steps", type=int, default=64)
    ap.add_argument("--out", default="hybrid_traj")
    ap.add_argument("--tau", type=float, default=None,
                     help="threshold; if not given, read from results/run.json")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    # τ from the C4 run (if available)
    tau = args.tau
    if tau is None:
        res = HERE / "results" / "run.json"
        if res.exists():
            import json
            with open(res) as f:
                tau = json.load(f)["config"]["tau_ours"]
            print(f"[gif] using tau_ours from run.json: {tau:.5f}")
        else:
            tau = 0.05
            print(f"[gif] no run.json yet, using fallback tau={tau}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(CKPT), device=device)
    test_path = DATA_DIR / "oregonator_test.h5"
    with h5py.File(test_path, "r") as f:
        state0 = np.array(f["states"][args.traj, 0])
        params = dict(eps=float(f["params"][args.traj, 1]),
                       q=float(f["params"][args.traj, 2]),
                       f=float(f["params"][args.traj, 0]),
                       D=float(f["params"][args.traj, 3]))
        dt_save = float(f.attrs["dt_save"])
    print(f"[gif] traj {args.traj}  params={params}  total_steps={args.total_steps}")

    # Run solver
    print("[gif] running solver...", flush=True)
    sol = solver_with_capture(state0, args.total_steps, params, dt_save)
    print(f"[gif] solver wall: {sol['wall_total']:.2f}s")

    # Run hybrid
    print("[gif] running hybrid-ours...", flush=True)
    hyb = hybrid_with_capture(model, state0, args.total_steps, params, dt_save,
                                tau, device)
    print(f"[gif] hybrid wall: {hyb['wall_total']:.2f}s  "
          f"speedup = {sol['wall_total']/max(hyb['wall_total'],1e-6):.1f}×")
    print(f"[gif] decisions: {len(hyb['decisions'])} macro-steps")
    for i, (h, kind, eh, w) in enumerate(hyb['decisions'][:10]):
        print(f"   #{i:2d}: h={h:>2d} ({kind})  ê={eh:.4f}  wall@step={w:.3f}s")

    # Build alignment between solver step indices and hybrid macro-step state
    # The hybrid took variable-h jumps. We need to render frames at each
    # hybrid macro-step boundary, with solver shown at the corresponding step.
    n_hybrid_macros = len(hyb["decisions"])
    cumulative_steps = 0
    frames = []
    last_h = None
    for k in range(n_hybrid_macros + 1):
        if k == 0:
            sim_step = 0
            last_h = None
            wall_h = 0.0
            wall_s = 0.0
        else:
            h_taken = hyb["decisions"][k - 1][0]
            cumulative_steps += h_taken
            sim_step = cumulative_steps
            last_h = h_taken if hyb["decisions"][k - 1][1] == "surrogate" else 1
            wall_h = hyb["decisions"][k - 1][3]
            wall_s = sol["times"][min(sim_step, len(sol["times"]) - 1)]
        u_sol = sol["states"][min(sim_step, len(sol["states"]) - 1)][0]    # u channel
        u_hyb = hyb["states"][k][0]
        frame = render_frame(u_sol, u_hyb, sim_step, args.total_steps, dt_save,
                              hyb["decisions"][:k], last_h, wall_s, wall_h)
        # Repeat slow frames so big-h jumps don't blink past
        n_repeat = max(1, last_h or 1) // 2
        for _ in range(n_repeat):
            frames.append(frame)

    out_path = GIF_DIR / f"{args.out}_{args.traj:04d}.gif"
    print(f"[gif] writing {len(frames)} frames @ {args.fps} fps -> {out_path}")
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                    duration=int(1000 / args.fps), loop=0)
    print(f"[gif] done  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
