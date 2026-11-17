#!/usr/bin/env python3
"""Fig 1: new panel (a) + submitted panel (b), pixel-exact.

panel (a) comes from the re-rendered figure (build_fig1_v2.py, measured
cpu_horizon_sweep.json). panel (b) is copied verbatim from the submitted PNG,
because the q-sweep JSONs panel (b) reads were rewritten after the submitted
figure was made (re-rendering would silently change panel (b)).

The split column is the centre of the widest all-white column gap between the
two panels, required to be white in BOTH images. Aborts if image sizes differ.
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
NEW = OUT / "fig1_unspliced" / "fig1_speedup_two_panel.png"
ORIG = HERE.parents[1] / "neurips_FINAL_FIGS" / "fig1_speedup_two_panel.png"

new_im, orig_im = Image.open(NEW), Image.open(ORIG)
new, orig = np.asarray(new_im.convert("RGBA")), np.asarray(orig_im.convert("RGBA"))
if new.shape != orig.shape:
    raise SystemExit(f"size mismatch: new {new.shape} vs submitted {orig.shape}; cannot splice")

W = new.shape[1]
white = lambda a: (a[:, :, :3] >= 250).all(axis=(0, 2))
both_white = white(new) & white(orig)
lo, hi = int(W * 0.35), int(W * 0.65)          # search the middle band only
best, run_start, best_span = None, None, (0, 0)
for x in range(lo, hi + 1):
    if x < hi and both_white[x]:
        run_start = x if run_start is None else run_start
    elif run_start is not None:
        if x - run_start > best_span[1] - best_span[0]:
            best_span = (run_start, x)
        run_start = None
if best_span[1] - best_span[0] < 5:
    raise SystemExit("no clean white gap between panels; cannot splice")
split = (best_span[0] + best_span[1]) // 2

out = np.concatenate([new[:, :split], orig[:, split:]], axis=1)
Image.fromarray(out).save(OUT / "fig1_speedup_two_panel.png", dpi=orig_im.info.get("dpi", (200, 200)))
diff_left = int((np.abs(new[:, :split].astype(int) - orig[:, :split].astype(int)).sum(2) > 0).sum())
print(f"white gap columns {best_span}, split at x={split} of {W}; "
      f"panel (a) pixels changed vs submitted: {diff_left}; panel (b) copied verbatim")
