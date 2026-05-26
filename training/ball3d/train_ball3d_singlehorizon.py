#!/usr/bin/env python3
"""Single-horizon variant of Ball 3D training: same recipe, but
HORIZONS = [64] only. Used for the single-horizon ablation."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import train_ball3d
train_ball3d.HORIZONS = [64]    # monkey-patch

# override default output dir before main() reads CLI args
import sys as _sys
if "--output_dir" not in _sys.argv:
    _sys.argv.extend([
        "--output_dir",
        str(HERE.parent / "checkpoints" / "shortcut_ball3d_singlehorizon"),
    ])
if "--epochs" not in _sys.argv:
    _sys.argv.extend(["--epochs", "40"])

train_ball3d.main()
