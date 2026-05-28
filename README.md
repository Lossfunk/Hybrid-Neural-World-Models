# Hybrid Neural World Models 

This work introduces a multi-horizon shortcut surrogate that learns to leap across
multiple time scales in a single network forward pass, paired with a label-free
step-doubling error map that detects per-cell extrapolation failure at
inference time. We evaluate on three physical systems: a 2D Oregonator
reaction-diffusion PDE, the 2D compressible Euler equations, and a 3D rigid-body
ball-in-box ODE.

Paper on Arxiv -[ arxiv.org/abs/2605.28317](https://arxiv.org/abs/2605.28317)

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10. PyTorch >= 2.0 with CUDA is required for training but
CPU inference is supported for the figure / ablation scripts.

## Folder layout

```
neruips_final_code/
├── README.md                      this file
├── requirements.txt               minimal pip requirements
├── data_generation/
│   ├── oregonator/                Oregonator solver + dataset generator
│   ├── euler2d/                   2D Euler v2 solver + dataset generator + OOD generator
│   └── ball3d/                    3D ball ODE env + dataset generator + MuJoCo XML
├── models/                        model architectures (one per environment)
│   ├── shortcut_oregonator_2d.py  periodic-BC U-Net + FiLM(dt)
│   ├── shortcut_pde_2d.py         zero-pad U-Net + FiLM(dt) for Euler 2D
│   └── shortcut_ball3d.py         residual MLP + FiLM(dt) for the 9-dim ball state
├── training/                      one trainer per environment
│   ├── oregonator/                trainer + dataset adapter + config
│   ├── euler2d/                   trainer + dataset adapter + config
│   └── ball3d/                    trainer + config
├── evaluation/
│   ├── ablations/                 the eleven ablation experiments (code only)
│   └── figures/                   paper figure / table builders
└── checkpoints/                   one trained `best.pt` per environment, seed 0
```

## Quick start

The supplementary checkpoints are seed-0 models, ready to evaluate. Datasets
are not included due to size — they can be regenerated from `data_generation/`.

### 1. Regenerate a dataset (optional)

```bash
# 2D Oregonator (≈ 30 GB for 400 train trajectories at fp32)
python data_generation/oregonator/generate_dataset.py --split train --n 400 \
    --workers 4
python data_generation/oregonator/convert_to_memmap.py --n_trajs 400 --dtype float16

# 2D Euler v2 (≈ 13 GB for 500 train trajectories at fp32)
python data_generation/euler2d/generate_euler2d_v2_streaming.py --split train --n 500
python data_generation/euler2d/generate_euler2d_v2_ood.py --variant near
python data_generation/euler2d/generate_euler2d_v2_ood.py --variant far

# 3D ball
python data_generation/ball3d/generate_ball3d.py --n_train 1000 --n_val 200 \
    --n_test 200 --n_ood_near 200 --n_ood_far 200
```

### 2. Train a model (optional — checkpoints provided)

```bash
# Oregonator
python training/oregonator/train_shortcut_oregonator.py \
    --config training/oregonator/config.yaml

# Euler 2D
python training/euler2d/train_shortcut_2d_dagger.py \
    --config training/euler2d/config.yaml

# Ball 3D
python training/ball3d/train_ball3d.py
```

### 3. Reproduce paper figures

The figure builders read precomputed evaluation results from
`evaluation/ablations/*/results/`. Those folders are intentionally empty in
the code (results are derived from the ablation code in this directory
plus the trained checkpoints).

```bash
# After running the ablations, regenerate figure 2 (Oregonator)
python evaluation/figures/build_fig2_oregonator.py
```

### 4. Run an ablation

Each subdirectory under `evaluation/ablations/` contains a self-contained
experiment. Example:

```bash
python evaluation/ablations/error_head/train_and_eval.py            # Oregonator
python evaluation/ablations/error_head/train_and_eval_euler.py      # Euler 2D
python evaluation/ablations/error_head/train_and_eval_ball3d.py     # Ball 3D
```

## Notes on this Code

- All datasets and result artefacts (figures, intermediate result JSON,
  cached evaluation outputs) are excluded. Only source code, configurations,
  and trained checkpoints are included.
- The shared eval helpers used by ablation scripts (model loading, AUROC
  computation, metrics) live under `evaluation/oregonator_eval/` and are
  imported via `sys.path` from each ablation script.
- Random seeds are fixed in every script; the canonical seed is 0. Seeds


@article{lakshmanan2026hybrid,
  title={Hybrid Neural World Models},
  author={Lakshmanan, Pranav and Chopra, Paras},
  journal={arXiv preprint arXiv:2605.28317},
  year={2026}
}
  1 and 2 are used for the cross-seed appendix runs; the corresponding
  checkpoints are not included to keep the supplement compact, but each
  trainer accepts a `--seed` argument that reproduces them.
