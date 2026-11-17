# Camera-ready scripts (NeurIPS 2026)

Scripts that produced the camera-ready tables, numbers and updated figures for
*Hybrid Neural World Models for Physical Dynamics* (NeurIPS 2026, Main Track,
[OpenReview](https://openreview.net/forum?id=uWsmvC2jq5)).

None of these scripts runs a simulation, training job, solver rollout or
evaluation. Each one reads existing result files (JSON results, training
histories, checkpoints) and only tabulates or plots them.

## Path layout

Every script locates its inputs relative to a workspace root `ROOT`, taken as
two levels above the script (`ROOT/camera_ready_out/scripts/<script>.py`), and
writes its outputs to `ROOT/camera_ready_out/`. Input paths under `ROOT` follow
the research workspace layout:

| Workspace path (under `ROOT`)            | In this repository                       |
|------------------------------------------|------------------------------------------|
| `ablations/<name>/...`                   | `evaluation/ablations/<name>/...`        |
| `oregonator_pipeline/checkpoints/...`    | Oregonator checkpoints / training runs   |
| `neurips/training/checkpoints/...`       | Euler 2D checkpoints / training runs     |
| `mujoco_pipeline/checkpoints/...`        | Ball 3D checkpoints / training runs      |

Result JSONs are not shipped (see the top-level README); produce them with the
ablation code in `evaluation/ablations/` first.

## Scripts

| Script | Output | Used in camera-ready |
|---|---|---|
| `compute_tables_and_numbers.py` | `table5.tex`, `table6.tex` (54-cell Mode 2 breakdown), `rmse_table.tex`, `numbers.md` | yes |
| `build_fig6_mode2.py` | `fig6_mode2.png/.pdf` (Mode 2 at h=64, q=0.75, from the quadrant results) | yes |
| `build_appendix_cross_seed.py` | `appendix_cross_seed.png/.pdf` (independent-draw cross-seed AUROC) | yes |
| `build_appendix_dagger.py` | `appendix_dagger.png/.pdf` (DAgger-weight ablation, legend/colour fix) | yes |
| `build_appendix_m1_vs_m2.py` | `appendix_m1_vs_m2.png/.pdf` (Mode 1 vs Mode 2 across horizons) | yes |
| `make_patched_scripts.py` | regenerates the five `build_*.py` scripts above from the original submission figure builders by exact-string patches (data source and output path only) | tooling |
| `make_text_fixes.py` | `text_fixes.md`, a find/replace list for the paper source; needs the paper `.tex`, which is not in this repo | tooling |
| `build_fig1_v2.py`, `splice_fig1.py` | an alternative Fig. 1 panel (a) from the per-horizon CPU sweep | no, the submitted Fig. 1 is kept |
| `build_appendix_richardson.py` | a re-titled Richardson appendix figure | no, the submitted figure is kept |

Requirements: `numpy`, `matplotlib`, `torch` (to read checkpoint metadata),
`Pillow` (for `splice_fig1.py`).
