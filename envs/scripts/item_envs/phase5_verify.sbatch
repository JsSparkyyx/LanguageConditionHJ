#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=06:00:00
#SBATCH --job-name=p5-verify
#SBATCH -o /work/nvme/bifq/project/tmp/logs/p5verify_%A.out

# Phase 5 GPU verification (PLAN v3 §6 Phase 5):
# - test_phase5_regression.py: Phase-0 h traces must be byte-identical
#   (place_empty_cup seed 0, ur5-wsg 143 ticks min_h=-0.0277 m;
#   franka-panda 149 ticks min_h=-0.0388 m).
# - test_phase5_dryrun.py: 3 nominal episodes per item task x embodiment
#   through the real collect wiring; sidecar margins == trace.npz exactly.

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c 'cd /work/nvme/bifq/project/LanguageConditionHJ/envs && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python -m pytest \
      tests/item_envs/test_phase5_regression.py tests/item_envs/test_phase5_dryrun.py \
      -m gpu -q --tb=short'
