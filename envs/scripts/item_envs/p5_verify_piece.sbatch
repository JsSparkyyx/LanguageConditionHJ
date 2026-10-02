#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=00:55:00
#SBATCH --job-name=p5v-piece
#SBATCH -o /work/nvme/bifq/project/tmp/logs/p5vpiece_%A.out

# Interactive piece-runner for phase5_verify.sbatch's contents (the batch
# job runs both test files; the interactive 1 h cap needs them in pieces).
#   sbatch envs/scripts/item_envs/p5_verify_piece.sbatch
#     -> regression only (default)
#   sbatch --export=ALL,PYTEST_FILES=tests/item_envs/test_phase5_dryrun.py,PYTEST_EXTRA="-k ur5" \
#     envs/scripts/item_envs/p5_verify_piece.sbatch
#     -> dryrun, one embodiment per job

: "${PYTEST_FILES:=tests/item_envs/test_phase5_regression.py}"
: "${PYTEST_EXTRA:=}"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c "cd /work/nvme/bifq/project/LanguageConditionHJ/envs && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python -m pytest \
      $PYTEST_FILES -m gpu -q --tb=short $PYTEST_EXTRA"
