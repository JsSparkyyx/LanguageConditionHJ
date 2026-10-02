#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=08:00:00
#SBATCH --job-name=f5-vusband
#SBATCH -o /work/nvme/bifq/project/tmp/logs/f5visband_%A.out

# F5 fix (P4-T1 visibility): compute the per-embodiment visible table band
# (visibility_band_*.npz) then re-measure T1 visibility with the band active
# (--validate, 20 seeds x 5 hosts x 2 embodiments, setup-only).

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c 'cd /work/nvme/bifq/project/LanguageConditionHJ && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python envs/scripts/item_envs/visibility_band.py \
      --embodiment both && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python envs/scripts/item_envs/visibility_band.py \
      --embodiment both --validate'
