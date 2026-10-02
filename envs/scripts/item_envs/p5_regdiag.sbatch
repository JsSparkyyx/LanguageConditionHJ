#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=02:00:00
#SBATCH --job-name=p5-regdiag
#SBATCH -o /work/nvme/bifq/project/tmp/logs/p5regdiag_%A.out

# Regression bisect: run the SAME Phase-5 regression test against a worktree
# at 14c6180 (pre-Phase-5 rollout.py/collect.py). 143 ticks -> Phase-5 broke
# the default path; 142 ticks -> test/reference reconstruction issue.

set -e
WT=/work/nvme/bifq/project/tmp/wt_pre5_${SLURM_JOB_ID}
cd /work/nvme/bifq/project/LanguageConditionHJ
git worktree add --detach "$WT" 14c6180
trap 'git worktree remove --force "$WT" || true' EXIT
mkdir -p "$WT/envs/tests/item_envs"
git show main:envs/tests/item_envs/test_phase5_regression.py > "$WT/envs/tests/item_envs/test_phase5_regression.py"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c "cd $WT/envs && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python -m pytest \
      tests/item_envs/test_phase5_regression.py -m gpu -q --tb=long"
