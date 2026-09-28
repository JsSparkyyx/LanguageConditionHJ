#!/bin/bash
# Presentation video recording (tests/record_presentation.py) on Delta.
#   sbatch --job-name=cehj-pres scripts/slurm_record_presentation.sbatch
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=48g --time=08:00:00
#SBATCH --constraint="work"
#SBATCH --job-name=cehj-pres
#SBATCH -o /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.out
#SBATCH -e /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.err

set -euo pipefail
TMP=/work/nvme/bifq/project/tmp

echo "[job] $(hostname) job $SLURM_JOB_ID record_presentation"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif bash -lc "
set -e
source /work/nvme/bifq/conda/activate_RoboTwin.sh
export TMP=$TMP TMPDIR=$TMP/tmp
export HOLOBRAIN_CKPT=$TMP/holobrain/HoloBrain_v0.0_GD
export HF_HOME=$TMP/hf-cache XDG_CACHE_HOME=$TMP/xdg-cache
export TRITON_CACHE_DIR=$TMP/triton-cache TORCH_EXTENSIONS_DIR=$TMP/torch-ext
export WANDB_MODE=offline
cd /work/nvme/bifq/project/CEHJ_safe
python tests/record_presentation.py \
  --candidates \$TMP/cehj/presentation_candidates.json \
  --out \$TMP/cehj/videos_presentation
"
