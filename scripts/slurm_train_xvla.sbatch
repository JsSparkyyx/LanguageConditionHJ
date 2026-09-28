#!/bin/bash
# Standard round-based HJ-SAC training on Delta. One job = one run.
# Usage:
#   sbatch --job-name=<run> scripts/slurm_train_standard.sbatch \
#       <run_name> [--leave-out X | --only-embodiment X] [--vanilla] [...]
#
# The training memmap is created empty on NODE-LOCAL /tmp and seeded from the
# shared warmup buffer via --init-buffer. Round collection appends to /tmp;
# checkpoints/metrics/videos persist under $TMP/cehj/{data,runs}/<run_name>.
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=48g --time=48:00:00
#SBATCH --constraint="work"
#SBATCH --job-name=cehj-train
#SBATCH -o /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.out
#SBATCH -e /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.err

set -euo pipefail
RUN_NAME="${1:?usage: sbatch ... slurm_train_standard.sbatch <run_name> [train.py flags]}"
shift

TMP=/work/nvme/bifq/project/tmp
WARMUP_BUF="${WARMUP_BUF:-$TMP/cehj/warmup500xvla/buffer}"
LOCAL=/tmp/cehj_${SLURM_JOB_ID}

# 350k-step ring claims ~226 GB; refuse to start on a node with less.
avail_kb=$(df --output=avail /tmp | tail -1 | tr -d ' ')
if [ "$avail_kb" -lt $((240 * 1024 * 1024)) ]; then
  echo "[job] /tmp has only $((avail_kb / 1024 / 1024)) GB free; need ~240 GB" >&2
  exit 1
fi

cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT

mkdir -p "$LOCAL"

echo "[job] $(hostname) job $SLURM_JOB_ID run=$RUN_NAME flags=$*"
echo "[job] training buffer: $LOCAL/buffer; seeding from $WARMUP_BUF"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif bash -lc "
set -e
source /work/nvme/bifq/conda/activate_RoboTwin.sh
export TMP=$TMP TMPDIR=$TMP/tmp
export HOLOBRAIN_CKPT=$TMP/holobrain/HoloBrain_v0.0_GD
export HF_HOME=$TMP/hf-cache XDG_CACHE_HOME=$TMP/xdg-cache
export TRITON_CACHE_DIR=$TMP/triton-cache TORCH_EXTENSIONS_DIR=$TMP/torch-ext
export WANDB_MODE=\${WANDB_MODE:-online} WANDB__SERVICE_WAIT=300
cd /work/nvme/bifq/project/CEHJ_safe/CEHJ_XVLA
python main/train/train.py \
  --capacity 300000 \
  --init-buffer $WARMUP_BUF \
  --buffer-dir $LOCAL/buffer \
  --collect-rounds 300 --episodes-per-round 3 \
  --grad-steps-per-round 1024  \
  --data $TMP/cehj/data/$RUN_NAME \
  --run  $TMP/cehj/runs/$RUN_NAME \
  --cross-eval-every-epochs 8 \
  --filter-release-margin 0.005 \
  $*
"
