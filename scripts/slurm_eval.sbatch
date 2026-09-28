#!/bin/bash
# Full checkpoint eval on Delta. One job = one (run, ckpt, margin, shard).
# Usage:
#   sbatch --job-name=<name> scripts/slurm_eval.sbatch \
#       <eval_ckpt.py path> --run <run_dir> [--ckpt X | --nominal] \
#       [--embodiments piper] [--episodes 50] [--filter-release-margin 0.005]
# Normally launched by scripts/eval_all.py.
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=48g --time=48:00:00
#SBATCH --constraint="work"
#SBATCH --job-name=cehj-eval
#SBATCH -o /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.out
#SBATCH -e /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.err

set -euo pipefail
EVAL_SCRIPT="${1:?usage: sbatch ... slurm_eval.sbatch <eval_ckpt.py> [args]}"
shift

TMP=/work/nvme/bifq/project/tmp

echo "[job] $(hostname) job $SLURM_JOB_ID eval: $EVAL_SCRIPT $*"

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
python $EVAL_SCRIPT $*
"
