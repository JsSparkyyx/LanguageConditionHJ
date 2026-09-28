#!/bin/bash
# Diagonal set remainder: aloha stack_bowls (t30, flaky replays — many trials).
#   sbatch scripts/slurm_diag_aloha_bowls.sbatch
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=48g --time=03:00:00
#SBATCH --constraint="work"
#SBATCH --job-name=cehj-diag-aloha
#SBATCH -o /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.out
#SBATCH -e /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.err

set -uo pipefail
TMP=/work/nvme/bifq/project/tmp

echo "[job] $(hostname) job $SLURM_JOB_ID diagonal aloha stack_bowls"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif bash -lc "
source /work/nvme/bifq/conda/activate_RoboTwin.sh
export TMP=$TMP TMPDIR=$TMP/tmp
export HOLOBRAIN_CKPT=$TMP/holobrain/HoloBrain_v0.0_GD
export HF_HOME=$TMP/hf-cache XDG_CACHE_HOME=$TMP/xdg-cache
export TRITON_CACHE_DIR=$TMP/triton-cache TORCH_EXTENSIONS_DIR=$TMP/torch-ext
export WANDB_MODE=offline
cd /work/nvme/bifq/project/CEHJ_safe
python tests/replace_extra_video.py --emb aloha-agilex --task stack_bowls_two \
  --budget 30 --max-trials 14 \
  --exclude-json \$TMP/cehj/videos_presentation/selections_used.json,\$TMP/cehj/videos_presentation/stackbowls_loo_franka.json \
  --selections-out \$TMP/cehj/videos_presentation/diagonal_loo_franka.json \
  --out \$TMP/cehj/videos_presentation || true
echo DIAG-ALOHA-EXIT-OK
"
