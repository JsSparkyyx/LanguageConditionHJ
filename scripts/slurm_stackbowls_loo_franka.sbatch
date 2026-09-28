#!/bin/bash
# stack_bowls_two across all 5 embodiments from loo_franka (presentation set).
#   sbatch scripts/slurm_stackbowls_loo_franka.sbatch
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=48g --time=06:00:00
#SBATCH --constraint="work"
#SBATCH --job-name=cehj-sb-lf
#SBATCH -o /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.out
#SBATCH -e /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.err

set -uo pipefail
TMP=/work/nvme/bifq/project/tmp

echo "[job] $(hostname) job $SLURM_JOB_ID stack_bowls loo_franka all-emb"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif bash -lc "
source /work/nvme/bifq/conda/activate_RoboTwin.sh
export TMP=$TMP TMPDIR=$TMP/tmp
export HOLOBRAIN_CKPT=$TMP/holobrain/HoloBrain_v0.0_GD
export HF_HOME=$TMP/hf-cache XDG_CACHE_HOME=$TMP/xdg-cache
export TRITON_CACHE_DIR=$TMP/triton-cache TORCH_EXTENSIONS_DIR=$TMP/torch-ext
export WANDB_MODE=offline
cd /work/nvme/bifq/project/CEHJ_safe
for emb in franka-panda ARX-X5 ur5-wsg piper; do
  python tests/replace_extra_video.py --emb \$emb \
    --task stack_bowls_two --budget 18 --max-trials 8 \
    --exclude-json \$TMP/cehj/videos_presentation/selections_used.json \
    --selections-out \$TMP/cehj/videos_presentation/stackbowls_loo_franka.json \
    --out \$TMP/cehj/videos_presentation || true
done
python tests/replace_extra_video.py --emb aloha-agilex \
  --task stack_bowls_two --budget 30 --max-trials 8 \
  --exclude-json \$TMP/cehj/videos_presentation/selections_used.json \
  --selections-out \$TMP/cehj/videos_presentation/stackbowls_loo_franka.json \
  --out \$TMP/cehj/videos_presentation || true
echo STACKBOWLS-EXIT-OK
"
