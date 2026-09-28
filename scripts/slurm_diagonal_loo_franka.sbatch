#!/bin/bash
# Diagonal set: 5 tasks x 5 embodiments, each used once, all loo_franka.
# Runs AFTER the stack_bowls job so its picks are excluded.
#   sbatch --dependency=afterany:<stackbowls_job> scripts/slurm_diagonal_loo_franka.sbatch
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=48g --time=06:00:00
#SBATCH --constraint="work"
#SBATCH --job-name=cehj-diag-lf
#SBATCH -o /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.out
#SBATCH -e /work/nvme/bifq/project/tmp/cehj/slurm/%x-%j.err

set -uo pipefail
TMP=/work/nvme/bifq/project/tmp

echo "[job] $(hostname) job $SLURM_JOB_ID diagonal loo_franka 5x5"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif bash -lc "
source /work/nvme/bifq/conda/activate_RoboTwin.sh
export TMP=$TMP TMPDIR=$TMP/tmp
export HOLOBRAIN_CKPT=$TMP/holobrain/HoloBrain_v0.0_GD
export HF_HOME=$TMP/hf-cache XDG_CACHE_HOME=$TMP/xdg-cache
export TRITON_CACHE_DIR=$TMP/triton-cache TORCH_EXTENSIONS_DIR=$TMP/torch-ext
export WANDB_MODE=offline
cd /work/nvme/bifq/project/CEHJ_safe
EXCL=\$TMP/cehj/videos_presentation/selections_used.json,\$TMP/cehj/videos_presentation/stackbowls_loo_franka.json
SEL=\$TMP/cehj/videos_presentation/diagonal_loo_franka.json
python tests/replace_extra_video.py --emb piper --task place_burger_fries \
  --budget 18 --max-trials 8 \
  --exclude-json \$EXCL --selections-out \$SEL \
  --out \$TMP/cehj/videos_presentation || true
# bread/franka: no clean successes exist — success-only (contact allowed)
python tests/replace_extra_video.py --emb franka-panda --task place_bread_basket \
  --budget 18 --allow-contact --max-trials 8 \
  --exclude-json \$EXCL --selections-out \$SEL \
  --out \$TMP/cehj/videos_presentation || true
python tests/replace_extra_video.py --emb aloha-agilex --task stack_bowls_two \
  --budget 30 --max-trials 8 \
  --exclude-json \$EXCL --selections-out \$SEL \
  --out \$TMP/cehj/videos_presentation || true
python tests/replace_extra_video.py --emb ARX-X5 --task stack_blocks_two \
  --budget 18 --max-trials 8 \
  --exclude-json \$EXCL --selections-out \$SEL \
  --out \$TMP/cehj/videos_presentation || true
python tests/replace_extra_video.py --emb ur5-wsg --task place_container_plate \
  --budget 18 --max-trials 8 \
  --exclude-json \$EXCL --selections-out \$SEL \
  --out \$TMP/cehj/videos_presentation || true
echo DIAGONAL-EXIT-OK
"
