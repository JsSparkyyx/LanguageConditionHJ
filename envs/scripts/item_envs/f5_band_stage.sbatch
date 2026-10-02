#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=01:00:00
#SBATCH --job-name=f5-band
#SBATCH -o /work/nvme/bifq/project/tmp/logs/f5band_%A.out

# F5 visibility band, interactive-partition staging (f5_visband.sbatch split
# to fit the 1 h cap). Submit with:
#   sbatch --export=ALL,STAGE=compute         envs/scripts/item_envs/f5_band_stage.sbatch
#   sbatch --export=ALL,STAGE=validate        envs/scripts/item_envs/f5_band_stage.sbatch  (both embodiments, only if < 1 h)
#   sbatch --export=ALL,STAGE=validate-ur5    envs/scripts/item_envs/f5_band_stage.sbatch
#   sbatch --export=ALL,STAGE=validate-franka envs/scripts/item_envs/f5_band_stage.sbatch

set -e
STAGE=${STAGE:-compute}
case "$STAGE" in
  compute)          ARGS="--embodiment both" ;;
  validate)         ARGS="--embodiment both --validate" ;;
  validate-ur5)     ARGS="--embodiment ur5-wsg --validate" ;;
  validate-franka)  ARGS="--embodiment franka-panda --validate" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
echo "[f5-band] stage=$STAGE args='$ARGS'"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c "cd /work/nvme/bifq/project/LanguageConditionHJ && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python envs/scripts/item_envs/visibility_band.py $ARGS"
