#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=01:00:00
#SBATCH --job-name=chain-stage

# Generic 1-h interactive stage for the chained GPU pipeline.
# Submit via interactive_chain.sh (or manually):
#   sbatch --export=ALL,STAGENAME=x,STAGECMD='...' envs/scripts/item_envs/stage_runner.sbatch
# STAGECMD runs with cwd = <repo>/envs inside the apptainer. STAGE_OK is echoed
# iff the command exits 0; a slurm timeout kill never prints it -> requeue.

set -u
STAGENAME=${STAGENAME:?unset}
STAGECMD=${STAGECMD:?unset}
echo "[stage] $STAGENAME cmd: $STAGECMD"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c "cd /work/nvme/bifq/project/LanguageConditionHJ/envs || exit 1; $STAGECMD"
rc=$?
echo "[stage] $STAGENAME inner rc=$rc"
[ $rc -eq 0 ] && echo "STAGE_OK $STAGENAME"
exit $rc
