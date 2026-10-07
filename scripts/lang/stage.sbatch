#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64g --time=01:00:00
#SBATCH --job-name=lang-stage
# One 1-hour interactive stage: STAGECMD runs inside the container, cwd = repo root.
#   sbatch --export=ALL,STAGENAME=x,STAGECMD='python -m ...' scripts/lang/stage.sbatch
set -u
STAGENAME=${STAGENAME:?unset}; STAGECMD=${STAGECMD:?unset}
echo "[stage] $STAGENAME on $(hostname): $STAGECMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
/work/nvme/bifq/project/LanguageConditionHJ/scripts/lang/in_container.sh bash -c "$STAGECMD"
rc=$?; echo "[stage] $STAGENAME rc=$rc"; exit $rc
