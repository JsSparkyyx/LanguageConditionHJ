#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=96g --time=48:00:00
#SBATCH --requeue --signal=B:USR1@300
#SBATCH --job-name=lang-long
# A long stage (partition maximum walltime). STAGECMD must be resumable (it is
# re-run from the start on requeue, e.g. `train rounds --resume`). Exit code 75
# (the init limit) re-launches the command in the same job.
#   sbatch --export=ALL,STAGENAME=x,STAGECMD='...' scripts/lang/long.sbatch
set -u
STAGENAME=${STAGENAME:?unset}; STAGECMD=${STAGECMD:?unset}
trap 'echo "[long] USR1: near walltime, requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
echo "[long] $STAGENAME on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}): $STAGECMD"
while true; do
  /work/nvme/bifq/project/LanguageConditionHJ/scripts/lang/in_container.sh bash -c "$STAGECMD" &
  wait $!; rc=$?
  [ $rc -eq 75 ] || break
  echo "[long] exit 75 (init limit): relaunching"
done
echo "[long] $STAGENAME rc=$rc"; exit $rc
