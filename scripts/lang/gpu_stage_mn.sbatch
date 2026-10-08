#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=lang-mn
# Multi-node stage (e.g. --nodes=1-4 --gpus-per-node=1: whatever single free GPUs exist). srun runs
# STAGECMD once per node inside the container (CUDA_VISIBLE_DEVICES set per node by Slurm); commands
# share work through main.lang.workqueue. STAGE_OK iff every node exits 0; STAGE_CONTINUE on 75.
set -u
STAGENAME=${STAGENAME:?unset}; STAGECMD=${STAGECMD:?unset}
echo "[stage-mn] $STAGENAME on $SLURM_JOB_NODELIST ($SLURM_NNODES nodes) $(date '+%F %T'): $STAGECMD"
export STAGE_DEADLINE=$(( $(date +%s) + ${STAGE_TIMEOUT:-3300} - 150 ))
export STAGECMD
srun --ntasks-per-node=1 --kill-on-bad-exit=0 bash -c '
  timeout --signal=TERM --kill-after=60 '"${STAGE_TIMEOUT:-3300}"' \
    /work/nvme/bifq/project/LanguageConditionHJ/scripts/lang/in_container.sh bash -c "$STAGECMD"
  rc=$?; echo "[node $SLURM_NODEID $(hostname)] rc=$rc"; exit $rc'
rc=$?
echo "[stage-mn] $STAGENAME rc=$rc $(date '+%F %T')"
[ $rc -eq 0 ] && echo "STAGE_OK $STAGENAME"
[ $rc -eq 75 ] && echo "STAGE_CONTINUE $STAGENAME"
# cut by the timeout (124) or killed (137): a failure, so a hang cannot loop forever (round 3, C8; as gpu_stage)
{ [ $rc -eq 124 ] || [ $rc -eq 137 ]; } && echo "[stage-mn] $STAGENAME rc=$rc: cut or killed; counted as a failure"
exit $rc
