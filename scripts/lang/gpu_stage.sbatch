#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=lang-chain
# One 1-hour interactive stage with 1-4 GPUs (chain.sh sets --gpus-per-node,
# --cpus-per-task and --mem). STAGECMD runs inside the container with cwd =
# repo root; multi-GPU stages pin each process with CUDA_VISIBLE_DEVICES
# (scripts/lang/par_gpu.sh). STAGE_OK is echoed iff the command exits 0.
set -u
STAGENAME=${STAGENAME:?unset}; STAGECMD=${STAGECMD:?unset}
echo "[stage] $STAGENAME on $(hostname) $(date '+%F %T'): $STAGECMD"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
( while sleep 120; do echo "[gpumem $(date +%T)] $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"; done ) &
mon=$!
# the stage stops itself before the walltime (STAGE_TIMEOUT s, default 55 min): resumable
# commands (collection shards, --time-budget training) exit 75 before it, and the
# chain resubmits the stage without counting a failure (STAGE_CONTINUE); a cut (124) or a kill (137) counts
export STAGE_DEADLINE=$(( $(date +%s) + ${STAGE_TIMEOUT:-3300} - 150 ))
timeout --signal=TERM --kill-after=60 "${STAGE_TIMEOUT:-3300}" \
  /work/nvme/bifq/project/LanguageConditionHJ/scripts/lang/in_container.sh bash -c "$STAGECMD"
rc=$?
kill $mon 2>/dev/null
echo "[stage] $STAGENAME rc=$rc $(date '+%F %T')"
[ $rc -eq 0 ] && echo "STAGE_OK $STAGENAME"
[ $rc -eq 75 ] && echo "STAGE_CONTINUE $STAGENAME"
# a stage cut by the stage timeout (124) or killed (137: kill-after, OOM) did not stop itself: count it as a
# failure, so a hang or a repeatable OOM kill ends after the chain's MAX_TRIES instead of looping (round 3, C8)
{ [ $rc -eq 124 ] || [ $rc -eq 137 ]; } && echo "[stage] $STAGENAME rc=$rc: cut by the timeout or killed; counted as a failure"
exit $rc
