#!/bin/bash
# The safety filter's data collection (collect.sh) as a self-resubmitting Slurm job: runs until READY.
#
#   cd <repo>
#   export HOLOBRAIN_CKPT=<dir> PY=<python> SBATCH_FLAGS="-A <account> -p GPU-shared --gpus=<rtx type>:1"
#   sbatch $SBATCH_FLAGS scripts/lang/bridges/collect.sbatch            # vanilla data -> data/lang/READY
#   WHAT=all sbatch $SBATCH_FLAGS scripts/lang/bridges/collect.sbatch   # also the Qwen replays -> data/lang_qwen/READY
#
# Site flags go in SBATCH_FLAGS (the resubmissions reuse them; flags typed only on the first sbatch line are lost).
# More GPUs per job (e.g. --gpus=<type>:4) = one queue worker per GPU. Several jobs may run at once: they share the
# queue (runs/lang_collect/<root name>/). The job stops starting shards 10 min before its end, pauses the running
# ones (state saved), and submits itself again unless the data is READY or a shard failed 3 times (then it stops
# for a look: runs/lang_collect/<root name>/logs, <ROOT>/logs). RESUBMIT=0 disables the resubmission.
# Env passed on: WHAT (vanilla | qwen | all), ROOT, QROOT, PAR (see collect.sh).
#SBATCH --job-name=lang-collect
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=runs/bridges_logs/lang-collect_%j.out
set -u
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
source scripts/lang/bridges/_env.sh
end=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null || echo 0)
[[ $end -gt 0 ]] || end=$(( $(date +%s) + 23 * 3600 ))
export STAGE_DEADLINE=$(( end - 600 ))
echo "[collect.sbatch] job $SLURM_JOB_ID on $(hostname), ${WHAT:-vanilla}, stop at $(date -d @"$STAGE_DEADLINE")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash scripts/lang/bridges/collect.sh "${WHAT:-vanilla}"
rc=$?
if [[ $rc -eq 75 && "${RESUBMIT:-1}" == 1 ]]; then
  # shellcheck disable=SC2086
  sbatch --export=ALL ${SBATCH_FLAGS:-} "${SLURM_SUBMIT_DIR}/scripts/lang/bridges/collect.sbatch" \
    && echo "[collect.sbatch] resubmitted"
fi
exit $rc
