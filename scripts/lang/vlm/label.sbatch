#!/bin/bash
# The VLM labeler (main/lang/vlm_label.py, VLM_FAILURE_PLAN) as a self-resubmitting Slurm job: one per VLM, on one
# GPU with 80 GB or more (A100 80 GB / H200; a 32B model in bf16 is ~66 GB). It serves a run's ring and its warmup
# shards until the run's final checkpoint exists.
#
#   VLM=qwen25vl32b ROOTS="<run>/ring <warmup>/shard_g*" UNTIL=<run>/checkpoint_final.pt \
#   DONE_FILE=<warmup>/LABELS_<vlm> DONE_ROOTS="<warmup>/shard_g*" DONE_AFTER=<data>/READY \
#   SBATCH_FLAGS="-A bifq-delta-gpu -p gpuA100x4 --gpus-per-node=1 --mem=120g" \
#   sbatch --export=ALL $SBATCH_FLAGS -J vlm-label-<vlm> scripts/lang/vlm/label.sbatch
#
# ROOTS / DONE_ROOTS are globs expanded by the labeler (never by this shell). LANG_REPO selects the checkout
# (in_container.sh). The job stops 5 min before its time limit (exit 75) and submits itself again (RESUBMIT=0 to
# disable); claims make a second labeler on the same roots safe.
#SBATCH --job-name=vlm-label
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=runs/bridges_logs/vlm-label_%j.out
set -u
set -f   # the globs go to the labeler
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
: "${VLM:?set VLM=<name>}" "${ROOTS:?set ROOTS=<roots or globs>}"
end=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null || echo 0)
[[ $end -gt 0 ]] || end=$(( $(date +%s) + 47 * 3600 ))
export STAGE_DEADLINE=$end
echo "[label.sbatch] job $SLURM_JOB_ID on $(hostname), $VLM, roots $ROOTS, until ${UNTIL:-never}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
args=(serve --vlm "$VLM" --roots $ROOTS)
[[ -n "${UNTIL:-}" ]] && args+=(--until "$UNTIL")
[[ -n "${DONE_FILE:-}" ]] && args+=(--done-file "$DONE_FILE" --done-roots $DONE_ROOTS --done-after "${DONE_AFTER:-}")
[[ -n "${BATCH:-}" ]] && args+=(--batch "$BATCH")
${PY:-bash scripts/lang/in_container.sh python} -m main.lang.vlm_label "${args[@]}"
rc=$?
echo "[label.sbatch] labeler exit $rc"
if [[ $rc -eq 75 && "${RESUBMIT:-1}" == 1 ]]; then
  # shellcheck disable=SC2086
  sbatch --export=ALL ${SBATCH_FLAGS:-} -J "${SLURM_JOB_NAME:-vlm-label}" "$SLURM_SUBMIT_DIR/scripts/lang/vlm/label.sbatch"
fi
exit $rc
