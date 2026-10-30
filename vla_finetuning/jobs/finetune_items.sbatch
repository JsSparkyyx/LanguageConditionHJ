#!/bin/bash
# Fine-tune HoloBrain on the item-scene safe demonstrations (Yuxuan 2026-10-28: T1 / T3 / T5 / T6 / T7, ONE model
# for all five tasks, the instruction tells them apart; ONE GPU per fine-tune job). The recipe of finetune.sbatch
# (from post_training_robotwin, 15000 optimizer steps, a batch of 32, lr 1e-4, the LR drop at 90 %) on one GPU:
# 8 per micro-batch x GRAD_ACCUM 4 micro-batches per optimizer step (train.sh), about 15-20 h, so the run resumes
# across jobs: jobs/ft_chain_step.sh trains from the newest complete checkpoint (a save every SAVE_EVERY = 500 steps)
# until 15 min before this job's end, then this job submits itself again. Workspace vla_finetuning/ws/ft_items_safe
# (resumes in ws/ft_items_safe_r<time>); done = ws/ft_items_safe.DONE (the final checkpoint's path).
# Submitted before the demos are all collected (the batch queue is days long): prep_items.sh packs them only when
# every task has its kept demos; until then the job waits up to WAIT_H hours (default 1), then resubmits itself.
#
#   sbatch vla_finetuning/jobs/finetune_items.sbatch
#SBATCH --job-name=vf-ft-items
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64g --time=48:00:00
#SBATCH --output=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning/logs/finetune_items_%j.out
set -uo pipefail
R=/work/nvme/bifq/project/LanguageConditionHJ
SELF=$R/vla_finetuning/jobs/finetune_items.sbatch
cd "$R"
export GRAD_ACCUM=${GRAD_ACCUM:-4}
# the job's end: Slurm's SLURM_JOB_END_TIME, else squeue (an empty answer must not reach `date -d ""`, which reads as
# today's midnight), else 47 h; anything not at least 1 h ahead is not this job's end (review 10-28)
end=${SLURM_JOB_END_TIME:-0}
if ! [ "$end" -gt $(( $(date +%s) + 3600 )) ] 2>/dev/null; then
  e=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
  end=0; [ -n "$e" ] && end=$(date -d "$e" +%s 2>/dev/null || echo 0)
fi
[ "$end" -gt $(( $(date +%s) + 3600 )) ] 2>/dev/null || end=$(( $(date +%s) + 47 * 3600 ))
export STAGE_DEADLINE=$(( end - 900 ))
echo "[finetune_items] job $SLURM_JOB_ID on $(hostname) $(date), grad_accum $GRAD_ACCUM, stop at $(date -d @$STAGE_DEADLINE)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
resubmit() {   # queue the next job; retry transient failures; never queue a second one (review 10-28)
  local i jid other
  for i in 1 2 3 4 5; do
    jid=$(sbatch --parsable --export=ALL "$SELF") && { echo "[finetune_items] resubmitted as $jid"; return 0; }
    sleep 60
    other=$(squeue -h -u "$USER" -n vf-ft-items -o %i | grep -vx "$SLURM_JOB_ID")   # a timed-out submit may have queued it
    [ -n "$other" ] && { echo "[finetune_items] next job already queued: $other"; return 0; }
  done
  echo "[finetune_items] resubmission FAILED after 5 tries: sbatch $SELF by hand"
  return 1
}
limit=$(( $(date +%s) + ${WAIT_H:-1} * 3600 ))
while true; do
  scripts/lang/in_container.sh bash vla_finetuning/jobs/ft_chain_step.sh ft_items_safe \
    "$R/vla_finetuning/data/lmdb/items_safe_selective_avoid,$R/vla_finetuning/data/lmdb/items_safe_push_zone,$R/vla_finetuning/data/lmdb/items_safe_tray_transport,$R/vla_finetuning/data/lmdb/items_safe_clean_pour,$R/vla_finetuning/data/lmdb/items_safe_gap_place" "${MAX_STEP:-15000}" "${SAVE_EVERY:-500}" 1 vla_finetuning/jobs/prep_items.sh
  rc=$?
  [ $rc -eq 0 ] && { echo "[finetune_items] done $(date)"; exit 0; }
  [ $rc -ne 75 ] && { echo "[finetune_items] failed ($rc)"; exit $rc; }
  if [ -f "$R/vla_finetuning/data/PREP_ITEMS_DONE" ]; then   # trained up to this job's deadline
    resubmit; exit $?
  fi
  if (( $(date +%s) > limit )); then                          # the demos are still short
    echo "[finetune_items] demos still short after ${WAIT_H:-1} h: resubmitting"
    resubmit; exit $?
  fi
  sleep 600
done
