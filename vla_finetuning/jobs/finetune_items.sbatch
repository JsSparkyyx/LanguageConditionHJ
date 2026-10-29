#!/bin/bash
# Fine-tune HoloBrain on the item-scene safe demonstrations (Yuxuan 2026-10-28: T1 / T3 / T5 / T6 / T7, ONE model
# for all five tasks, the instruction tells them apart): the recipe of finetune.sbatch (4 GPUs from
# post_training_robotwin, MAX_STEP 15000 at 8 per GPU, a checkpoint every SAVE_EVERY 2500, the last 3 kept),
# workspace vla_finetuning/ws/ft_items_safe. Submitted before the demos are all collected (the batch queue is days
# long): prep_items.sh packs them only when every task has its kept demos; until then the job waits up to WAIT_H
# hours (default 1) and then submits itself again and exits, so an early start holds the GPUs at most that long.
#
#   sbatch vla_finetuning/jobs/finetune_items.sbatch
#SBATCH --job-name=vf-ft-items
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 --cpus-per-task=64
#SBATCH --mem=220g --time=12:00:00
#SBATCH --output=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning/logs/finetune_items_%j.out
set -uo pipefail
R=/work/nvme/bifq/project/LanguageConditionHJ
SELF=$R/vla_finetuning/jobs/finetune_items.sbatch
cd "$R"
echo "[finetune_items] job $SLURM_JOB_ID on $(hostname) $(date)"
limit=$(( $(date +%s) + ${WAIT_H:-1} * 3600 ))
while true; do
  scripts/lang/in_container.sh bash vla_finetuning/jobs/prep_items.sh
  rc=$?
  [ $rc -eq 0 ] && break
  [ $rc -ne 75 ] && { echo "[finetune_items] prep failed ($rc)"; exit $rc; }
  if (( $(date +%s) > limit )); then
    echo "[finetune_items] demos still short after ${WAIT_H:-1} h: resubmitting"
    sbatch --export=ALL "$SELF"
    exit 0
  fi
  sleep 600
done
if [ -d "$R/vla_finetuning/ws/ft_items_safe/checkpoints" ]; then
  echo "[finetune_items] ws/ft_items_safe already has checkpoints: a resume needs train.sh's resume argument"; exit 2
fi
scripts/lang/in_container.sh bash vla_finetuning/jobs/train.sh ft_items_safe "$R/vla_finetuning/data/lmdb/items_safe" \
  "${MAX_STEP:-15000}" "${SAVE_EVERY:-2500}" 4
rc=$?
echo "[finetune_items] train.sh exit $rc $(date)"
exit $rc
