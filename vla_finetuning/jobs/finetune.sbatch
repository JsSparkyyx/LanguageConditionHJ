#!/bin/bash
# Fine-tune HoloBrain on the safe demonstrations (PLAN.md "Fine-tuning"): pack them (prep_data.sh, once), then
# HoloBrain's train.py on 4 GPUs from post_training_robotwin.
#
#   sbatch vla_finetuning/jobs/finetune.sbatch
#
# MAX_STEP (default 15000) at 8 per GPU x 4 GPUs, a checkpoint every SAVE_EVERY (2500; the last 3 kept); the final
# checkpoint is the one evaluated. Workspace vla_finetuning/ws/ft_safe.
#SBATCH --job-name=vf-ft-safe
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 --cpus-per-task=64
#SBATCH --mem=220g --time=12:00:00
#SBATCH --output=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning/logs/finetune_%j.out
set -uo pipefail
R=/work/nvme/bifq/project/LanguageConditionHJ
cd "$R"
echo "[finetune] job $SLURM_JOB_ID on $(hostname) $(date)"
scripts/lang/in_container.sh bash vla_finetuning/jobs/prep_data.sh || exit 1
scripts/lang/in_container.sh bash vla_finetuning/jobs/train.sh ft_safe "$R/vla_finetuning/data/lmdb/safe" \
  "${MAX_STEP:-15000}" "${SAVE_EVERY:-2500}" 4
rc=$?
echo "[finetune] train.sh exit $rc $(date)"
exit $rc
