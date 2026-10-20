#!/bin/bash
# Fine-tune HoloBrain on one demo set (PLAN.md "Fine-tuning"): prepare the matched data (once, shared by the two
# arms), then HoloBrain's train.py on 4 GPUs from post_training_robotwin.
#
#   sbatch --export=ALL,ARM=safe    vla_finetuning/jobs/finetune.sbatch
#   sbatch --export=ALL,ARM=nominal vla_finetuning/jobs/finetune.sbatch
#
# Same settings for both arms: MAX_STEP (default 15000) at 8 per GPU x 4 GPUs, a checkpoint every SAVE_EVERY
# (2500; the last 3 kept), the final checkpoint is the one evaluated. Workspace vla_finetuning/ws/ft_<arm>.
#SBATCH --job-name=vf-ft
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 --cpus-per-task=64
#SBATCH --mem=220g --time=12:00:00
#SBATCH --output=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning/logs/finetune_%j.out
set -uo pipefail
ARM=${ARM:?ARM=safe or ARM=nominal}
R=/work/nvme/bifq/project/LanguageConditionHJ
cd "$R"
echo "[finetune] arm $ARM job $SLURM_JOB_ID on $(hostname) $(date)"
scripts/lang/in_container.sh bash vla_finetuning/jobs/prep_data.sh || exit 1
scripts/lang/in_container.sh bash vla_finetuning/jobs/train.sh "ft_$ARM" "$R/vla_finetuning/data/lmdb/matched_$ARM" \
  "${MAX_STEP:-15000}" "${SAVE_EVERY:-2500}" 4
rc=$?
echo "[finetune] train.sh exit $rc $(date)"
exit $rc
