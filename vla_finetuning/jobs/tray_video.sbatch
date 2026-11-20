#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=96g --time=01:00:00
#SBATCH --job-name=tray-video -o /work/nvme/bifq/project/tmp/logs/tray_video_%j.out
# Videos of the sliding-tray test (items/video_tray.py): per scene in SCENES (env, default "0 1 2") and per prompt, the
# fine-tuned model NAME (env, default ft_tray; models/<NAME>/model, exported by tray_eval.sbatch) above the careless and
# the safe nominal, on the A40 (the cameras need ray tracing). Resubmitting resumes (finished videos are skipped).
#   sbatch vla_finetuning/jobs/tray_video.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
RES=${RES:-$R/vla_finetuning/results_tray}; NAME=${NAME:-ft_tray}; SCENES=${SCENES:-"0 1 2"}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 180 ))
mkdir -p $RES/_logs
while true; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.video_tray --model-dir $R/vla_finetuning/models/$NAME/model \
    --model-name $NAME --res $RES --scenes $SCENES >> $RES/_logs/video_tray.log 2>&1
  rc=$?
  [ $rc -eq 75 ] && [ $(( STAGE_DEADLINE - $(date +%s) )) -gt 600 ] && continue
  break
done
echo "[tray-video] $(date '+%T') rc=$rc, videos: $(ls $RES/videos/*.mp4 2>/dev/null | wc -l)"
