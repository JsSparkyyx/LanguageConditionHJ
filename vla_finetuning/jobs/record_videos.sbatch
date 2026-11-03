#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1-4 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-videos -o /work/nvme/bifq/project/tmp/logs/vf_videos_%j.out
# Success / fail videos of every controller and item task (items/record_videos.py, Yuxuan 11-03): the 25
# (controller, task) parts spread over the nodes this job got, SLOTS per GPU; resubmit until every part is done (each
# part resumes per scene). ITEMS (optional env): a space-separated subset "controller:task ...". OUT: default vla_videos.
#   sbatch vla_finetuning/jobs/record_videos.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
OUT=${OUT:-$R/vla_videos}; SLOTS=${SLOTS:-3}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 240 ))
CL=$OUT/_claims/$SLURM_JOB_ID; mkdir -p $CL
echo "[videos] job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST ($SLURM_NNODES), deadline $(date -d @$STAGE_DEADLINE +%T)"
srun --ntasks-per-node=1 --kill-on-bad-exit=0 scripts/lang/in_container.sh bash vla_finetuning/jobs/record_videos_pool.sh \
  $CL $SLOTS $OUT "${ITEMS:-}"
scripts/lang/in_container.sh python -m vla_finetuning.items.record_videos --index --out $OUT
