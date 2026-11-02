#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1-4 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-eval-nominal -o /work/nvme/bifq/project/tmp/logs/vf_eval_nominal_%j.out
# The item scenes' careless nominal and safe nominal on the fine-tune test's 100 scenes (evaluate_nominal.py): the 10
# (mode, task) parts spread over the nodes this job got, 3 per GPU; resubmit until both summaries exist (each part
# resumes per episode). At the end, if every part is complete, the parent per mode writes summary.json.
#   sbatch vla_finetuning/jobs/eval_nominal.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; RES=$R/vla_finetuning/results_items
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 300 ))
CL=$RES/nominal_claims/$SLURM_JOB_ID; mkdir -p $CL
echo "[eval_nominal] job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST ($SLURM_NNODES), deadline $(date -d @$STAGE_DEADLINE +%T)"
srun --ntasks-per-node=1 --kill-on-bad-exit=0 scripts/lang/in_container.sh bash vla_finetuning/jobs/eval_nominal_pool.sh $CL 3
for m in careless safe; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_nominal --mode $m --out $RES/nominal_$m \
    --ref $RES/base > $RES/nominal_$m/log_parent_${SLURM_JOB_ID}.txt 2>&1
  echo "[eval_nominal] $m parent rc=$? summary: $(ls $RES/nominal_$m/summary.json 2>/dev/null)"
done
