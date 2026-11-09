#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1-4 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-ceval -o /work/nvme/bifq/project/tmp/logs/vf_ceval_%j.out
# The paired constraint test (items/evaluate_constraint.py) of NAMES (env, default "scripted ft_items_safe_x2"):
# scene lists first, then every (name, task, part); resubmitted by constraint_feeder.sh until every marker exists.
# A fine-tuned model is exported first when its training is DONE and models/<name>/model is missing.
#   NAMES="scripted ft_items_safe_x2 ft_constraint" sbatch --export=ALL vla_finetuning/jobs/constraint_eval.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
NAMES=${NAMES:-"scripted ft_items_safe_x2"}; SLOTS=${SLOTS:-3}
# RES (env): the results root (11-08: results_constraint_t1, the 100-scene T1 test); the pool also reads EVAL_TASKS,
# SCENES and PARTS from the environment (constraint_eval_pool.sh)
export RES=${RES:-$R/vla_finetuning/results_constraint}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 240 ))
for n in $NAMES; do
  if [ "$n" != scripted ] && [ ! -d vla_finetuning/models/$n/model ] && [ -f vla_finetuning/ws/$n.DONE ]; then
    ck=$(head -1 vla_finetuning/ws/$n.DONE)/model/model.safetensors
    echo "[ceval] export $n from $ck"
    scripts/lang/in_container.sh bash vla_finetuning/jobs/export.sh $n $ck || exit 1
  fi
done
CL=$RES/_claims/$SLURM_JOB_ID; mkdir -p $CL
echo "[ceval] job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST res $RES tasks ${EVAL_TASKS:-all} scenes ${SCENES:-20} parts ${PARTS:-default} names $NAMES deadline $(date -d @$STAGE_DEADLINE +%T)"
srun --ntasks-per-node=1 --kill-on-bad-exit=0 scripts/lang/in_container.sh bash vla_finetuning/jobs/constraint_eval_pool.sh \
  $CL $SLOTS "$NAMES"
