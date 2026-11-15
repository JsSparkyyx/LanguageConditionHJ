#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA100x4-interactive,gpuA100x8-interactive,gpuH200x8-interactive
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-ctrain -o /work/nvme/bifq/project/tmp/logs/vf_ctrain_%j.out
# The constraint fine-tune (Yuxuan 11-03): HoloBrain from post_training_robotwin (the config's default start) on the
# constraint demos of the five tasks, the 10-28 recipe (one GPU, GRAD_ACCUM 4 = batch 32, 15k steps), resumed from the
# newest complete checkpoint each hour (ft_chain_step.sh); the first job packs the data (prep_constraint.sh). The A40
# interactive partition is left to the collection and the tests. Submitted by constraint_feeder.sh.
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
RUN=${RUN:-ft_constraint}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 240 ))
export GRAD_ACCUM=4
# TASKS (env): train on these tasks only (Yuxuan 11-06: a T1-only Qwen run); default every active task.
# VF_CONFIG (env) passes through to train.sh (the Qwen config for the Qwen runs).
T=${TASKS:-$(scripts/lang/in_container.sh python -m vla_finetuning.items.constraint --active)}
[ -n "$T" ] || { echo "[ctrain] task lookup failed"; exit 75; }
L=$(printf "$R/vla_finetuning/data/lmdb/constraint_%s," $T)
L=${L%,}
# LMDBS (env): the packed data, comma-separated (11-08, the 500-scene T1 runs: the 100 scenes above + the t1x set's
# 400); CSET then makes the prep step pack that set. MAX_STEP (env): the run's total steps (default 15000).
L=${LMDBS:-$L}
echo "[ctrain] $RUN tasks $T data $L set ${CSET:-main} config ${VF_CONFIG:-gd} on $(hostname) $SLURM_JOB_PARTITION $(nvidia-smi --query-gpu=name --format=csv,noheader), deadline $(date -d @$STAGE_DEADLINE +%T)"
# PREP (env): the data prep script (11-13: jobs/prep_tray.sh for the sliding-tray run, with LMDBS=data/lmdb/tray_slide)
scripts/lang/in_container.sh bash vla_finetuning/jobs/ft_chain_step.sh $RUN $L ${MAX_STEP:-15000} ${SAVE_EVERY:-250} 1 \
  ${PREP:-vla_finetuning/jobs/prep_constraint.sh}
rc=$?
echo "[ctrain] rc=$rc $(date '+%F %T')"
exit $rc
