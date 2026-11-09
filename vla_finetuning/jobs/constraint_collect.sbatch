#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1-4 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-ccollect -o /work/nvme/bifq/project/tmp/logs/vf_ccollect_%j.out
# The constraint fine-tune's demo collection (items/collect_constraint.py, Yuxuan 11-03): the shards spread over the
# nodes this job got, SLOTS per GPU; resubmitted by constraint_collect_feeder.sh until every shard is DONE.
# TASKS (optional env): a space-separated subset of the tasks. CSET (optional env): a shard set of items/constraint.py
# (t1x: the extra T1 scenes of 11-08, root data/demos_constraint_t1x).
#   sbatch vla_finetuning/jobs/constraint_collect.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
# SLOTS: collectors per GPU, 2 from 11-09 (was 3). Each collector holds ~15-17 GB of the A40's 44 GB, so 3 filled the
# GPU: about once per job all three froze together (the pool's 15-min watchdog then killed them, 15-20 min of the
# whole GPU lost, and the relaunch often found no memory for cuRobo, rc 3), and the GPU was already 60-100 % busy.
ROOT=${ROOT:-$R/vla_finetuning/data/demos_constraint${CSET:+_$CSET}}; SLOTS=${SLOTS:-2}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 240 ))
CL=${ROOT}_logs/_claims/$SLURM_JOB_ID; mkdir -p $CL
echo "[collect] set ${CSET:-main} root $ROOT job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST ($SLURM_NNODES), deadline $(date -d @$STAGE_DEADLINE +%T)"
srun --ntasks-per-node=1 --kill-on-bad-exit=0 scripts/lang/in_container.sh bash vla_finetuning/jobs/constraint_collect_pool.sh \
  $CL $SLOTS $ROOT ${TASKS:-}
