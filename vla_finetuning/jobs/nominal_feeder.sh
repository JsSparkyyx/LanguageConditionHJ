#!/bin/bash
# The safety-agnostic VLA (Yuxuan 2026-11-21: careless demos that finish the task, "a variant of the nominal curobo
# planner"). Phase 1: the careless T6 / T8 demos (nominal_collect.sbatch) as 1-hour gpuA40x4-interactive pieces until
# every shard is DONE. Phase 2: the fine-tune RUN (default ft_nominal; ft_c_train.sbatch, the 10-28 recipe: one GPU,
# batch 32, 15k steps) as 1-hour A100 / H200 interactive pieces until ws/$RUN.DONE; its first piece packs the data
# (prep_nominal.sh: the T1 / T3 / T7 view + the collected demos). A refused sbatch (one interactive job per user and
# partition) is retried after 5 minutes; a piece of this feeder's kind already queued is waited for, not doubled.
#   ssh dt-login01 'cd /work/nvme/bifq/project/LanguageConditionHJ && setsid nohup bash vla_finetuning/jobs/nominal_feeder.sh \
#     > /dev/null 2>> /work/nvme/bifq/project/tmp/logs/nominal_feeder.err < /dev/null &'
R=/work/nvme/bifq/project/LanguageConditionHJ
ROOT=$R/vla_finetuning/data/demos_nominal
RUN=${RUN:-ft_nominal}; MAX_COLLECT=${MAX_COLLECT:-10}; MAX_TRAIN=${MAX_TRAIN:-30}
TASKS="selective_avoid push_zone clean_pour gap_place tray_slide"
LMDBS=$(printf "$R/vla_finetuning/data/lmdb/nominal_%s," $TASKS); LMDBS=${LMDBS%,}
SHARDS="clean_pour/n0 tray_slide/n0 clean_pour/n1 tray_slide/n1 clean_pour/n2 tray_slide/n2 clean_pour/n3 tray_slide/n3"
LOG=/work/nvme/bifq/project/tmp/logs/nominal_feeder.log
cd $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
wait_job() { while squeue -h -j $1 2>/dev/null | grep -q .; do sleep 60; done; }
mine() { squeue --me -h -n $1 -o %i | head -1; }
log "feeder start: $RUN (pid $$ on $(hostname -s))"

piece=0
while true; do
  left=0
  for s in $SHARDS; do [ -f ${ROOT}_logs/$s/DONE ] || left=$((left + 1)); done
  [ $left -eq 0 ] && { log "collection: every shard DONE"; break; }
  [ $piece -ge $MAX_COLLECT ] && { log "collection: stopped after $MAX_COLLECT pieces ($left shards left)"; exit 1; }
  j=$(mine nominal-collect)
  if [ -z "$j" ]; then
    j=$(sbatch --parsable vla_finetuning/jobs/nominal_collect.sbatch 2>&1)
    if ! [[ "$j" =~ ^[0-9]+$ ]]; then log "collect sbatch refused: $(echo $j | tail -c 200)"; sleep 300; continue; fi
  fi
  piece=$((piece + 1)); log "collect piece $piece job $j ($left shards left)"
  wait_job $j
  log "collect piece $piece job $j ended: $(sacct -j $j -X -n -o State,Elapsed | tr -s ' ')"
done

piece=0
while [ ! -f vla_finetuning/ws/$RUN.DONE ]; do
  [ $piece -ge $MAX_TRAIN ] && { log "train: stopped after $MAX_TRAIN pieces"; exit 1; }
  j=$(mine vf-nominal-train)
  if [ -z "$j" ]; then
    j=$(RUN=$RUN TASKS="$TASKS" LMDBS=$LMDBS PREP=vla_finetuning/jobs/prep_nominal.sh PRUNE_OLD_CKPT=1 \
        sbatch --parsable --export=ALL --job-name=vf-nominal-train vla_finetuning/jobs/ft_c_train.sbatch 2>&1)
    if ! [[ "$j" =~ ^[0-9]+$ ]]; then log "train sbatch refused: $(echo $j | tail -c 200)"; sleep 300; continue; fi
  fi
  piece=$((piece + 1)); log "train piece $piece job $j"
  wait_job $j
  log "train piece $piece job $j ended: $(sacct -j $j -X -n -o State,Elapsed,Partition | tr -s ' ')"
done
log "$RUN DONE: $(cat vla_finetuning/ws/$RUN.DONE)"
