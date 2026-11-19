#!/bin/bash
# The sliding-tray fine-tune (Yuxuan 2026-11-12; demos collected 11-18: T 101, TC 102, C 100) as a chain of 1-hour
# interactive A100 / H200 jobs (jobs/ft_c_train.sbatch, one GPU, the 10-28 recipe: batch 32, MAX_STEP steps): the
# first job packs the demos (jobs/prep_tray.sh -> data/lmdb/tray_slide), each job resumes from the newest complete
# checkpoint, the last writes ws/$RUN.DONE. A refused sbatch (one interactive job per user and partition) is retried
# after 5 minutes. Stops at ws/$RUN.DONE or after MAX_PIECES pieces (a piece trains ~750 steps on an A100-40GB: 15k = ~20).
#   ssh dt-login01 'cd /work/nvme/bifq/project/LanguageConditionHJ && setsid nohup bash vla_finetuning/jobs/tray_train_feeder.sh \
#     > /dev/null 2>> /work/nvme/bifq/project/tmp/logs/tray_train_feeder.err < /dev/null &'
R=/work/nvme/bifq/project/LanguageConditionHJ
RUN=${RUN:-ft_tray}; MAX_PIECES=${MAX_PIECES:-30}
LOG=/work/nvme/bifq/project/tmp/logs/tray_train_feeder.log
cd $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "train feeder start: $RUN (pid $$ on $(hostname -s))"
piece=0
while [ $piece -lt $MAX_PIECES ]; do
  [ -f vla_finetuning/ws/$RUN.DONE ] && { log "$RUN DONE: $(cat vla_finetuning/ws/$RUN.DONE)"; exit 0; }
  out=$(RUN=$RUN TASKS=tray_slide LMDBS=$R/vla_finetuning/data/lmdb/tray_slide PREP=vla_finetuning/jobs/prep_tray.sh \
        PRUNE_OLD_CKPT=1 sbatch --parsable --export=ALL --job-name=vf-tray-train vla_finetuning/jobs/ft_c_train.sbatch 2>&1)
  if ! [[ "$out" =~ ^[0-9]+$ ]]; then log "sbatch refused: $(echo $out | tail -c 200)"; sleep 300; continue; fi
  piece=$((piece + 1)); log "piece $piece job $out"
  while squeue -h -j $out 2>/dev/null | grep -q .; do sleep 60; done
  log "piece $piece job $out ended: $(sacct -j $out -X -n -o State,Elapsed,Partition | tr -s ' ')"
done
log "stopped after $MAX_PIECES pieces"
