#!/bin/bash
# Check 4: keep one retrain pool job on gpuA100x8-interactive until every variant is finished (login node, setsid nohup).
# Waits while ANY of my jobs holds that partition (the actor-stats job first), asks for 3 GPUs, falls back to 2, then 1
# when a request has not started within 10 min, and resubmits after each 55-min job.
#   setsid nohup bash scripts/lang/diag_1030/a100x8_feeder.sh >> runs/lang/diag_1030/retrain/a100x8_feeder.log 2>&1 &
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
OUT=$R/runs/lang/diag_1030/retrain
SB=$OUT/jobs/pool_A100x8.sbatch
P=gpuA100x8-interactive
VARS=det,stoch,stoch_mlp:5000,zero,data,stoch_norm
STEPS=20000
cd $R
log() { echo "$(date '+%F %T') $*"; }
steps_of() { [ -f $OUT/$1/diag.jsonl ] || { echo 0; return; }; tail -1 $OUT/$1/diag.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$' || echo 0; }
all_done() {
  local spec v cap
  for spec in ${VARS//,/ }; do
    v=${spec%%:*}; cap=$STEPS; [[ $spec == *:* ]] && cap=${spec#*:}
    [ -f $OUT/$v/checkpoint_final.pt ] && [ "$(steps_of $v)" -ge "$cap" ] || return 1
  done
}
log "feeder start pid $$"
gpus=3
while ! all_done; do
  while squeue -u $USER -h -p $P 2>/dev/null | grep -q .; do sleep 30; done
  all_done && break
  jid=$(sbatch --gpus-per-node=$gpus --cpus-per-task=$((16 * gpus)) --mem=$((60 * gpus))g "$SB" | awk '{print $4}')
  [ -n "$jid" ] || { log "submit failed; retry in 5 min"; sleep 300; continue; }
  log "submitted $jid ($gpus GPUs)"
  waited=0
  while [ "$(squeue -j $jid -h -o %T 2>/dev/null)" = PENDING ]; do
    sleep 30; waited=$((waited + 30))
    if [ $waited -ge 600 ] && [ $gpus -gt 1 ]; then
      scancel $jid; gpus=$((gpus - 1)); log "$jid not started in 10 min: cancelled, asking for $gpus"; continue 2
    fi
  done
  log "$jid started ($gpus GPUs)"
  while squeue -j $jid -h 2>/dev/null | grep -q .; do sleep 60; done
  log "$jid ended: $(for spec in ${VARS//,/ }; do v=${spec%%:*}; printf '%s %s  ' $v "$(steps_of $v)"; done)"
done
log "every variant finished: feeder stops"
