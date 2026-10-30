#!/bin/bash
# Check 4: keep one retrain pool job on gpuA40x4-interactive until every variant is finished (login node, setsid nohup).
# Starts only after the T5-branch agent is done (no diag1030-t5branch job for 15 min and its workflow agent idle), then
# keeps one 1-4 node x 1 GPU pool job on the partition (resubmitted after each 55-min job)
# when a request has not started within 10 min, and resubmits after each 55-min job.
#   setsid nohup bash scripts/lang/diag_1030/a40_feeder.sh >> runs/lang/diag_1030/retrain/a40_feeder.log 2>&1 &
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
OUT=$R/runs/lang/diag_1030/retrain
SB=$OUT/jobs/pool_A40.sbatch
P=gpuA40x4-interactive
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
J=/u/yyang69/.claude/projects/-work-nvme-bifq-project-LanguageConditionHJ/ecb33a26-96d0-41d8-932b-092304bf99bb/subagents/workflows/wf_4ef2a3f1-4a3
T5LOG=$(grep -l '"description":"t5-branch"' $J/agent-*.meta.json | sed 's/.meta.json$/.jsonl/')
idle=0
while [ $idle -lt 30 ]; do   # 15 min with no T5-branch job and no activity of its agent
  if squeue -u $USER -h -o %j | grep -q diag1030-t5branch || [ $(( $(date +%s) - $(stat -c %Y "$T5LOG") )) -lt 600 ]; then
    idle=0; else idle=$((idle + 1)); fi
  sleep 30
done
log "T5-branch check idle: taking the A40 slot"
gpus=1
while ! all_done; do
  while squeue -u $USER -h -p $P 2>/dev/null | grep -q .; do sleep 30; done
  all_done && break
  jid=$(sbatch "$SB" | awk '{print $4}')
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
