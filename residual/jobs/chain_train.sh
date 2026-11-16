#!/bin/bash
# Feeder: runs a PPO run as a chain of 1-hour gpuA40x4-interactive jobs (residual/jobs/train_chunk.sbatch) until
# <run>/done exists. Start it over ssh on a login node, never from a Claude shell:
#   ssh dt-login03 "nohup setsid bash $REPO/residual/jobs/chain_train.sh <RUN> [UPDATES] ['k=v k=v'] > /dev/null 2>&1 < /dev/null &"
# The chain stops when three pieces in a row end without a new checkpoint (no progress).
# PART=<partition> picks the partition (default gpuA40x4-interactive); on gpuA100x4-interactive the feeder sets
# CUROBO_SRC to the A100 cuRobo build unless it is already set (see train_chunk.sbatch).
RUN=${1:?run name}; UPDATES=${2:-350}
# optional 3rd argument: the --set overrides of a new run (train_chunk.sbatch reads SETS; --export=ALL passes it)
[ -n "${3:-}" ] && export SETS="$3"
REPO=/work/nvme/bifq/project/LanguageConditionHJ
PART=${PART:-gpuA40x4-interactive}
[[ "$PART" == gpuA100* && -z "${CUROBO_SRC:-}" ]] && export CUROBO_SRC=/work/nvme/bifq/project/tmp/curobo_sm80/src
DIR=$REPO/runs/residual/$RUN
LOG=/work/nvme/bifq/project/tmp/residual/chain_train_$RUN.log
mkdir -p "$(dirname "$LOG")"
cd $REPO
echo "$(date '+%F %T') feeder pid $$ on $(hostname) for $RUN ($UPDATES updates) on $PART sets: ${SETS:-default}" >> $LOG
sq() { local o; o=$(squeue --me -h -n residual-train -o %i 2>/dev/null) || return 2; [[ -z "$o" || "$o" =~ ^[0-9[:space:]]+$ ]] || return 2; echo "$o"; }
ckpt_stamp() { stat -c %Y "$DIR/checkpoint.pt" 2>/dev/null || echo 0; }
stale=0; last=$(ckpt_stamp); submitted=0
while true; do
  [ -f "$DIR/done" ] && { echo "$(date '+%F %T') done" >> $LOG; exit 0; }
  q=$(sq); rc=$?
  if [ $rc -ne 0 ] || [ -n "$q" ]; then sleep 120; continue; fi
  if [ $submitted -eq 1 ]; then
    now=$(ckpt_stamp)
    if [ "$now" = "$last" ]; then stale=$((stale+1)); else stale=0; fi
    last=$now
    if [ $stale -ge 3 ]; then echo "$(date '+%F %T') three pieces without a new checkpoint: chain stops" >> $LOG; exit 1; fi
  fi
  out=$(sbatch -p "$PART" --export=ALL,RUN=$RUN,UPDATES=$UPDATES residual/jobs/train_chunk.sbatch 2>&1)
  if ! [[ "$out" =~ Submitted\ batch\ job\ [0-9]+ ]]; then
    # sbatch refused (Slurm maintenance 11-13 07:43): retry later, not counted as a piece
    echo "$(date '+%F %T') sbatch failed: $(echo "$out" | grep -v '^=*$' | head -2 | tr '\n' ' ')" >> $LOG
    sleep 300; continue
  fi
  submitted=1
  echo "$(date '+%F %T') piece submitted: $out" >> $LOG
  sleep 120
done
