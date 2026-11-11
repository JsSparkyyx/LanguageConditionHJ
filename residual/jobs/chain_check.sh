#!/bin/bash
# Feeder: runs residual/jobs/check.sbatch as a chain of 1-hour gpuA40x4-interactive jobs, one stage per job
# (smoke -> train -> throughput). A stage whose job ends without its done file is retried once, then the chain stops.
REPO=/work/nvme/bifq/project/LanguageConditionHJ
OUT=$REPO/runs/residual/check_1109
LOG=/work/nvme/bifq/project/tmp/residual/chain_check.log
cd $REPO
declare -A tries
sq() { local o; o=$(squeue --me -h -n residual-check -o %i 2>/dev/null) || return 2; [[ -z "$o" || "$o" =~ ^[0-9[:space:]]+$ ]] || return 2; echo "$o"; }
while true; do
  next=""
  for st in smoke train throughput; do [ -f "$OUT/done.$st" ] || { next=$st; break; }; done
  [ -z "$next" ] && { echo "$(date '+%F %T') all stages done" >> $LOG; exit 0; }
  q=$(sq); rc=$?
  if [ $rc -ne 0 ]; then sleep 120; continue; fi
  if [ -n "$q" ]; then sleep 120; continue; fi
  t=${tries[$next]:-0}
  if [ "$t" -ge 2 ]; then echo "$(date '+%F %T') stage $next failed $t times: chain stops" >> $LOG; exit 1; fi
  tries[$next]=$((t+1))
  out=$(sbatch --partition=gpuA40x4-interactive --time=01:00:00 \
        --export=ALL,STAGES=$next,STAGE_TIMEOUT=3300 residual/jobs/check.sbatch 2>&1)
  echo "$(date '+%F %T') stage $next try $((t+1)): $out" >> $LOG
  sleep 120
done
