#!/bin/bash
# Sliding-tray demo collection as 1-hour pieces on gpuA40x4-interactive (the long A40 queue held jobs for days,
# 2026-11-15): each piece runs all four shards (SLOTS=4) on one A40 and resumes where the last stopped, until every
# shard is DONE or MAX_PIECES pieces ran. A refused sbatch (maintenance, limit) is retried after 5 minutes.
# Start it on a login node over ssh, detached:
#   ssh dt-login01 'cd /work/nvme/bifq/project/LanguageConditionHJ && setsid nohup bash vla_finetuning/jobs/tray_feeder.sh \
#     > /dev/null 2>> /work/nvme/bifq/project/tmp/logs/tray_feeder.err < /dev/null &'
R=/work/nvme/bifq/project/LanguageConditionHJ
ROOT=${ROOT:-$R/vla_finetuning/data/demos_tray_slide}
LOG=/work/nvme/bifq/project/tmp/logs/tray_feeder.log
MAX_PIECES=${MAX_PIECES:-8}
cd $R
shards=${SHARDS:-"s0:25:700000:4:710000 s1:25:700001:4:710000 s2:25:700002:4:710000 s3:25:700003:4:710000"}  # collect_tray --shards
echo "== $(date '+%F %T') feeder start, shards: $shards" >> $LOG
piece=0
while [ $piece -lt $MAX_PIECES ]; do
  left=0
  for sh in $shards; do [ -f ${ROOT}_logs/tray_slide/${sh%%:*}/DONE ] || left=$((left + 1)); done
  [ $left -eq 0 ] && { echo "$(date '+%F %T') all shards DONE" >> $LOG; exit 0; }
  out=$(SHARDS="$shards" SLOTS=4 ROOT=$ROOT sbatch --export=ALL --parsable --partition=gpuA40x4-interactive \
        --time=01:00:00 --mem=96g vla_finetuning/jobs/tray_collect.sbatch 2>&1)
  if ! [[ "$out" =~ ^[0-9]+$ ]]; then
    echo "$(date '+%F %T') sbatch refused: $out" >> $LOG; sleep 300; continue
  fi
  piece=$((piece + 1))
  echo "$(date '+%F %T') piece $piece job $out ($left shards left)" >> $LOG
  while squeue -h -j $out 2>/dev/null | grep -q .; do sleep 60; done
  echo "$(date '+%F %T') piece $piece job $out ended: $(sacct -j $out -X -n -o State,Elapsed | tr -s ' ')" >> $LOG
done
echo "$(date '+%F %T') stopped after $MAX_PIECES pieces" >> $LOG
