#!/bin/bash
# One collect_demos shard for one stage job (inside the container): relaunches the collector after each process
# recycle (exit 75 from --max-attempts) while the job has time left, so a 1 h job keeps collecting.
#   bash vla_finetuning/jobs/collect_loop.sh <collect_demos args...>
set -u
cd /work/nvme/bifq/project/LanguageConditionHJ
while true; do
  python -m vla_finetuning.collect_demos "$@"
  rc=$?
  [ $rc -eq 75 ] || exit $rc
  left=$(( ${STAGE_DEADLINE:-0} - $(date +%s) ))
  [ -n "${STAGE_DEADLINE:-}" ] && [ $left -lt 420 ] && exit 75
done
