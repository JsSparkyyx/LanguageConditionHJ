#!/bin/bash
# One collect_items shard for one stage job (inside the container), like jobs/collect_loop.sh: relaunches the collector
# after each exit 75 (process recycling after --max-attempts, or a renderer error: a broken renderer stays broken
# in-process) while the job has time left. A collector killed by a signal (rc >= 128, e.g. a renderer abort; its
# unfinished seed counts as tried) is relaunched too, up to 3 times. Five quick exits in a row (each under 240 s: the
# renderer fails at once on this node) or the third crash end the loop with 75, leaving the shard to a later job.
#   bash vla_finetuning/items/jobs/collect_items_loop.sh <collect_items args...>
set -u
cd "$(dirname "$(readlink -f "$0")")/../../.." || exit 1
quick=0
crashes=0
while true; do
  t0=$(date +%s)
  python -m vla_finetuning.items.collect_items "$@"
  rc=$?
  if [ $rc -ge 128 ]; then
    crashes=$((crashes + 1))
    echo "[collect_items_loop] collector killed by signal $((rc - 128)) (crash $crashes of 3)"
    [ $crashes -ge 3 ] && exit 75
  elif [ $rc -ne 75 ]; then
    exit $rc
  fi
  if [ $(( $(date +%s) - t0 )) -lt 240 ]; then quick=$((quick + 1)); else quick=0; fi
  if [ $quick -ge 5 ]; then
    echo "[collect_items_loop] 5 quick exits in a row: leaving this shard to a later job"
    exit 75
  fi
  left=$(( ${STAGE_DEADLINE:-0} - $(date +%s) ))
  [ -n "${STAGE_DEADLINE:-}" ] && [ $left -lt 420 ] && exit 75
done
