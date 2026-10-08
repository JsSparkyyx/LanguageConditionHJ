#!/bin/bash
# Relaunch rounds training until it completes (exit 75 = init limit / time budget -> --resume).
#   bash scripts/lang/run_rounds.sh <train rounds args...>   (inside the container)
set -u
while true; do
  python -m main.lang.train rounds "$@" --resume
  rc=$?
  [ $rc -eq 75 ] || exit $rc
  # a time-budget stop: leave the rest to the next job (the chain resubmits on 75); relaunching here would
  # only exit 75 again at once, until the job's time limit
  if [ -n "${STAGE_DEADLINE:-}" ] && [ "$(date +%s)" -gt $(( ${STAGE_DEADLINE%.*} - 900 )) ]; then
    echo "[run_rounds] exit 75 near the job's deadline: leaving the rest to the next job"
    exit 75
  fi
  echo "[run_rounds] exit 75 (init limit): relaunching with --resume"
done
