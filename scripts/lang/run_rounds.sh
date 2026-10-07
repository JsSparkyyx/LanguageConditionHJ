#!/bin/bash
# Relaunch rounds training until it completes (exit 75 = init limit / time budget -> --resume).
#   bash scripts/lang/run_rounds.sh <train rounds args...>   (inside the container)
set -u
while true; do
  python -m main.lang.train rounds "$@" --resume
  rc=$?
  [ $rc -eq 75 ] || exit $rc
  echo "[run_rounds] exit 75: relaunching with --resume"
done
