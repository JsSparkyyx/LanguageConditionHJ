#!/usr/bin/env bash
# Submit eval jobs for the RECENT ckpt (checkpoint.pt) of every LOO, L4O
# and LOO-vanilla run: release margin 0, 50 eps per (task, embodiment),
# one job per run. 15 runs x 1 ckpt = 15 jobs.
# Usage: bash scripts/submit_eval_recent.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=${TMP:-/work/nvme/bifq/project/tmp}
mkdir -p "$TMP/cehj/slurm"

SUBMIT="--submit"
if [ "${1:-}" = "--dry-run" ]; then
  SUBMIT=""
fi

for family in loo l4o loo_vanilla; do
  for short in piper franka arx ur5 aloha; do
    run_dir="$TMP/cehj/runs/${family}_${short}"
    if [ ! -f "$run_dir/checkpoint.pt" ]; then
      echo "[skip] $run_dir (no checkpoint.pt)" >&2
      continue
    fi
    echo "== ${family}_${short} =="
    # shellcheck disable=SC2086
    python scripts/eval_all.py "$run_dir" \
      --last-k 1 --release-margins 0 --episodes 50 --video-every 10 \
      --shard none $SUBMIT
  done
done
