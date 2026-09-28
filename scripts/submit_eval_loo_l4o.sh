#!/usr/bin/env bash
# Submit eval jobs for all LOO, L4O and LOO-vanilla runs:
# latest 3 ckpts each, release margin 0, 50 eps per (task, embodiment),
# one job per (run, ckpt). 15 runs x 3 ckpts = 45 jobs.
# Usage: bash scripts/submit_eval_loo_l4o.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=${TMP:-/work/nvme/bifq/project/tmp}
mkdir -p "$TMP/cehj/slurm"

SUBMIT="--submit"
if [ "${1:-}" = "--dry-run" ]; then
  SUBMIT=""
fi

runs=()
for short in piper franka arx ur5 aloha; do
  runs+=("loo_$short" "l4o_$short" "loo_vanilla_$short")
done

for name in "${runs[@]}"; do
  run_dir="$TMP/cehj/runs/$name"
  if [ ! -f "$run_dir/checkpoint.pt" ]; then
    echo "[skip] $run_dir (no checkpoint.pt)" >&2
    continue
  fi
  echo "== $name =="
  # shellcheck disable=SC2086
  python scripts/eval_all.py "$run_dir" \
    --last-k 3 --release-margins 0 --episodes 50 --shard none $SUBMIT
done
