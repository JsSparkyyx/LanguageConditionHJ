#!/usr/bin/env bash
# Merge eval shards -> combined summary.json for every LOO, L4O and
# LOO-vanilla run (every eval_full/<ckpt>[/t<budget>] dir). Run after the
# per-embodiment shard jobs finish. Also covers the nominal baseline dir.
# Usage: bash scripts/merge_eval_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=${TMP:-/work/nvme/bifq/project/tmp}

for family in loo l4o loo_vanilla; do
  for short in piper franka arx ur5 aloha; do
    python scripts/eval_all.py --merge "$TMP/cehj/runs/${family}_${short}"
  done
done
python scripts/eval_all.py --merge "$TMP/cehj/runs/nominal" || true
