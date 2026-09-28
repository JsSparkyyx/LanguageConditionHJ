#!/usr/bin/env bash
# Interactive full-grid eval (no sbatch). Usage:
#   bash scripts/run_eval_full.sh <run_dir> [ckpt_name] [release_margin]
# Defaults: ckpt_name=checkpoint.pt, release_margin=0.
# Example:
#   bash scripts/run_eval_full.sh $TMP/cehj/runs/loo_franka
#   bash scripts/run_eval_full.sh $TMP/cehj/runs/loo_franka checkpoint_epoch140.pt 0.005
set -euo pipefail
RUN="${1:?usage: run_eval_full.sh <run_dir> [ckpt] [margin]}"
CKPT="${2:-checkpoint.pt}"
MARGIN="${3:-0}"
cd "$(dirname "$0")/.."
exec python main/train/eval_ckpt.py \
  --run "$RUN" \
  --ckpt "$CKPT" \
  --filter-release-margin "$MARGIN" \
  --episodes 50
