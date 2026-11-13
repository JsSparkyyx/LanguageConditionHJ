#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA100x8-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64g --time=01:00:00
#SBATCH --job-name=residual-tray-check
#SBATCH --output=/work/nvme/bifq/project/tmp/logs/residual_tray_check_%j.out
# residual/PLAN.md section 8: the sliding tray (tray_slide) on one GPU, no cameras: smoke test -> two-update training.
#   sbatch [--export=ALL,LANG_REPO=<checkout>,OUT=<dir>] residual/jobs/tray_check.sbatch
# LANG_REPO: the checkout whose code runs (default the main one); rerunning skips the stages done before.
set -u
REPO=${LANG_REPO:-/work/nvme/bifq/project/LanguageConditionHJ}
export LANG_REPO=$REPO
OUT=${OUT:-/work/nvme/bifq/project/tmp/residual/tray_check}
STAGES=${STAGES:-"smoke train"}
mkdir -p "$OUT"
cd "$REPO"
echo "[tray-check] $(hostname) $(date '+%F %T') job ${SLURM_JOB_ID:-local} code $REPO out $OUT stages: $STAGES"
echo "[tray-check] commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
run() { timeout --signal=TERM --kill-after=60 "${STAGE_TIMEOUT:-2400}" scripts/lang/in_container.sh "$@"; }
rc=0; failed=""
for st in $STAGES; do
  if [ -f "$OUT/done.$st" ]; then echo "[tray-check] $st done before: skipped"; continue; fi
  echo "[tray-check] === $st $(date '+%F %T')"
  case $st in
    smoke)
      run python -m residual.smoke_tray --out "$OUT/smoke" --n-seeds 3; rc=$? ;;
    train)
      run python -m residual.train --run "$OUT/train" --updates 2 --workers 4 --no-wandb --deadline-min 22 \
        --eval-timeout-min 10 \
        --set task=tray_slide steps_per_update=1000 minibatch=512 eval_every=1 eval_pairs=2 eval_video_scenes=0
      rc=$?
      [ $rc -eq 75 ] && { echo "[tray-check] train stopped at its deadline (75): counted as run"; rc=0; } ;;
  esac
  echo "[tray-check] $st rc=$rc $(date '+%F %T')"
  [ $rc -eq 0 ] && touch "$OUT/done.$st"
  [ $rc -ne 0 ] && failed="$failed $st"
done
grep -l "CuRobo init failed" "$OUT"/train/logs/*.log 2>/dev/null | head -3
echo "[tray-check] end, failed stages:${failed:- none} $(date '+%F %T')"
[ -z "$failed" ]
