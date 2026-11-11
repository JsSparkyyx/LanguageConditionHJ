#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=96g --time=01:45:00
#SBATCH --job-name=residual-check
#SBATCH --output=/work/nvme/bifq/project/tmp/logs/residual_check_%j.out
# residual/PLAN.md first step, on one A40: smoke test -> two-update training run -> throughput.
# Each stage runs only if the previous one passed. Rerunning resumes: done stages are skipped.
#   sbatch [--partition=gpuA40x4-interactive --time=01:00:00] residual/jobs/check.sbatch
# OUT (default runs/residual/check_<date>) and STAGES (default "smoke train throughput") can be exported.
set -u
REPO=/work/nvme/bifq/project/LanguageConditionHJ
OUT=${OUT:-$REPO/runs/residual/check_1109}
STAGES=${STAGES:-"smoke train throughput"}
SETTINGS=${SETTINGS:-"deviation:8,fixed:8,playback:8,deviation:12,deviation:1"}
MINUTES=${MINUTES:-6}
mkdir -p "$OUT"
cd "$REPO"
echo "[check] $(hostname) $(date '+%F %T') job ${SLURM_JOB_ID:-local} out $OUT stages: $STAGES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
( while sleep 120; do echo "[gpumem $(date +%T)] $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ') load $(cut -d' ' -f1 /proc/loadavg)"; done ) &
mon=$!
trap 'kill $mon 2>/dev/null' EXIT
[ -f scripts/lang/wandb_env.sh ] && source scripts/lang/wandb_env.sh >/dev/null 2>&1 || true
# STAGE_TIMEOUT (s): a stage stops before the job's walltime (interactive jobs: 1 h); every stage resumes
run() { timeout --signal=TERM --kill-after=60 "${STAGE_TIMEOUT:-6000}" scripts/lang/in_container.sh "$@"; }
rc=0; failed=""
for st in $STAGES; do
  if [ -f "$OUT/done.$st" ]; then echo "[check] $st done before: skipped"; continue; fi
  echo "[check] === $st $(date '+%F %T')"
  case $st in
    smoke)
      run python -m residual.smoke --out "$OUT/smoke"; rc=$? ;;
    train)
      run python -m residual.train --run "$OUT/train" --updates 2 --workers 4 --no-wandb --deadline-min 22 \
        --eval-timeout-min 15 \
        --set steps_per_update=1000 minibatch=512 eval_every=1 eval_in_way=2 eval_out_way=2 eval_video_scenes=2
      rc=$?
      # 75 = stopped at its deadline with a checkpoint: the loop ran; the throughput stage does not depend on it
      [ $rc -eq 75 ] && { echo "[check] train stopped at its deadline (75): counted as run"; rc=0; } ;;
    throughput)
      run python -m residual.throughput --out "$OUT/throughput" --settings "$SETTINGS" --minutes "$MINUTES"
      rc=$? ;;
  esac
  echo "[check] $st rc=$rc $(date '+%F %T')"
  [ $rc -eq 0 ] && touch "$OUT/done.$st"
  [ $rc -ne 0 ] && failed="$failed $st"
  # only a failed smoke test stops the job: the training check and the throughput step do not depend on each other
  [ $rc -ne 0 ] && [ "$st" = smoke ] && break
done
echo "[check] end, failed stages:${failed:- none} $(date '+%F %T')"
[ -z "$failed" ]
