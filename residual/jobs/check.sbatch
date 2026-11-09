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
run() { scripts/lang/in_container.sh "$@"; }
rc=0
for st in $STAGES; do
  if [ -f "$OUT/done.$st" ]; then echo "[check] $st done before: skipped"; continue; fi
  echo "[check] === $st $(date '+%F %T')"
  case $st in
    smoke)
      run python -m residual.smoke --out "$OUT/smoke"; rc=$? ;;
    train)
      run python -m residual.train --run "$OUT/train" --updates 2 --workers 4 --no-wandb --deadline-min 25 \
        --set steps_per_update=1500 minibatch=512 eval_every=1 eval_in_way=2 eval_out_way=2 eval_video_scenes=2
      rc=$? ;;
    throughput)
      run python -m residual.throughput --out "$OUT/throughput" --settings "$SETTINGS" --minutes "$MINUTES"
      rc=$? ;;
  esac
  echo "[check] $st rc=$rc $(date '+%F %T')"
  [ $rc -eq 0 ] && touch "$OUT/done.$st"
  [ $rc -ne 0 ] && break
done
echo "[check] end rc=$rc $(date '+%F %T')"
exit $rc
