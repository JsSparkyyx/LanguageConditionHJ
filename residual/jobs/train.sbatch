#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=96g --time=48:00:00
#SBATCH --requeue --signal=B:USR1@300
#SBATCH --job-name=residual-train
#SBATCH --output=/work/nvme/bifq/project/tmp/logs/residual_train_%j.out
# PPO training of the residual policy (residual/PLAN.md). Resumable: the learner checkpoints every update, exits 75
# at its deadline or on a signal, and this script relaunches it on 75; USR1 (300 s before the walltime) requeues the
# job. One GPU per job (Yuxuan 11-10). Throughput 11-12, one A40, deviation trigger, delta = 0: 8 workers 208
# ticks/s (~135 policy transitions/s), 12 workers 233 ticks/s (CPU-bound on 16 cores). Not to be submitted before
# the budget is agreed with Yuxuan.
#   sbatch --export=ALL,RUN=residual_dev_success_v1,WPG=8,UPDATES=2000 residual/jobs/train.sbatch
#   (first run: replan_trigger "deviation", completion "success"; SETS passes other --set overrides)
set -u
REPO=/work/nvme/bifq/project/LanguageConditionHJ
RUN=${RUN:?unset}; WPG=${WPG:-8}; UPDATES=${UPDATES:-2000}
SETS=${SETS:-"replan_trigger=\"deviation\" completion=\"success\""}
cd "$REPO"
source scripts/lang/wandb_env.sh
trap 'echo "[train] USR1: near walltime, requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
echo "[train] $RUN on $(hostname) restart ${SLURM_RESTART_COUNT:-0} $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
while true; do
  scripts/lang/in_container.sh bash -c "python -m residual.train --run $REPO/runs/residual/$RUN --updates $UPDATES \
      --workers-per-gpu $WPG --set $SETS" &
  wait $!; rc=$?
  [ $rc -eq 75 ] || break
  echo "[train] exit 75: relaunching $(date '+%F %T')"
done
echo "[train] $RUN rc=$rc $(date '+%F %T')"; exit $rc
