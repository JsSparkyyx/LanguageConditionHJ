#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=96g --time=01:00:00
#SBATCH --job-name=residual-train
#SBATCH --output=/work/nvme/bifq/project/tmp/logs/residual_train_%j.out
# One 1-hour piece of a PPO run (interactive partition, Yuxuan 11-12): the learner resumes from the run's checkpoint,
# stops at DEADLINE minutes with a checkpoint (exit 75) and the feeder (residual/jobs/chain_train.sh) submits the next
# piece. Exit 0 = all UPDATES done: the script writes <run>/done. SETS applies to the first piece only (a resumed run
# keeps its config.json).
#   sbatch --export=ALL,RUN=residual_dev_success_s025_1112,UPDATES=350 residual/jobs/train_chunk.sbatch
set -u
REPO=/work/nvme/bifq/project/LanguageConditionHJ
RUN=${RUN:?unset}; WPG=${WPG:-8}; UPDATES=${UPDATES:-350}; DEADLINE=${DEADLINE:-50}
SETS=${SETS:-"replan_trigger=\"deviation\" completion=\"success\" init_std=0.025"}
cd "$REPO"
source scripts/lang/wandb_env.sh >/dev/null 2>&1 || true
echo "[train-chunk] $RUN on $(hostname) job ${SLURM_JOB_ID:-local} $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
timeout --signal=TERM --kill-after=60 57m scripts/lang/in_container.sh bash -c \
  "python -m residual.train --run $REPO/runs/residual/$RUN --updates $UPDATES --workers-per-gpu $WPG \
   --deadline-min $DEADLINE --set $SETS"
rc=$?
echo "[train-chunk] rc=$rc $(date '+%F %T')"
[ $rc -eq 0 ] && touch "$REPO/runs/residual/$RUN/done"
# 75 = stopped at its deadline with a checkpoint: a normal piece
[ $rc -eq 75 ] && exit 0
exit $rc
