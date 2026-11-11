#!/bin/bash
# Submit the action-sensitivity experiment's jobs on a machine without Delta's queues (plan section 7).
#   cd <repo>; export HOLOBRAIN_CKPT=<dir> PY=<python> SBATCH_FLAGS="-A <account> -p <partition> --gpus=<type>:1"
#   scripts/lang/action_sens/portable/submit.sh collect [K]     K collection jobs side by side (default 1), ROUND=2
#   scripts/lang/action_sens/portable/submit.sh train D2 G B    one training job per run (needs no simulator)
#   scripts/lang/action_sens/portable/submit.sh score [runs]    the scores of the finished runs -> scores.json
# Every job asks for what SBATCH_FLAGS says (one GPU). COLLECT_FLAGS / TRAIN_FLAGS replace it for that kind, e.g. a
# ray-tracing GPU type for the collection and any GPU for the trainings.
# Order: `train D2 G B` can start as soon as `transfer.sh ... train` has run; `train B2 H D3` after the collection
# has written data/lang_asens2/COLLECT_DONE; `score` last.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
mkdir -p runs/lang/action_sens/logs
kind=${1:?collect | train | score}; shift
case $kind in
  collect)
    for _ in $(seq ${1:-1}); do
      ROUND=${ROUND:-2} sbatch ${COLLECT_FLAGS:-${SBATCH_FLAGS:-}} --export=ALL scripts/lang/action_sens/portable/collect.sbatch
      sleep 2
    done ;;
  train)
    for r in "$@"; do
      NAME=$r sbatch ${TRAIN_FLAGS:-${SBATCH_FLAGS:-}} --export=ALL -J "as-tr-$r" scripts/lang/action_sens/portable/train.sbatch
    done ;;
  score)
    runs=${*:-A B C D F D2 G B2 H D3}
    sbatch ${TRAIN_FLAGS:-${SBATCH_FLAGS:-}} --export=ALL -J as-score --cpus-per-task=16 --time=01:00:00 \
      -o runs/lang/action_sens/logs/score_%j.out \
      --wrap "source scripts/lang/bridges/_env.sh && \$PY scripts/lang/action_sens/train_critic.py score --names $runs" ;;
  *) echo "collect | train | score"; exit 2 ;;
esac
