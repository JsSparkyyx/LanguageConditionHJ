#!/bin/bash
# Submit the action-sensitivity experiment's jobs with LSF (bsub), e.g. WashU RIS compute1. The Slurm twin is
# submit.sh; both run the same job scripts (train.sbatch, collect.sbatch: plain bash, the name is historical).
#
#   cd <repo>
#   export HOLOBRAIN_CKPT=<dir>/HoloBrain_v0.0_GD
#   export PY=<python of the simulator environment>          # e.g. /path/to/conda/envs/RoboTwin/bin/python
#   export BSUB_FLAGS="<everything your one-GPU jobs need>"   # group, queue, container, cores, memory, ONE GPU
#       e.g.  BSUB_FLAGS="-G compute-<group> -q general -a 'docker(<image>)' -n 16 -R 'rusage[mem=64GB]' -gpu 'num=1:gmodel=<A40 model name>'"
#   bash scripts/lang/action_sens/portable/submit_lsf.sh collect [K]     K collection jobs side by side (default 1)
#   bash scripts/lang/action_sens/portable/submit_lsf.sh train D2 G B    one job per run
#   bash scripts/lang/action_sens/portable/submit_lsf.sh score [runs]    the scores of the finished runs
#   DRY=1 in front: write the job files and print the bsub lines, submit nothing.
#
# COLLECT_FLAGS / TRAIN_FLAGS replace BSUB_FLAGS for that kind (the collection needs a ray-tracing GPU such as an
# A40; the trainings and the scores run on any CUDA GPU).
# Each job is a small file under runs/lang/action_sens/logs/jobs/ that sets the variables itself (so nothing depends
# on what the scheduler or the container passes through), goes to the repo and runs the job script. The repo and its
# data must be visible at the SAME paths inside the job (with RIS containers: LSF_DOCKER_VOLUMES).
# Logs: runs/lang/action_sens/logs/<job name>_<job id>.out.
# The jobs have no time budget here: a training runs its 6,000 steps (about 80 min on an A40), a collection job runs
# until every episode is done. If a job is killed, submit the same line again: it resumes.
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
: "${HOLOBRAIN_CKPT:?set HOLOBRAIN_CKPT to the HoloBrain_v0.0_GD directory}"
J=runs/lang/action_sens/logs/jobs
mkdir -p "$J" runs/lang/action_sens/runs
kind=${1:?collect | train | score}; shift

job() {   # name flags command...   -> writes the job file, submits it
  local name=$1 flags=$2; shift 2
  local f="$REPO/$J/$name.$(date +%m%d_%H%M%S).sh" v
  {
    echo '#!/bin/bash'
    echo "cd $(printf %q "$REPO")"
    for v in HOLOBRAIN_CKPT PY HF_HOME LANG_TMP WANDB_MODE NAME ROUND N_EPISODES WORKERS EXTRA ALLOW_RASTER; do
      [ -n "${!v:-}" ] && printf 'export %s=%q\n' "$v" "${!v}"
    done
    echo "echo \"[lsf] $name on \$(hostname), job \${LSB_JOBID:-?}\""
    echo "$*"
  } > "$f"
  local line="bsub $flags -J $name -oo $REPO/runs/lang/action_sens/logs/${name}_%J.out /bin/bash $f"
  echo "$line"
  [ "${DRY:-0}" = 1 ] || eval "$line"
}

case $kind in
  collect)
    for k in $(seq ${1:-1}); do
      ROUND=${ROUND:-2} job "as-collect-$k" "${COLLECT_FLAGS:-${BSUB_FLAGS:-}}" bash scripts/lang/action_sens/portable/collect.sbatch
      sleep 1
    done ;;
  train)
    for r in "$@"; do
      NAME=$r job "as-tr-$r" "${TRAIN_FLAGS:-${BSUB_FLAGS:-}}" bash scripts/lang/action_sens/portable/train.sbatch
      sleep 1
    done ;;
  score)
    runs=${*:-A B C D F D2 G B2 H D3}
    job as-score "${TRAIN_FLAGS:-${BSUB_FLAGS:-}}" \
      "source scripts/lang/bridges/_env.sh && \$PY scripts/lang/action_sens/train_critic.py score --names $runs" ;;
  *) echo "collect | train | score"; exit 2 ;;
esac
