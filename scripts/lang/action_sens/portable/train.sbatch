#!/bin/bash
#SBATCH --job-name=as-train
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=runs/lang/action_sens/logs/train_%x_%j.out
# One critic run of the action-sensitivity experiment on a machine without Delta's container (train_critic.py train).
#   cd <repo>; export HOLOBRAIN_CKPT=<dir> PY=<python> SBATCH_FLAGS="-A <account> -p <partition> --gpus=<type>:1"
#   NAME=D2 sbatch $SBATCH_FLAGS --export=ALL -J as-tr-D2 scripts/lang/action_sens/portable/train.sbatch
# or through submit.sh. One GPU (the runs took 0.7-0.8 s per step on an A40 / A100-40GB: 6,000 steps = ~80 min).
# No simulator: any CUDA GPU works. Resumable: the trainer stops before the time limit (exit 75) and, unless
# RESUBMIT=0, the job submits itself again. Also runs without Slurm: NAME=D2 bash .../train.sbatch
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/lang/bridges/_env.sh
NAME=${NAME:?set NAME to a run of train_critic.py RUNS}
mkdir -p runs/lang/action_sens/logs runs/lang/action_sens/runs
budget=()
if [ -n "${SLURM_JOB_ID:-}" ]; then
  left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{s=0; for(i=1;i<=NF;i++) s=s*(i==2&&NF==4?24:60)+$i; print s}')
  budget=(--time-budget $(( ${left:-10800} - 420 )))
fi
echo "[as-train] $NAME on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1) ${budget[*]:-no time budget}"
$PY scripts/lang/action_sens/train_critic.py train --name "$NAME" --resume "${budget[@]}" ${EXTRA:-}
rc=$?
echo "[as-train] $NAME rc=$rc $(date '+%F %T')"
[ $rc -eq 0 ] && [ -z "${EXTRA:-}" ] && date '+%F %T' > runs/lang/action_sens/runs/$NAME.DONE
if [ $rc -eq 75 ] && [ "${RESUBMIT:-1}" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  sbatch ${SBATCH_FLAGS:-} --export=ALL -J "as-tr-$NAME" scripts/lang/action_sens/portable/train.sbatch
fi
exit $rc
