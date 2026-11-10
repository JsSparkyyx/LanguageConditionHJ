#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64g --time=01:00:00
#SBATCH --job-name=as-train
# One job of an action-sensitivity critic run (ACTION_SENSITIVITY_PLAN step 3; train_critic.py train).
# NAME (env): the run, A .. F. Resumable: the trainer stops itself before the time limit (exit 75) and the next job
# continues from its state. The partition comes from the command line (feeder.sh). EXTRA (env): more arguments.
#   sbatch --partition=gpuA100x4-interactive --export=ALL,NAME=E -J as-tr-E -o runs/lang/action_sens/logs/train_E_%j.out scripts/lang/action_sens/train.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
NAME=${NAME:?unset}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
budget=$(( ${left:-3600} - 420 ))
echo "[as-train] $NAME on $(hostname) $SLURM_JOB_PARTITION $(nvidia-smi --query-gpu=name --format=csv,noheader), budget $budget s"
scripts/lang/in_container.sh python scripts/lang/action_sens/train_critic.py train --name $NAME --resume \
  --time-budget $budget ${EXTRA:-}
rc=$?
echo "[as-train] $NAME rc=$rc $(date '+%F %T')"
[ $rc -eq 0 ] && [ -z "${EXTRA:-}" ] && date '+%F %T' > runs/lang/action_sens/runs/$NAME.DONE
exit $rc
