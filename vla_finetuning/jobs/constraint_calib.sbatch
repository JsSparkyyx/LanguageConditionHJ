#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-calib -o /work/nvme/bifq/project/tmp/logs/vf_calib_%j.out
# Calibration of the constraint fine-tune's careless knobs (items/calibrate.py, Yuxuan 11-03): T6 on 3 processes, T7
# on 1, one GPU; resumable (resubmit until every run is done), then the summary.
#   sbatch vla_finetuning/jobs/constraint_calib.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
OUT=${OUT:-$R/vla_finetuning/results_constraint/calib}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 180 ))
mkdir -p $OUT/_logs
run() { scripts/lang/in_container.sh python -m vla_finetuning.items.calibrate --task $1 --out $OUT --slot $2 --slots $3 \
          >> $OUT/_logs/${1}_s$2.log 2>&1; echo "[calib] $1 s$2 rc=$?"; }
run clean_pour 0 3 & sleep 20
run clean_pour 1 3 & sleep 20
run clean_pour 2 3 & sleep 20
run gap_place 0 1 &
wait
scripts/lang/in_container.sh python -m vla_finetuning.items.calibrate --summary --out $OUT
