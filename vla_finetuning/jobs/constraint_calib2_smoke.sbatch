#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-calib2 -o /work/nvme/bifq/project/tmp/logs/vf_calib2_%j.out
# T6 calibration round 2 (2 processes) + the collector smoke test on T1 / T3 / T5 / T7 (one process each, RT cameras).
#   sbatch vla_finetuning/jobs/constraint_calib2_smoke.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
OUT=$R/vla_finetuning/results_constraint/calib; ROOT=$R/vla_finetuning/data/smoke_constraint
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 180 ))
mkdir -p $OUT/_logs ${ROOT}_logs
for k in 0 1; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.calibrate --task clean_pour --round 2 --out $OUT \
    --slot $k --slots 2 >> $OUT/_logs/clean_pour_r2_s$k.log 2>&1 &
  sleep 20
done
for t in selective_avoid push_zone tray_transport gap_place; do
  host=""; [ "$t" = selective_avoid ] && host="--host place_container_plate"
  i=$(python3 -c "print(['selective_avoid','push_zone','tray_transport','clean_pour','gap_place'].index('$t'))")
  base=$(( 500000 + 10000 * i + 9000 ))
  scripts/lang/in_container.sh python -m vla_finetuning.items.collect_constraint --task $t $host --root $ROOT \
    --shard smoke --seed-start $base --seed-stop $((base + 40)) --target 1 --max-runs 8 --videos 2 \
    >> ${ROOT}_logs/smoke_$t.log 2>&1 &
  sleep 20
done
wait
scripts/lang/in_container.sh python -m vla_finetuning.items.calibrate --summary --out $OUT
