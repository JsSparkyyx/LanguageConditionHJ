#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-csmoke -o /work/nvme/bifq/project/tmp/logs/vf_csmoke_%j.out
# Smoke test of collect_constraint.py (logic only: on an A100 the cameras rasterize, so the data is thrown away).
#   TASKS="selective_avoid push_zone tray_transport gap_place" sbatch vla_finetuning/jobs/constraint_smoke.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
ROOT=${ROOT:-$R/vla_finetuning/data/smoke_constraint}
TASKS=${TASKS:-"selective_avoid push_zone tray_transport gap_place"}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 180 ))
mkdir -p ${ROOT}_logs
i=0
for t in $TASKS; do
  host=""; [ "$t" = selective_avoid ] && host="--host place_container_plate"
  base=$(( 500000 + 10000 * $(python3 -c "print(['selective_avoid','push_zone','tray_transport','clean_pour','gap_place'].index('$t'))") + 9000 ))
  scripts/lang/in_container.sh python -m vla_finetuning.items.collect_constraint --task $t $host --root $ROOT \
    --shard smoke --seed-start $base --seed-stop $((base + 40)) --target 1 --max-runs 8 --videos 2 \
    > ${ROOT}_logs/smoke_$t.log 2>&1 &
  i=$((i + 1)); sleep 30
done
wait
echo done
