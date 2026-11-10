#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1-4 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=as-collect -o /work/nvme/bifq/project/LanguageConditionHJ/runs/lang/action_sens/logs/collect_%j.out
# The boundary branch collection (ACTION_SENSITIVITY_PLAN step 2; collect_branches.py): 1 to 4 A40 nodes, WORKERS
# processes per GPU (default 2; a worker holds the HoloBrain encoder and a ray-traced simulator). Resumable: a
# finished planner episode is skipped, an unfinished one leaves no rows; the workers stop before the time limit.
# Exit 0 = every episode done, 75 = run again (feeder.sh resubmits).
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + ${left:-3600} - 240 ))
echo "[as-collect] job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST ($SLURM_NNODES), deadline $(date -d @$STAGE_DEADLINE +%T)"
srun --ntasks-per-node=1 --kill-on-bad-exit=0 scripts/lang/in_container.sh \
  python scripts/lang/action_sens/collect_branches.py run --workers ${WORKERS:-2} ${EXTRA:-}
rc=$?
left=$(ls data/lang_asens/episodes/*/result.json 2>/dev/null | wc -l)
echo "[as-collect] rc=$rc, $left episodes done $(date '+%F %T')"
if [ "$left" -ge "${N_EPISODES:-25}" ]; then
  scripts/lang/in_container.sh python scripts/lang/action_sens/collect_branches.py table && date '+%F %T' > data/lang_asens/COLLECT_DONE
  exit 0
fi
exit 75
