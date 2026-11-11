#!/bin/bash
#SBATCH --job-name=as-collect
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=runs/lang/action_sens/logs/collect_%j.out
# The boundary branch collection (collect_branches.py) on a machine without Delta's container.
#   cd <repo>; export HOLOBRAIN_CKPT=<dir> PY=<python> SBATCH_FLAGS="-A <account> -p <partition> --gpus=<rtx type>:1"
#   ROUND=2 sbatch $SBATCH_FLAGS --export=ALL scripts/lang/action_sens/portable/collect.sbatch      (or submit.sh)
# ONE GPU per job, WORKERS processes on it (default 2; a second worker adds about 20%). The GPU must render with
# ray tracing (RTX class: A40, L40S, RTX 6000 ...): on a V100 / A100 / H100 the simulator rasterizes and the images,
# so the tokens, differ from every other shard (ALLOW_RASTER=1 overrides). Needs RoboTwin next to the repo and the
# simulator environment (NEW_MACHINE_SETUP.md). Several jobs at once are fine: the workers claim episodes by lock
# files, and a finished episode (episodes/<name>/result.json, also one copied from Delta) is skipped.
# A worker's shard holds about 14 planner episodes, then it exits 75 for a fresh one: the loop below starts it
# again until every episode is done or the time is nearly up (then, unless RESUBMIT=0, the job resubmits itself).
# About 7.5 min per planner episode per worker on an A40 (4 states x 9 decisions).
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/lang/bridges/_env.sh
ROUND=${ROUND:-2}
D=data/lang_asens; N=${N_EPISODES:-25}
[ "$ROUND" = 2 ] && { D=data/lang_asens2; N=${N_EPISODES:-125}; }
mkdir -p runs/lang/action_sens/logs
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "[as-collect] round $ROUND on $(hostname) $gpu"
case "$gpu" in *V100*|*A100*|*H100*|*H200*|*P100*|*T4*)
  [ "${ALLOW_RASTER:-0}" = 1 ] || { echo "[as-collect] $gpu has no ray tracing: refusing (ALLOW_RASTER=1 overrides)"; exit 2; };;
esac
if [ -n "${SLURM_JOB_ID:-}" ]; then
  left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{s=0; for(i=1;i<=NF;i++) s=s*(i==2&&NF==4?24:60)+$i; print s}')
  export STAGE_DEADLINE=$(( $(date +%s) + ${left:-28800} - 300 ))
fi
rc=75
while [ $rc -eq 75 ]; do
  $PY scripts/lang/action_sens/collect_branches.py run --round $ROUND --workers ${WORKERS:-2} ${EXTRA:-}
  rc=$?
  n=$(ls $D/episodes/*/result.json 2>/dev/null | wc -l)
  echo "[as-collect] rc=$rc, $n of $N episodes done $(date '+%F %T')"
  [ "$n" -ge "$N" ] && break
  [ -n "${STAGE_DEADLINE:-}" ] && [ $(( $(date +%s) + 1800 )) -gt "$STAGE_DEADLINE" ] && break
done
if [ "$n" -ge "$N" ]; then
  $PY scripts/lang/action_sens/collect_branches.py table --round $ROUND && date '+%F %T' > $D/COLLECT_DONE
  exit 0
fi
if [ $rc -eq 75 ] && [ "${RESUBMIT:-1}" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  sbatch ${SBATCH_FLAGS:-} --export=ALL scripts/lang/action_sens/portable/collect.sbatch
fi
exit $rc
