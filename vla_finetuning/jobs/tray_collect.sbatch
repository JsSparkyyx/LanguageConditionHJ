#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4,gpuA40x4-preempt --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=08:00:00
#SBATCH --job-name=tray-collect -o /work/nvme/bifq/project/tmp/logs/tray_collect_%j.out
# Demo collection of the sliding-tray task (items/collect_tray.py, Yuxuan 2026-11-13): A40 only (the cameras need
# ray tracing; A100 / H200 have no RT cores). SHARDS (env): the shards this job runs, "name:target:start:step:stop"
# each, SLOTS at a time on its one GPU; each shard is relaunched after an exit 75 (process recycling, a renderer error)
# until it is done (exit 0 -> <log>/DONE) or the deadline; a planner / code error (exit 3) leaves it for a look. A
# collector that writes nothing for HANG_S is killed and relaunched (at most 3 times). Resubmitting the same job resumes.
#   SHARDS="s0:25:700000:4:710000 s1:25:700001:4:710000" sbatch vla_finetuning/jobs/tray_collect.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
ROOT=${ROOT:-$R/vla_finetuning/data/demos_tray_slide}; SLOTS=${SLOTS:-2}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 240 ))
mkdir -p ${ROOT}_logs/_pool
echo "[tray] root $ROOT job $SLURM_JOB_ID on $(hostname -s) $(nvidia-smi --query-gpu=name --format=csv,noheader), shards: $SHARDS"
slot() {
  local name target start step stop log rc pid killed last m now hung=0 crashes=0
  IFS=: read -r name target start step stop <<< "$1"
  log=${ROOT}_logs/_pool/$name.log
  while [ ! -f ${ROOT}_logs/tray_slide/$name/DONE ]; do
    [ $(( STAGE_DEADLINE - $(date +%s) )) -lt 420 ] && return 0
    setsid scripts/lang/in_container.sh python -m vla_finetuning.items.collect_tray --root $ROOT --shard $name \
      --target $target --seed-start $start --seed-step $step --seed-stop $stop --max-runs ${MAX_RUNS:-10} \
      < /dev/null >> $log 2>&1 &
    pid=$!; killed=0
    while kill -0 $pid 2> /dev/null; do
      sleep 30
      now=$(date +%s); last=$(stat -c %Y $log)
      for m in $(stat -c %Y ${ROOT}_logs/tray_slide/$name/runs.jsonl 2> /dev/null); do [ $m -gt $last ] && last=$m; done
      if [ $(( now - last )) -gt ${HANG_S:-900} ]; then
        echo "[tray-pool] $(date '+%T') $name: no output for $(( now - last ))s, killing $pid" >> $log
        kill -9 -- -$pid 2> /dev/null || kill -9 $pid 2> /dev/null; killed=1; break
      fi
    done
    wait $pid; rc=$?
    echo "[tray-pool] $(date '+%F %T') $name rc=$rc" >> $log
    if [ $rc -eq 0 ]; then date '+%F %T' > ${ROOT}_logs/tray_slide/$name/DONE; break; fi
    if [ $killed -eq 1 ]; then hung=$((hung + 1)); [ $hung -ge 3 ] && break; continue; fi
    if [ $rc -ge 128 ]; then crashes=$((crashes + 1)); [ $crashes -ge 3 ] && break; continue; fi
    [ $rc -ne 75 ] && { echo "[tray-pool] $name rc=$rc: left for a look" >> $log; break; }
  done
}
run=0
for sh in $SHARDS; do
  while [ $(jobs -rp | wc -l) -ge $SLOTS ]; do sleep 20; done
  slot "$sh" & run=$((run + 1)); sleep 30
done
wait
echo "[tray] job done $(date '+%F %T')"
