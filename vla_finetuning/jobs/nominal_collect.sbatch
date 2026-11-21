#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=96g --time=01:00:00
#SBATCH --job-name=nominal-collect -o /work/nvme/bifq/project/tmp/logs/nominal_collect_%j.out
# Careless demos of the safety-agnostic VLA (items/collect_nominal.py, Yuxuan 2026-11-21): T6 careless pours and T8
# careless tray drags, on one A40 (the cameras need ray tracing). SHARDS (env): "task:shard:target:start:step:stop"
# each (default: collect_nominal --shards), SLOTS at a time; a shard is relaunched after an exit 75 until it is done
# (exit 0 -> <log>/DONE) or the deadline; exit 3 leaves it for a look. A collector that writes nothing for HANG_S is
# killed and relaunched (at most 3 times). Resubmitting resumes. Submitted by nominal_feeder.sh.
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
ROOT=${ROOT:-$R/vla_finetuning/data/demos_nominal}; SLOTS=${SLOTS:-4}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 240 ))
SHARDS=${SHARDS:-$(scripts/lang/in_container.sh python -m vla_finetuning.items.collect_nominal --shards)}
mkdir -p ${ROOT}_logs/_pool
echo "[nominal] root $ROOT job $SLURM_JOB_ID on $(hostname -s) $(nvidia-smi --query-gpu=name --format=csv,noheader), shards: $SHARDS"
slot() {
  local task name target start step stop log rc pid killed last m now hung=0 crashes=0
  IFS=: read -r task name target start step stop <<< "$1"
  log=${ROOT}_logs/_pool/${task}_$name.log
  while [ ! -f ${ROOT}_logs/$task/$name/DONE ]; do
    [ $(( STAGE_DEADLINE - $(date +%s) )) -lt 420 ] && return 0
    setsid scripts/lang/in_container.sh python -m vla_finetuning.items.collect_nominal --task $task --root $ROOT \
      --shard $name --target $target --seed-start $start --seed-step $step --seed-stop $stop \
      --max-runs ${MAX_RUNS:-10} < /dev/null >> $log 2>&1 &
    pid=$!; killed=0
    while kill -0 $pid 2> /dev/null; do
      sleep 30
      now=$(date +%s); last=$(stat -c %Y $log)
      for m in $(stat -c %Y ${ROOT}_logs/$task/$name/runs.jsonl 2> /dev/null); do [ $m -gt $last ] && last=$m; done
      if [ $(( now - last )) -gt ${HANG_S:-900} ]; then
        echo "[nominal-pool] $(date '+%T') $task $name: no output for $(( now - last ))s, killing $pid" >> $log
        kill -9 -- -$pid 2> /dev/null || kill -9 $pid 2> /dev/null; killed=1; break
      fi
    done
    wait $pid; rc=$?
    echo "[nominal-pool] $(date '+%F %T') $task $name rc=$rc" >> $log
    if [ $rc -eq 0 ]; then date '+%F %T' > ${ROOT}_logs/$task/$name/DONE; break; fi
    if [ $killed -eq 1 ]; then hung=$((hung + 1)); [ $hung -ge 3 ] && break; continue; fi
    if [ $rc -ge 128 ]; then crashes=$((crashes + 1)); [ $crashes -ge 3 ] && break; continue; fi
    [ $rc -ne 75 ] && { echo "[nominal-pool] $task $name rc=$rc: left for a look" >> $log; break; }
  done
}
for sh in $SHARDS; do
  IFS=: read -r t n _ <<< "$sh"
  [ -f ${ROOT}_logs/$t/$n/DONE ] && continue
  while [ $(jobs -rp | wc -l) -ge $SLOTS ]; do sleep 20; done
  slot "$sh" & sleep 30
done
wait
echo "[nominal] job done $(date '+%F %T')"
