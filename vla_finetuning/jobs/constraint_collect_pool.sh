#!/bin/bash
# One node's collectors of constraint_collect.sbatch (inside the container): SLOTS loops on this node's GPU, each
# claiming the next unfinished shard (items/constraint.py shards(); mkdir in this job's claim dir) and running
# collect_constraint on it, relaunched after each exit 75 (process recycling, a renderer error), until the shard is
# done (exit 0 -> <log>/DONE) or the stage deadline. A planner / code error (exit 3) leaves the shard for a look.
#   bash vla_finetuning/jobs/constraint_collect_pool.sh <claim dir> <slots> <root> [tasks...]
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
CL=$1; SLOTS=${2:-3}; ROOT=$3; shift 3
cd $R
ITEMS=$(python -m vla_finetuning.items.constraint ${CSET:+--set $CSET} "$@")   # CSET (env): t1x = the extra T1 set
# the shards with the most scenes left first (longest work first: on a 1-node job the slow shards would otherwise
# start last); the table order when the count fails
SORTED=$(ITEMS="$ITEMS" python - "$ROOT" <<'PYEOF'
import json, os, sys
root, rows = sys.argv[1], [l for l in os.environ["ITEMS"].splitlines() if l.strip()]
def left(line):
    task, host, shard, target = line.split()[:4]
    ka, kb, ks = set(), set(), set()
    try:
        for l in open(f"{root}_logs/{task}/{shard}/runs.jsonl"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("reason") == "kept":
                {"A": ka, "B": kb, "S": ks}.get(r.get("kind"), set()).add(r.get("seed"))
    except OSError:
        pass
    return int(target) - len(ks) - len(ka & kb)
print("\n".join(sorted(rows, key=lambda l: -left(l))))
PYEOF
) && [ -n "$SORTED" ] && ITEMS=$SORTED
mkdir -p ${ROOT}_logs/_pool
# MAX_RUNS (env, default 10; was collect_constraint's 30): Env builds per process before it exits 75 and is relaunched.
# HANG_S (env, default 900): a collector that writes neither its log nor its shard's runs.jsonl for this long is killed
# (its process group) and relaunched, at most 3 times per shard and job. Why (11-08): a collector holds ~14-15 GB of
# the A40's 46 GB from its first minute (ray tracing + cuRobo), so 3 per GPU leave ~1 GB; transient allocations
# (OIDN denoiser, item masks: "cannot create buffer") then fail, and once all six t1x collectors hung for 30 min.
slot() {
  local task host shard target start step stop log rc quick t0 crashes hung pid killed last m now
  while read -r task host shard target start step stop; do
    [ -z "$task" ] && continue
    [ -f ${ROOT}_logs/$task/$shard/DONE ] && continue
    mkdir "$CL/${task}_${shard}" 2> /dev/null || continue
    hostarg=""; [ "$host" != "-" ] && hostarg="--host $host"
    log=${ROOT}_logs/_pool/${task}_${shard}.log
    echo "[pool $(hostname -s) s$1] $(date '+%T') $task $shard"
    quick=0; crashes=0; hung=0
    while true; do
      t0=$(date +%s)
      # its own process group (setsid execs it: same pid), so a hang kill takes any children too
      setsid python -m vla_finetuning.items.collect_constraint --task $task $hostarg --root $ROOT --shard $shard \
        --target $target --seed-start $start --seed-step $step --seed-stop $stop --max-runs ${MAX_RUNS:-10} \
        < /dev/null >> $log 2>&1 &
      pid=$!; killed=0
      while kill -0 $pid 2> /dev/null; do
        sleep 30
        now=$(date +%s); last=$t0
        for m in $(stat -c %Y $log ${ROOT}_logs/$task/$shard/runs.jsonl 2> /dev/null); do [ $m -gt $last ] && last=$m; done
        if [ $(( now - last )) -gt ${HANG_S:-900} ]; then
          echo "[pool] $(date '+%T') $task $shard: no output for $(( now - last ))s, killing pid $pid (hang $((hung + 1)))" >> $log
          kill -9 -- -$pid 2> /dev/null || kill -9 $pid 2> /dev/null
          killed=1; break
        fi
      done
      wait $pid; rc=$?
      if [ $rc -eq 0 ]; then date '+%F %T' > ${ROOT}_logs/$task/$shard/DONE; break; fi
      if [ $killed -eq 1 ]; then hung=$((hung + 1)); [ $hung -ge 3 ] && break
      elif [ $rc -ge 128 ]; then crashes=$((crashes + 1)); [ $crashes -ge 3 ] && break
      elif [ $rc -ne 75 ]; then echo "[pool] $task $shard rc=$rc: left for a look" >> $log; break; fi
      if [ $(( $(date +%s) - t0 )) -lt 240 ]; then quick=$((quick + 1)); else quick=0; fi
      [ $quick -ge 5 ] && break
      [ $(( ${STAGE_DEADLINE:-0} - $(date +%s) )) -lt 420 ] && return 0
    done
    echo "[pool $(hostname -s) s$1] $(date '+%T') $task $shard rc=$rc"
  done <<< "$ITEMS"
}
for s in $(seq 1 $SLOTS); do slot $s & sleep 30; done
wait
