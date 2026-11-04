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
ITEMS=$(python -m vla_finetuning.items.constraint "$@")
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
slot() {
  local task host shard target start step stop log rc quick t0 crashes
  while read -r task host shard target start step stop; do
    [ -z "$task" ] && continue
    [ -f ${ROOT}_logs/$task/$shard/DONE ] && continue
    mkdir "$CL/${task}_${shard}" 2> /dev/null || continue
    hostarg=""; [ "$host" != "-" ] && hostarg="--host $host"
    log=${ROOT}_logs/_pool/${task}_${shard}.log
    echo "[pool $(hostname -s) s$1] $(date '+%T') $task $shard"
    quick=0; crashes=0
    while true; do
      t0=$(date +%s)
      python -m vla_finetuning.items.collect_constraint --task $task $hostarg --root $ROOT --shard $shard \
        --target $target --seed-start $start --seed-step $step --seed-stop $stop < /dev/null >> $log 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then date '+%F %T' > ${ROOT}_logs/$task/$shard/DONE; break; fi
      if [ $rc -ge 128 ]; then crashes=$((crashes + 1)); [ $crashes -ge 3 ] && break
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
