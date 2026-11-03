#!/bin/bash
# One node's workers of constraint_eval.sbatch (inside the container): SLOTS loops claiming items (mkdir in this job's
# claim dir) until nothing is left or the stage deadline: first the scene lists (one per task, drawn once and shared
# by every model), then each model's paired test, per task and part (T6 / T3 / T1 split: long model episodes).
# An item is done when its marker exists: results_constraint/scenes/DONE_<task>, results_constraint/<name>/DONE_<task>.p<k>.
#   bash vla_finetuning/jobs/constraint_eval_pool.sh <claim dir> <slots> "<name> ..."
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; RES=$R/vla_finetuning/results_constraint
CL=$1; SLOTS=${2:-3}; NAMES=$3
cd $R
TASKS=$(python -m vla_finetuning.items.constraint --active)
[ -n "$TASKS" ] || { echo "[eval pool] task lookup failed"; exit 75; }
parts() { case "$2:$1" in scripted:*) echo 1;; *:clean_pour) echo 4;; *:push_zone|*:selective_avoid) echo 2;; *) echo 1;; esac; }
items() {
  local t n k name
  for t in $TASKS; do echo "scenes $t 0 1"; done
  for name in $NAMES; do for t in $TASKS; do
    n=$(parts $t $name); for k in $(seq 0 $((n - 1))); do echo "$name $t $k $n"; done
  done; done
}
mkdir -p $RES/_logs
run_item() {   # name task part parts -> rc
  local name=$1 t=$2 k=$3 n=$4
  if [ "$name" = scenes ]; then
    python -m vla_finetuning.items.evaluate_constraint --make-scenes --tasks $t --child --out $RES/scenes_only \
      < /dev/null >> $RES/_logs/scenes_$t.log 2>&1
    return $?
  fi
  local agent=model md=""
  if [ "$name" = scripted ]; then agent=scripted; else md="--model-dir $R/vla_finetuning/models/$name/model"; fi
  python -m vla_finetuning.items.evaluate_constraint --agent $agent $md --model-name $name --out $RES/$name \
    --tasks $t --child --part $k --parts $n < /dev/null >> $RES/_logs/${name}_${t}_p$k.log 2>&1
}
marker() { if [ "$1" = scenes ]; then echo $RES/scenes/DONE_$2; else echo $RES/$1/DONE_$2.p$3; fi; }
slot() {
  local work name t k n m rc
  while true; do
    work=0
    while read -r name t k n; do
      m=$(marker $name $t $k)
      [ -f $m ] && continue
      if [ "$name" != scenes ] && [ ! -f $(marker scenes $t 0) ]; then   # its scenes first
        [ -f $CL/scenes_$t.failed ] || work=1                         # (not if they failed in this job)
        continue
      fi
      mkdir "$CL/${name}_${t}_$k" 2> /dev/null || continue
      work=1
      echo "[eval $(hostname -s) s$1] $(date '+%T') $name $t part $k/$n"
      run_item $name $t $k $n; rc=$?
      if [ $rc -eq 0 ]; then mkdir -p $(dirname $m); date '+%F %T' > $m
      elif [ "$name" = scenes ]; then touch $CL/scenes_$t.failed; fi
      echo "[eval $(hostname -s) s$1] $(date '+%T') $name $t part $k/$n rc=$rc"
      [ $(( ${STAGE_DEADLINE:-0} - $(date +%s) )) -lt 420 ] && return 0
    done < <(items)
    [ $work -eq 0 ] && return 0
    sleep 60
    [ $(( ${STAGE_DEADLINE:-0} - $(date +%s) )) -lt 420 ] && return 0
  done
}
for s in $(seq 1 $SLOTS); do slot $s & sleep 20; done
wait
