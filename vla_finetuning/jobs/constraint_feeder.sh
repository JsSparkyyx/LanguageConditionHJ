#!/bin/bash
# The constraint fine-tune's pipeline on the login node (Yuxuan 11-03; jobs only, no heavy python here): every 5 min,
#   1. collection: a vf-ccollect job (A40 interactive) until every shard is DONE;
#   2. then training: a vf-ctrain job (A100 / H200 interactive, packs the data first) until ws/ft_constraint.DONE,
#      while the A40 slot runs the baselines' paired tests (vf-ceval NAMES="scripted ft_items_safe_x2");
#   3. then the new model's test (vf-ceval with ft_constraint added; its job exports the model first);
#   4. the summaries (numpy, here) and DONE.
# Stops at MAX_JOBS submissions per kind (a stuck stage then waits for a look). Log: results_constraint/_logs/feeder.log
#   setsid nohup bash vla_finetuning/jobs/constraint_feeder.sh > /dev/null 2>&1 < /dev/null &
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; VF=$R/vla_finetuning; RES=$VF/results_constraint
ROOT=$VF/data/demos_constraint; RUN=ft_constraint; MAX_JOBS=${MAX_JOBS:-60}
mkdir -p $RES/_logs; LOG=$RES/_logs/feeder.log
cd $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
queued() { squeue -h -u $USER -n $1 -o %i | head -1; }
declare -A n_sub
busy() {   # partition list -> 0 when this user already has a job in one of them (interactive QOS: 1 per user)
  local p; for p in ${1//,/ }; do [ -n "$(squeue -h -u $USER -p $p -o %i | head -1)" ] && return 0; done; return 1
}
submit() {   # kind partitions args... (only accepted submissions count towards MAX_JOBS)
  local k=$1 parts=$2; shift 2
  if [ ${n_sub[$k]:-0} -ge $MAX_JOBS ]; then log "$k: $MAX_JOBS jobs submitted, waiting for a look"; return 1; fi
  if busy $parts; then return 1; fi
  local id; id=$(sbatch --parsable "$@" 2>> $LOG) || { log "sbatch $k rejected"; return 1; }
  n_sub[$k]=$(( ${n_sub[$k]:-0} + 1 ))
  log "submitted $k $id (${n_sub[$k]})"
}
A40=gpuA40x4-interactive; TRAINP=gpuA100x4-interactive,gpuA100x8-interactive,gpuH200x8-interactive
shards_left() {   # -> the number of unfinished shards, or "ERR" when the shard table cannot be read
  local table n=0
  table=$(python3 -m vla_finetuning.items.constraint 2>> $LOG) && [ -n "$table" ] || { echo ERR; return; }
  while read -r task host shard rest; do [ -f ${ROOT}_logs/$task/$shard/DONE ] || n=$((n + 1)); done <<< "$table"
  echo $n
}
evals_left() {   # names -> number of (name, task, part) markers missing, or "ERR"
  local n=0 name t parts k tasks
  tasks=$(python3 -m vla_finetuning.items.constraint --active 2>> $LOG) && [ -n "$tasks" ] || { echo ERR; return; }
  for name in $1; do for t in $tasks; do
    case "$name:$t" in scripted:*) parts=1;; *:clean_pour) parts=4;; *:push_zone|*:selective_avoid) parts=2;; *) parts=1;; esac
    for k in $(seq 0 $((parts - 1))); do [ -f $RES/$name/DONE_$t.p$k ] || n=$((n + 1)); done
  done; done
  echo $n
}
log "feeder start (pid $$)"
while true; do
  left=$(shards_left)
  if [ "$left" = ERR ]; then log "shard lookup failed"; sleep 300; continue; fi
  if [ "$left" -gt 0 ]; then
    [ -z "$(queued vf-ccollect)" ] && { log "collection: $left shards left"; submit collect $A40 vla_finetuning/jobs/constraint_collect.sbatch || true; }
    sleep 300; continue
  fi
  base="scripted ft_items_safe_x2"
  if [ ! -f $VF/ws/$RUN.DONE ]; then
    [ -z "$(queued vf-ctrain)" ] && { log "training"; submit train $TRAINP vla_finetuning/jobs/ft_c_train.sbatch || true; }
    el=$(evals_left "$base")
    if [ "$el" != ERR ] && [ "$el" -gt 0 ] && [ -z "$(queued vf-ceval)" ]; then
      log "baseline tests: $el parts left"
      export NAMES="$base"; submit eval $A40 --export=ALL vla_finetuning/jobs/constraint_eval.sbatch || true
    fi
    sleep 300; continue
  fi
  all="$base $RUN"
  el=$(evals_left "$all")
  if [ "$el" = ERR ]; then log "task lookup failed"; sleep 300; continue; fi
  if [ "$el" -gt 0 ]; then
    [ -z "$(queued vf-ceval)" ] && { log "tests: $el parts left"; export NAMES="$all"; submit eval $A40 --export=ALL vla_finetuning/jobs/constraint_eval.sbatch || true; }
    sleep 300; continue
  fi
  for name in $all; do
    python3 -m vla_finetuning.items.evaluate_constraint --summary --out $RES/$name > /dev/null 2>> $LOG || log "summary $name failed"
  done
  date '+%F %T' > $RES/DONE
  log "all done"
  break
done
