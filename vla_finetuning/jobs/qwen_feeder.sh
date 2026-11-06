#!/bin/bash
# The Qwen constraint fine-tunes on the login node (Yuxuan 11-06: "the qwen version, one for collision avoidance only,
# one for all the tasks ... use the existing data first"; jobs only, no heavy python here). Two runs of HoloBrain's
# Qwen version (configs/config_holobrain_qwen_common.py) on the balanced constraint demos already packed, the GD
# run's recipe (15k steps, batch 32 on one GPU):
#   qwen_constraint     the five tasks (as ft_constraint, GD)
#   qwen_constraint_t1  T1 selective_avoid only (the collision-avoidance task), tested on T1 only
# Every 5 min:
#   1. per run not DONE: a training job (ft_c_train.sbatch, job name vf-qt-<run>) on a training partition where this
#      user has no job (interactive QOS: 1 per user per partition), so the two chains run side by side;
#   2. per run DONE: its paired test (vf-ceval, A40 interactive; the job exports the model first), both names in one job;
#   3. the summaries (numpy, here) and results_constraint/DONE_qwen.
# Stops submitting a kind after MAX_JOBS (a stuck stage then waits for a look). Log: results_constraint/_logs/qwen_feeder.log
#   setsid nohup bash vla_finetuning/jobs/qwen_feeder.sh > /dev/null 2>&1 < /dev/null &
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; VF=$R/vla_finetuning; RES=$VF/results_constraint
MAX_JOBS=${MAX_JOBS:-60}
declare -A TASKS_OF=([qwen_constraint]="" [qwen_constraint_t1]="selective_avoid")   # "" = every active task
RUNS="qwen_constraint qwen_constraint_t1"
export VF_CONFIG=configs/config_holobrain_qwen_common.py
mkdir -p $RES/_logs; LOG=$RES/_logs/qwen_feeder.log
cd $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
for run in $RUNS; do   # the test's task list of a model trained on some tasks only (constraint_eval_pool.sh reads it)
  if [ -n "${TASKS_OF[$run]}" ]; then mkdir -p $RES/$run; echo "${TASKS_OF[$run]}" > $RES/$run/TASKS; fi
done
# A failed squeue (Slurm controller down, 11-05) must not read as "nothing queued": queued prints ERR, never empty.
queued() { local o; o=$(squeue -h -u $USER -n $1 -o %i 2>/dev/null) || { echo ERR; return; }; echo "$o" | head -1; }
declare -A n_sub
TRAINP="gpuA100x4-interactive gpuA100x8-interactive gpuH200x8-interactive"; A40=gpuA40x4-interactive
free_parts() {   # partitions (space list) -> those where this user has no job, comma-joined; ERR on a failed squeue
  local p o out=""
  for p in $1; do o=$(squeue -h -u $USER -p $p -o %i 2>/dev/null) || { echo ERR; return; }; [ -z "$o" ] && out="$out,$p"; done
  echo "${out#,}"
}
submit() {   # kind partitions sbatch-args... (only accepted submissions count towards MAX_JOBS)
  local k=$1 parts=$2; shift 2
  if [ ${n_sub[$k]:-0} -ge $MAX_JOBS ]; then log "$k: $MAX_JOBS jobs submitted, waiting for a look"; return 1; fi
  local id; id=$(sbatch --parsable --partition=$parts "$@" 2>> $LOG) || { log "sbatch $k rejected"; return 1; }
  n_sub[$k]=$(( ${n_sub[$k]:-0} + 1 ))
  log "submitted $k $id on $parts (${n_sub[$k]})"
}
evals_left() {   # names -> number of (name, task, part) markers missing, or "ERR"
  local n=0 name t parts k all tasks
  all=$(python3 -m vla_finetuning.items.constraint --active 2>> $LOG) && [ -n "$all" ] || { echo ERR; return; }
  for name in $1; do
    tasks=$all; [ -f $RES/$name/TASKS ] && tasks=$(cat $RES/$name/TASKS)
    for t in $tasks; do
      case "$t" in clean_pour) parts=4;; push_zone|selective_avoid) parts=2;; *) parts=1;; esac
      for k in $(seq 0 $((parts - 1))); do [ -f $RES/$name/DONE_$t.p$k ] || n=$((n + 1)); done
    done
  done
  echo $n
}
log "qwen feeder start (pid $$ on $(hostname -s)): $RUNS"
while true; do
  done_runs=""
  for run in $RUNS; do
    if [ -f $VF/ws/$run.DONE ]; then done_runs="$done_runs $run"; continue; fi
    q=$(queued vf-qt-$run)
    [ -n "$q" ] && continue                      # queued or running (or squeue failed: ERR)
    fp=$(free_parts "$TRAINP")
    [ "$fp" = ERR ] && { log "squeue failed"; continue; }
    [ -z "$fp" ] && continue                     # every training partition holds one of this user's jobs
    log "training $run"
    TASKS="${TASKS_OF[$run]}" RUN=$run submit train_$run $fp --export=ALL --job-name=vf-qt-$run \
      vla_finetuning/jobs/ft_c_train.sbatch || true
  done
  done_runs=${done_runs# }
  if [ -n "$done_runs" ]; then
    el=$(evals_left "$done_runs")
    if [ "$el" = ERR ]; then log "task lookup failed"
    elif [ "$el" -gt 0 ]; then
      q=$(queued vf-ceval)
      if [ -z "$q" ] && [ "$(free_parts $A40)" = "$A40" ]; then
        log "tests ($done_runs): $el parts left"
        NAMES="$done_runs" submit eval $A40 --export=ALL vla_finetuning/jobs/constraint_eval.sbatch || true
      fi
    elif [ "$done_runs" = "$RUNS" ]; then
      for name in $RUNS; do
        python3 -m vla_finetuning.items.evaluate_constraint --summary --out $RES/$name > /dev/null 2>> $LOG \
          || log "summary $name failed"
      done
      date '+%F %T' > $RES/DONE_qwen
      log "all done"
      break
    fi
  fi
  sleep 300
done
