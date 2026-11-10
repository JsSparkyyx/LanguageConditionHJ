#!/bin/bash
# The T1 data test on the login node (Yuxuan 11-08: "run t1 only qwen and gd both for more data to confirm if it's the
# data issue"; jobs only, no heavy python here). T1-only constraint fine-tunes, the same recipe (15k steps, batch 32 on
# one GPU), 100 vs 300 training scenes, GD vs Qwen:
#   qwen_constraint_t1      Qwen, the 100 T1 scenes of data/demos_constraint (done 11-07; tested again on the new scenes)
#   ft_constraint_t1        GD,   the same 100 scenes (starts at once)
#   ft_constraint_t1_300    GD,   300 scenes: the 100 + the 200 of the t1x set (constraint.py; its first job packs them)
#   qwen_constraint_t1_300  Qwen, 300 scenes (waits for the t1x pack: two packers would race on one lmdb)
# (500 scenes until 11-09, Yuxuan: "i think 200 new is enough"; the cluster gave 1-hour jobs every 4-8 h)
# The new runs prune their superseded checkpoints (PRUNE_OLD_CKPT, ft_chain_step.sh: ws/ is on the HDD pool, 740 of 1,000 GiB used on 11-08).
# Test: the paired A / B / C / B2 / C2 test on 100 T1 scenes (results_constraint_t1: the 20 of results_constraint, same
# ids, + 80 more), with scripted; 5 parts per model. Every 5 min:
#   1. the t1x collection (vf-ccollect, A40 interactive) until every shard is DONE;
#   2. per run not DONE: a training job (vf-tx-<run>) on a training partition where this user has no job
#      (interactive QOS: 1 per user per partition); the 300-scene runs only after the collection. While more than one
#      run can train, each job asks for ONE partition (11-10: a waiting job that lists several partitions holds this
#      user's slot in each, so the runs trained one at a time); a run training alone lists every free one.
#      From 11-10 14:40 the GENERAL partitions instead (see TRAINP below): no such limit, 24-hour training jobs;
#   3. after the collection: the tests of scripted, qwen_constraint_t1 and every DONE run (vf-ceval, A40 interactive;
#      the job exports a DONE model first);
#   4. the summaries (numpy, here) and results_constraint_t1/DONE.
# Stops submitting a kind after MAX_JOBS (a stuck stage then waits for a look). Log: results_constraint_t1/_logs/feeder.log
#   setsid nohup bash vla_finetuning/jobs/t1x_feeder.sh > /dev/null 2>> vla_finetuning/results_constraint_t1/_logs/feeder.stderr < /dev/null &
# Start it over ssh (ssh <login node> 'cd <repo> && setsid nohup ... &'), not from a Claude Code shell: on 11-10 two
# feeders started from Claude Code's shell (setsid, nohup) were killed silently within a turn or two; the ssh-started ones
# run for days.
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; VF=$R/vla_finetuning; RES=$VF/results_constraint_t1; LM=$VF/data/lmdb
MAX_JOBS=${MAX_JOBS:-80}
unset VF_CONFIG TASKS CSET LMDBS NAMES EVAL_TASKS SCENES PARTS RUN
RUNS="ft_constraint_t1 ft_constraint_t1_300 qwen_constraint_t1_300"
QWEN=configs/config_holobrain_qwen_common.py
declare -A CFG=([ft_constraint_t1]="" [ft_constraint_t1_300]="" [qwen_constraint_t1_300]=$QWEN)
declare -A BIG=([ft_constraint_t1]=0 [ft_constraint_t1_300]=1 [qwen_constraint_t1_300]=1)
PACKER=ft_constraint_t1_300
PACKED=$LM/constraint_t1x_selective_avoid/PACK_DONE
LBIG="$LM/constraint_selective_avoid,$LM/constraint_t1x_selective_avoid"
TESTED="scripted qwen_constraint_t1 $RUNS"; T=selective_avoid; NPARTS=5; NSCENES=100
CROOT=$VF/data/demos_constraint_t1x
mkdir -p $RES/_logs; LOG=$RES/_logs/feeder.log
cd $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
# A failed squeue (Slurm controller down, 11-05) must not read as "nothing queued": queued prints ERR, never empty.
# --me, not -u $USER: on 11-09 the cluster's user lookup failed for hours ("Invalid user: yyang69") and squeue -u then
# printed that error with exit 0 and no jobs, so the feeder took every queue for empty. Any output that is not job ids
# (an error message) counts as a failure too.
sq() { local o; o=$(squeue -h --me -o %i "$@" 2>&1) || return 1; echo "$o" | grep -qv '^[0-9]*$' && return 1; echo "$o"; }
queued() { local o; o=$(sq -n $1) || { echo ERR; return; }; echo "$o" | head -1; }
declare -A n_sub
# 11-10 (Yuxuan: "move it to non-interactive queues and let them wait and run on general queues, the chance for
# interactive should be left for this experiment", the action-sensitivity test): the general partitions, with longer
# jobs (the wait is long, so a job should do a whole stage once it starts). No 1-job-per-partition limit there: one
# job per run / stage at a time, each listing every partition of its kind.
TRAINP="gpuA100x4 gpuA100x8 gpuH200x8"; A40=gpuA40x4
TRAIN_TIME=${TRAIN_TIME:-24:00:00}; A40_TIME=${A40_TIME:-06:00:00}
# One GPU per job (Yuxuan 11-10: "for vla finetuning don't request 4 gpu"): the collection and test jobs ask for
# --nodes=1 (their scripts allow 1-4 nodes with one GPU each).
free_parts() { echo "$1" | tr ' ' ','; }   # general queues: every partition of the list is open to a new job
submit() {   # kind partitions sbatch-args... (only accepted submissions count towards MAX_JOBS)
  local k=$1 parts=$2; shift 2
  if [ ${n_sub[$k]:-0} -ge $MAX_JOBS ]; then log "$k: $MAX_JOBS jobs submitted, waiting for a look"; return 1; fi
  local id; id=$(sbatch --parsable --partition=$parts "$@" 2>> $LOG) || { log "sbatch $k rejected"; return 1; }
  n_sub[$k]=$(( ${n_sub[$k]:-0} + 1 ))
  log "submitted $k $id on $parts (${n_sub[$k]})"
}
shards_left() {   # -> unfinished t1x shards, or "ERR"
  local table n=0
  table=$(python3 -m vla_finetuning.items.constraint --set t1x 2>> $LOG) && [ -n "$table" ] || { echo ERR; return; }
  while read -r task host shard rest; do [ -f ${CROOT}_logs/$task/$shard/DONE ] || n=$((n + 1)); done <<< "$table"
  echo $n
}
evals_left() {   # names -> missing (name, part) markers (+1 while the scene list is short)
  local n=0 name k
  [ -f $RES/scenes/DONE_$T ] || n=1
  for name in $1; do for k in $(seq 0 $((NPARTS - 1))); do [ -f $RES/$name/DONE_$T.p$k ] || n=$((n + 1)); done; done
  echo $n
}
log "t1x feeder start (pid $$ on $(hostname -s)): $RUNS"
while true; do
  cl=$(shards_left)
  # 1. the collection
  if [ "$cl" = ERR ]; then log "shard lookup failed"
  elif [ "$cl" -gt 0 ]; then
    q=$(queued vf-ccollect)
    if [ -z "$q" ] && [ "$(free_parts $A40)" = "$A40" ]; then
      log "collection: $cl shards left"
      CSET=t1x submit collect $A40 --time=$A40_TIME --nodes=1 --export=ALL vla_finetuning/jobs/constraint_collect.sbatch || true
    fi
  fi
  # 2. the trainings
  n_train=0   # runs that can train now
  for run in $RUNS; do
    [ -f $VF/ws/$run.DONE ] && continue
    if [ ${BIG[$run]} = 1 ]; then [ "$cl" = 0 ] || continue; [ $run = $PACKER ] || [ -f $PACKED ] || continue; fi
    n_train=$((n_train + 1))
  done
  done_runs=""
  for run in $RUNS; do
    if [ -f $VF/ws/$run.DONE ]; then done_runs="$done_runs $run"; continue; fi
    if [ ${BIG[$run]} = 1 ]; then
      [ "$cl" = 0 ] || continue
      [ $run = $PACKER ] || [ -f $PACKED ] || continue
    fi
    q=$(queued vf-tx-$run)
    [ -n "$q" ] && continue                      # queued or running (or squeue failed: ERR)
    fp=$(free_parts "$TRAINP")
    [ "$fp" = ERR ] && { log "squeue failed"; continue; }
    [ -z "$fp" ] && continue                     # every training partition holds one of this user's jobs
    log "training $run"
    if [ ${BIG[$run]} = 1 ]; then cs=t1x; lm=$LBIG; else cs=""; lm=""; fi
    RUN=$run TASKS=$T PRUNE_OLD_CKPT=1 VF_CONFIG="${CFG[$run]}" CSET="$cs" LMDBS="$lm" \
      submit train_$run $fp --time=$TRAIN_TIME --export=ALL --job-name=vf-tx-$run vla_finetuning/jobs/ft_c_train.sbatch || true
  done
  # 3. the tests (the A40 slot is the collection's until it is done)
  if [ "$cl" = 0 ]; then
    names="scripted qwen_constraint_t1$done_runs"
    el=$(evals_left "$names")
    if [ "$el" -gt 0 ]; then
      q=$(queued vf-ceval)
      if [ -z "$q" ] && [ "$(free_parts $A40)" = "$A40" ]; then
        log "tests ($names): $el parts left"
        NAMES="$names" RES=$RES EVAL_TASKS=$T SCENES=$NSCENES PARTS=$NPARTS \
          submit eval $A40 --time=$A40_TIME --nodes=1 --export=ALL vla_finetuning/jobs/constraint_eval.sbatch || true
      fi
    elif [ "$names" = "$TESTED" ]; then
      # 4. the summaries
      for name in $TESTED; do
        python3 -m vla_finetuning.items.evaluate_constraint --summary --out $RES/$name --tasks $T > /dev/null 2>> $LOG \
          || log "summary $name failed"
      done
      date '+%F %T' > $RES/DONE
      log "all done"
      break
    fi
  fi
  sleep 300
done
