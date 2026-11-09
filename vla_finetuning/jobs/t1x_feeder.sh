#!/bin/bash
# The T1 data test on the login node (Yuxuan 11-08: "run t1 only qwen and gd both for more data to confirm if it's the
# data issue"; jobs only, no heavy python here). T1-only constraint fine-tunes, the same recipe (15k steps, batch 32 on
# one GPU), 100 vs 500 training scenes, GD vs Qwen:
#   qwen_constraint_t1      Qwen, the 100 T1 scenes of data/demos_constraint (done 11-07; tested again on the new scenes)
#   ft_constraint_t1        GD,   the same 100 scenes (starts at once)
#   ft_constraint_t1_500    GD,   500 scenes: the 100 + the 400 of the t1x set (constraint.py; its first job packs them)
#   qwen_constraint_t1_500  Qwen, 500 scenes (waits for the t1x pack: two packers would race on one lmdb)
# The new runs prune their superseded checkpoints (PRUNE_OLD_CKPT, ft_chain_step.sh: the SSD pool had 0.5 TB left).
# Test: the paired A / B / C / B2 / C2 test on 100 T1 scenes (results_constraint_t1: the 20 of results_constraint, same
# ids, + 80 more), with scripted; 5 parts per model. Every 5 min:
#   1. the t1x collection (vf-ccollect, A40 interactive) until every shard is DONE;
#   2. per run not DONE: a training job (vf-tx-<run>) on a training partition where this user has no job
#      (interactive QOS: 1 per user per partition); the 500-scene runs only after the collection;
#   3. after the collection: the tests of scripted, qwen_constraint_t1 and every DONE run (vf-ceval, A40 interactive;
#      the job exports a DONE model first);
#   4. the summaries (numpy, here) and results_constraint_t1/DONE.
# Stops submitting a kind after MAX_JOBS (a stuck stage then waits for a look). Log: results_constraint_t1/_logs/feeder.log
#   setsid nohup bash vla_finetuning/jobs/t1x_feeder.sh > /dev/null 2>&1 < /dev/null &
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; VF=$R/vla_finetuning; RES=$VF/results_constraint_t1; LM=$VF/data/lmdb
MAX_JOBS=${MAX_JOBS:-80}
unset VF_CONFIG TASKS CSET LMDBS NAMES EVAL_TASKS SCENES PARTS RUN
RUNS="ft_constraint_t1 ft_constraint_t1_500 qwen_constraint_t1_500"
QWEN=configs/config_holobrain_qwen_common.py
declare -A CFG=([ft_constraint_t1]="" [ft_constraint_t1_500]="" [qwen_constraint_t1_500]=$QWEN)
declare -A BIG=([ft_constraint_t1]=0 [ft_constraint_t1_500]=1 [qwen_constraint_t1_500]=1)
PACKER=ft_constraint_t1_500
PACKED=$LM/constraint_t1x_selective_avoid/PACK_DONE
L500="$LM/constraint_selective_avoid,$LM/constraint_t1x_selective_avoid"
TESTED="scripted qwen_constraint_t1 $RUNS"; T=selective_avoid; NPARTS=5; NSCENES=100
CROOT=$VF/data/demos_constraint_t1x
mkdir -p $RES/_logs; LOG=$RES/_logs/feeder.log
cd $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
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
      CSET=t1x submit collect $A40 --export=ALL vla_finetuning/jobs/constraint_collect.sbatch || true
    fi
  fi
  # 2. the trainings
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
    if [ ${BIG[$run]} = 1 ]; then cs=t1x; lm=$L500; else cs=""; lm=""; fi
    RUN=$run TASKS=$T PRUNE_OLD_CKPT=1 VF_CONFIG="${CFG[$run]}" CSET="$cs" LMDBS="$lm" \
      submit train_$run $fp --export=ALL --job-name=vf-tx-$run vla_finetuning/jobs/ft_c_train.sbatch || true
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
          submit eval $A40 --export=ALL vla_finetuning/jobs/constraint_eval.sbatch || true
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
