#!/bin/bash
# The action-sensitivity experiment's jobs (ACTION_SENSITIVITY_PLAN steps 2 to 4), from a login node. Every 5 min:
#   1. the branch collection (as-collect, A40 interactive) until data/lang_asens/COLLECT_DONE;
#   2. per run of RUNS not DONE: one training job (as-tr-<run>) on an interactive partition where this user has no
#      job (interactive QOS: 1 job per user per partition; one partition per job, so the runs train side by side).
#      The runs with branch data (A B C D) wait for the collection. The A40 partition is used for training only
#      once the collection is done;
#   3. when every run is DONE: the scores (as-score), then stop.
# Stops submitting a kind after MAX_JOBS accepted submissions (a stuck stage then waits for a look).
# Start it over ssh, not from a Claude Code shell (it kills what its shell started):
#   ssh dt-login01 'cd /work/nvme/bifq/project/LanguageConditionHJ && setsid nohup bash scripts/lang/action_sens/feeder.sh > /dev/null 2>> runs/lang/action_sens/logs/feeder.stderr < /dev/null &'
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; O=$R/runs/lang/action_sens; D=$R/data/lang_asens
RUNS=${RUNS:-"E F A B C D"}; COLLECT=${COLLECT:-1}
MAX_JOBS=${MAX_JOBS:-12}
TRAINP=${TRAINP:-"gpuA100x8-interactive gpuA100x4-interactive gpuH200x8-interactive"}; A40=gpuA40x4-interactive   # H200 last: its queue held a job 13 h on 11-10
LOG=$O/logs/feeder.log
mkdir -p $O/logs $O/runs
cd $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
sq() { local o; o=$(squeue -h --me -o %i "$@" 2>&1) || return 1; echo "$o" | grep -qv '^[0-9]*$' && return 1; echo "$o"; }
declare -A n_sub
submit() {   # kind partition sbatch-args...
  local k=$1 p=$2; shift 2
  if [ ${n_sub[$k]:-0} -ge $MAX_JOBS ]; then log "$k: $MAX_JOBS jobs submitted, waiting for a look"; return 1; fi
  local id; id=$(sbatch --parsable --partition=$p "$@" 2>> $LOG) || { log "sbatch $k rejected"; return 1; }
  n_sub[$k]=$(( ${n_sub[$k]:-0} + 1 ))
  log "submitted $k $id on $p (${n_sub[$k]})"
}
free_part() {   # partitions -> the first where this user has no job ("" none, ERR a failed squeue)
  local p o
  for p in $1; do o=$(sq -p $p) || { echo ERR; return; }; [ -z "$o" ] && { echo $p; return; }; done
}
log "feeder start (pid $$ on $(hostname -s)): runs $RUNS, collect $COLLECT"
while true; do
  # 1. the collection
  if [ "$COLLECT" = 1 ] && [ ! -f $D/COLLECT_DONE ]; then
    q=$(sq -n as-collect) || q=ERR
    if [ -z "$q" ]; then
      p=$(free_part $A40)
      [ -n "$p" ] && [ "$p" != ERR ] && { log "collection: $(ls $D/episodes/*/result.json 2>/dev/null | wc -l) episodes done"; submit collect $p --export=ALL scripts/lang/action_sens/collect.sbatch || true; }
    fi
  fi
  # 2. the trainings
  parts=$TRAINP; [ -f $D/COLLECT_DONE ] || [ "$COLLECT" != 1 ] && parts="$TRAINP $A40"
  all=1
  for run in $RUNS; do
    [ -f $O/runs/$run.DONE ] && continue
    all=0
    case $run in A|B|C|D) [ -f $D/COLLECT_DONE ] || continue;; esac
    q=$(sq -n as-tr-$run) || q=ERR
    [ -n "$q" ] && continue
    p=$(free_part "$parts")
    { [ -z "$p" ] || [ "$p" = ERR ]; } && continue
    submit train_$run $p --export=ALL,NAME=$run -J as-tr-$run -o $O/logs/train_${run}_%j.out scripts/lang/action_sens/train.sbatch || true
  done
  # 3. the scores
  if [ $all = 1 ]; then
    if [ -f $O/SCORED ]; then log "all done"; break; fi
    q=$(sq -n as-score) || q=ERR
    if [ -z "$q" ]; then
      p=$(free_part "$parts")
      [ -n "$p" ] && [ "$p" != ERR ] && submit score $p -J as-score -o $O/logs/score_%j.out \
        --export=ALL,STAGENAME=as-score,STAGECMD="python scripts/lang/action_sens/train_critic.py score --names $RUNS && date > $O/SCORED" scripts/lang/stage.sbatch || true
    fi
  fi
  sleep 300
done
