#!/bin/bash
# The action-sensitivity experiment's jobs, second round (ACTION_SENSITIVITY_PLAN section 7), from a login node.
# Every 5 min:
#   1. the branch collection of round ROUND (as-collect, A40 interactive) until <data root>/COLLECT_DONE;
#   2. per run of RUNS not DONE: one training job (as-tr-<run>) on an interactive partition where this user has no
#      job (interactive QOS: 1 job per user per partition; one partition per job, so the runs train side by side).
#      The runs of WAIT need the collection's data and wait for it. The A40 partition is used for training only
#      once the collection is done;
#   3. when every run is DONE: the scores of SCORE_RUNS (as-score), then stop.
# Every job asks for one GPU on one node. Stops submitting a kind after MAX_JOBS accepted submissions (a stuck
# stage then waits for a look).
# Start it over ssh, not from a Claude Code shell (it kills what its shell started):
#   ssh dt-login01 'cd /work/nvme/bifq/project/LanguageConditionHJ && setsid nohup bash scripts/lang/action_sens/feeder.sh > /dev/null 2>> runs/lang/action_sens/logs/feeder.stderr < /dev/null &'
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; O=$R/runs/lang/action_sens
ROUND=${ROUND:-2}; N_EPISODES=${N_EPISODES:-125}
D=$R/data/lang_asens; [ "$ROUND" = 2 ] && D=$R/data/lang_asens2
RUNS=${RUNS:-"F D2 G B B2 H D3"}; WAIT=${WAIT:-"B2 H D3"}; COLLECT=${COLLECT:-1}
SCORE_RUNS=${SCORE_RUNS:-"A B C D F D2 G B2 H D3"}
MAX_JOBS=${MAX_JOBS:-20}
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
log "feeder start (pid $$ on $(hostname -s)): round $ROUND, runs $RUNS (waiting for the collection: $WAIT), collect $COLLECT"
while true; do
  # 1. the collection
  if [ "$COLLECT" = 1 ] && [ ! -f $D/COLLECT_DONE ]; then
    q=$(sq -n as-collect) || q=ERR
    if [ -z "$q" ]; then
      p=$(free_part $A40)
      [ -n "$p" ] && [ "$p" != ERR ] && { log "collection: $(ls $D/episodes/*/result.json 2>/dev/null | wc -l) of $N_EPISODES episodes done"; submit collect $p --export=ALL,ROUND=$ROUND,N_EPISODES=$N_EPISODES scripts/lang/action_sens/collect.sbatch || true; }
    fi
  fi
  # 2. the trainings
  parts=$TRAINP; { [ -f $D/COLLECT_DONE ] || [ "$COLLECT" != 1 ]; } && parts="$TRAINP $A40"
  all=1
  for run in $RUNS; do
    [ -f $O/runs/$run.DONE ] && continue
    all=0
    case " $WAIT " in *" $run "*) [ -f $D/COLLECT_DONE ] || continue;; esac
    q=$(sq -n as-tr-$run) || q=ERR
    [ -n "$q" ] && continue
    p=$(free_part "$parts")
    { [ -z "$p" ] || [ "$p" = ERR ]; } && continue
    submit train_$run $p --export=ALL,NAME=$run -J as-tr-$run -o $O/logs/train_${run}_%j.out scripts/lang/action_sens/train.sbatch || true
  done
  # 3. the scores
  if [ $all = 1 ]; then
    if [ -f $O/SCORED$ROUND ]; then log "all done"; break; fi
    q=$(sq -n as-score) || q=ERR
    if [ -z "$q" ]; then
      p=$(free_part "$parts")
      [ -n "$p" ] && [ "$p" != ERR ] && submit score $p -J as-score -o $O/logs/score_%j.out \
        --export=ALL,STAGENAME=as-score,STAGECMD="python scripts/lang/action_sens/train_critic.py score --names $SCORE_RUNS && date > $O/SCORED$ROUND" scripts/lang/stage.sbatch || true
    fi
  fi
  sleep 300
done
