#!/bin/bash
# Check 4 (diag 2026-10-30): a pool of critic-retrain workers on THIS node, run OUTSIDE the container (one srun task
# per node); each worker process runs inside it (scripts/lang/in_container.sh).
#
#   bash scripts/lang/diag_1030/retrain_pool.sh <variants a,b,..> <grad steps> <run root> <slots per GPU> <end epoch> \
#        [budget cap s] [extra train args]
#
# A variant spec "name:steps" caps that variant below <grad steps>. Ties in steps keep the order of <variants>.
# Every slot (GPU lane x slots per GPU) loops: claim the unfinished variant with the FEWEST steps done that no live job
# holds (mkdir <root>/<v>.claim, atomic; a claim of a job that is not RUNNING any more is stale), train it until
# <end epoch> - MARGIN (or the budget cap), release, repeat. A run dir with a state.pt is resumed. A variant is
# finished when <root>/<v>/checkpoint_final.pt exists and its diag.jsonl reached <grad steps>. One log per variant and job: <root>/<v>/job_<jobid>_<host>.log
set -u
VARS=${1:?variants}; STEPS=${2:?steps}; ROOT=${3:?root}; SLOTS=${4:?slots per gpu}; END=${5:?end epoch}
CAP=${6:-0}; shift 6 2>/dev/null || shift $#
EXTRA="$*"
REPO=/work/nvme/bifq/project/LanguageConditionHJ
S=$REPO/scripts/lang/in_container.sh
MARGIN=${MARGIN:-240}       # seconds kept free before the job end (final save, container exit)
MIN_LEFT=${MIN_LEFT:-480}   # do not start a worker with less time than this (setup ~2-3 min)
cd $REPO
mkdir -p "$ROOT"
IFS=, read -ra V <<< "$VARS"
JOB=${SLURM_JOB_ID:-local}
HOST=$(hostname -s)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"; else
  G=(); for k in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do G+=("$k"); done; fi
echo "[pool $HOST] job $JOB: ${#G[@]} GPU(s) ${G[*]} x $SLOTS slot(s), variants ${V[*]} -> $STEPS steps in $ROOT," \
     "ends $(date -d @$END '+%T') ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1))"

cap_of() {   # a variant spec "name:steps" caps that variant's steps below <grad steps>
  case "$1" in *:*) echo "${1#*:}";; *) echo "$STEPS";; esac
}

steps_of() {   # last probed step of a variant (0 if none)
  local f=$ROOT/$1/diag.jsonl
  [ -f "$f" ] || { echo 0; return; }
  tail -1 "$f" | grep -o '"step": [0-9]*' | grep -o '[0-9]*$' || echo 0
}

finished() {   # $1 = spec; checkpoint_final.pt written AND probed at >= its steps (a smoke pass to 100 then 150 resumes)
  local v=${1%%:*}
  [ -f "$ROOT/$v/checkpoint_final.pt" ] && [ "$(steps_of $v)" -ge "$(cap_of $1)" ]
}

claim_live() {   # is the claim dir $1 held by a RUNNING job?
  local own jid
  own=$(cat "$1/owner" 2>/dev/null) || return 1
  jid=${own%% *}
  [ "$jid" = "$JOB" ] && return 0
  if command -v squeue > /dev/null; then
    [ "$(squeue -h -j "$jid" -o %T 2>/dev/null)" = "RUNNING" ]
  else
    [ $(( $(date +%s) - $(stat -c %Y "$1/owner") )) -lt 4200 ]
  fi
}

claim_next() {   # prints the claimed variant (empty: nothing left)
  local order v c sp
  order=$(for sp in "${V[@]}"; do finished $sp && continue; echo "$(steps_of ${sp%%:*}) $sp"; done \
          | sort -s -n -k1,1 | awk '{print $2}')
  for sp in $order; do
    v=${sp%%:*}
    c=$ROOT/$v.claim
    if [ -d "$c" ] && ! claim_live "$c"; then
      echo "[pool $HOST] stale claim on $v ($(cat $c/owner 2>/dev/null)) removed" >&2
      rm -rf "$c"
    fi
    if mkdir "$c" 2> /dev/null; then
      echo "$JOB $HOST $1 $$ $(date +%s)" > "$c/owner"
      echo "$sp"
      return
    fi
  done
}

worker() {   # $1 = GPU id, $2 = slot tag
  local gpu=$1 tag=$2 v sp st left b res rc R log
  while true; do
    left=$(( END - $(date +%s) - MARGIN ))
    [ $left -lt $MIN_LEFT ] && { echo "[pool $HOST $tag] ${left}s left: done"; return; }
    sp=$(claim_next "gpu$gpu")
    [ -z "$sp" ] && { echo "[pool $HOST $tag] nothing left to claim"; return; }
    v=${sp%%:*}; st=$(cap_of "$sp")
    b=$left; [ "$CAP" -gt 0 ] && [ "$CAP" -lt "$b" ] && b=$CAP
    R=$ROOT/$v; mkdir -p "$R"
    res=""; [ -f "$R/state.pt" ] && res="--resume"
    log=$R/job_${JOB}_${HOST}.log
    echo "[pool $HOST $tag] $(date '+%T') $v from step $(steps_of $v) to $st ($res), budget ${b}s, GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $S python scripts/lang/diag_1030/critic_retrain.py train --target "$v" \
      --run "$R" --grad-steps "$st" --time-budget "$b" $res $EXTRA >> "$log" 2>&1
    rc=$?
    rm -rf "$ROOT/$v.claim"
    echo "[pool $HOST $tag] $(date '+%T') $v rc=$rc at step $(steps_of $v): $(grep -E '^\[(retrain|lang-train)\]' "$log" | tail -1)"
    # a crash (not 0 / 75) is not retried in this job: it would only crash again
    [ $rc -ne 0 ] && [ $rc -ne 75 ] && { echo "[pool $HOST $tag] $v FAILED (rc $rc), see $log"; tail -5 "$log"; return; }
    [ $rc -eq 75 ] && return
  done
}

k=0
for g in "${G[@]}"; do
  for s in $(seq 1 "$SLOTS"); do
    worker "$g" "g${g}s${s}" &
    k=$((k + 1))
    sleep 20   # stagger the setups (sampler indexing reads every shard)
  done
done
wait
echo "[pool $HOST] all slots done $(date '+%T')"
