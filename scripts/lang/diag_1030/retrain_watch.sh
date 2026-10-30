#!/bin/bash
# Check 4 (diag 2026-10-30): login-node watcher that keeps one critic-retrain pool job queued on each listed partition
# until every variant is finished. Light: one squeue per partition per minute, no python.
#   setsid nohup bash scripts/lang/diag_1030/retrain_watch.sh >> runs/lang/diag_1030/retrain/watch.log 2>&1 &
# Stops when: every variant reached its steps (checkpoint_final.pt), the file runs/lang/diag_1030/retrain/STOP exists,
# or two A100 resubmissions in a row found no new steps since the previous one (a crash loop).
set -u
REPO=/work/nvme/bifq/project/LanguageConditionHJ
cd $REPO
J=runs/lang/diag_1030/retrain/jobs
OUT=runs/lang/diag_1030/retrain
VARS="det stoch stoch_mlp:5000 zero data stoch_norm"
PARTS="gpuA100x4-interactive:pool_A100x4.sbatch gpuH200x8-interactive:pool_H200.sbatch"

step_of() { local f=$OUT/$1/diag.jsonl s; [ -f "$f" ] || { echo 0; return; }
  s=$(tail -1 "$f" | grep -o '"step": [0-9]*' | grep -o '[0-9]*$'); echo "${s:-0}"; }
cap_of() { case "$1" in *:*) echo "${1#*:}";; *) echo 20000;; esac; }
all_done() { local sp v; for sp in $VARS; do v=${sp%%:*}
  [ -f "$OUT/$v/checkpoint_final.pt" ] && [ "$(step_of $v)" -ge "$(cap_of $sp)" ] || return 1; done; }
total() { local sp t=0; for sp in $VARS; do t=$((t + $(step_of ${sp%%:*}))); done; echo $t; }

echo "[watch] $(date '+%F %T') start on $(hostname -s), pid $$; variants $VARS"
last_a100=-1; stale=0
while true; do
  [ -f $OUT/STOP ] && { echo "[watch] $(date '+%T') STOP file: exit"; exit 0; }
  all_done && { echo "[watch] $(date '+%T') every variant finished: exit"; exit 0; }
  for pp in $PARTS; do
    p=${pp%%:*}; f=${pp#*:}
    n=$(squeue -h -u "$USER" -p "$p" -o %j 2> /dev/null | grep -c '^diag1030-retrain')
    [ "$n" -gt 0 ] && continue
    if [ "$p" = "gpuA100x4-interactive" ]; then
      t=$(total)
      if [ "$last_a100" -ge 0 ] && [ "$t" -le "$last_a100" ]; then stale=$((stale + 1)); else stale=0; fi
      if [ $stale -ge 2 ]; then
        echo "[watch] $(date '+%T') no new steps over two A100 jobs (total $t): writing STOP"; date > $OUT/STOP; exit 1
      fi
      last_a100=$t
    fi
    echo "[watch] $(date '+%T') $p free: total steps $(total); $(sbatch $J/$f 2>&1)"
  done
  sleep 60
done
