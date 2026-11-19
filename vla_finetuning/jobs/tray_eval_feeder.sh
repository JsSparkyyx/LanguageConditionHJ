#!/bin/bash
# The sliding-tray closed-loop test (jobs/tray_eval.sbatch -> items/evaluate_tray.py) as 1-hour pieces on
# gpuA40x4-interactive (the cameras need ray tracing). NAMES: the agents and models to test; a model joins once its
# training is DONE (ws/<name>.DONE; the piece exports it). A (name, part) is done when a piece printed its rc=0; each
# piece runs only the names with a part left and resumes where the last stopped. A refused sbatch is retried after
# 2 minutes. Stops when every name is done or after MAX_PIECES pieces.
#   ssh dt-login01 'cd /work/nvme/bifq/project/LanguageConditionHJ && setsid nohup bash vla_finetuning/jobs/tray_eval_feeder.sh \
#     > /dev/null 2>> /work/nvme/bifq/project/tmp/logs/tray_eval_feeder.err < /dev/null &'
R=/work/nvme/bifq/project/LanguageConditionHJ
RES=${RES:-$R/vla_finetuning/results_tray}; NAMES=${NAMES:-"scripted careless ft_tray"}
PARTS=${PARTS:-4}; SLOTS=${SLOTS:-4}; SCENES=${SCENES:-20}; MAX_PIECES=${MAX_PIECES:-20}
LOG=/work/nvme/bifq/project/tmp/logs/tray_eval_feeder.log; DONE=$RES/_feeder_done
cd $R; mkdir -p $DONE $RES/_logs
log() { echo "$(date '+%F %T') $*" >> $LOG; }
is_agent() { case $1 in scripted|careless|hold) return 0 ;; *) return 1 ;; esac; }
left() {   # names with a part left; $1 = 1: only those that can run now (a model needs ws/<name>.DONE)
  local n p out=""
  for n in $NAMES; do
    [ "$1" = 1 ] && ! is_agent $n && [ ! -f vla_finetuning/ws/$n.DONE ] && continue
    for p in $(seq 0 $((PARTS - 1))); do [ -f $DONE/$n.p$p ] || { out="$out $n"; break; }; done
  done
  echo $out
}
log "eval feeder start: $NAMES (pid $$ on $(hostname -s))"
piece=0
while [ $piece -lt $MAX_PIECES ]; do
  [ -z "$(left 0)" ] && { log "all done"; exit 0; }
  names=$(left 1)
  [ -z "$names" ] && { sleep 600; continue; }          # only models still training
  out=$(NAMES="$names" RES=$RES PARTS=$PARTS SLOTS=$SLOTS SCENES=$SCENES sbatch --parsable --export=ALL \
        --partition=gpuA40x4-interactive --time=01:00:00 --mem=96g --job-name=tray-eval-int \
        vla_finetuning/jobs/tray_eval.sbatch 2>&1)
  if ! [[ "$out" =~ ^[0-9]+$ ]]; then sleep 120; continue; fi
  piece=$((piece + 1)); log "piece $piece job $out: $names"
  while squeue -h -j $out 2>/dev/null | grep -q .; do sleep 60; done
  grep -oE "\[tray-eval\] [0-9:]+ [^ ]+ part [0-9]+ rc=0" /work/nvme/bifq/project/tmp/logs/tray_eval_$out.out \
    | awk '{print $3 ".p" $5}' | while read -r m; do date '+%F %T' > $DONE/$m; done
  log "piece $piece job $out ended: $(sacct -j $out -X -n -o State,Elapsed | tr -s ' '); done: $(ls $DONE | tr '\n' ' ')"
done
log "stopped after $MAX_PIECES pieces"
