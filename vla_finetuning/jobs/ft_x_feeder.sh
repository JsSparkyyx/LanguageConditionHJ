#!/bin/bash
# Login-node feeder for the item-scene fine-tune's extension runs (11-01): while results_items/decision.json says
# "EXTEND k" and ws/ft_items_safe_x<k>.DONE is missing, hold the chain's ft_items stage and keep one ft_x_train.sbatch
# job queued (1 GPU, the first free A100 / H200 / A40 interactive slot); once the run is DONE, release the chain, which
# exports, evaluates on the A40 and decides (ft_items_pipeline.sh). Stops on STOP, or after 3 training jobs in a row
# that end without a new checkpoint. Light: squeue every 30 s, no python.
#   setsid nohup bash vla_finetuning/jobs/ft_x_feeder.sh >> runs/lang_pipeline_ft/ftx_feeder.log 2>&1 < /dev/null &
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
cd $R
P=runs/lang_pipeline_ft; HOLD=$P/state/hold.ft_items; WS=vla_finetuning/ws
DEC=vla_finetuning/results_items/decision.json
SB=vla_finetuning/jobs/ft_x_train.sbatch
log() { echo "$(date '+%F %T') $*"; }
decision() { grep -o '"decision": "[A-Z0-9 ]*"' $DEC 2>/dev/null | sed 's/.*: "//; s/"//'; }
MARK=$P/state/.ftx_mark
# a checkpoint saved after the marker (each job writes a new ws/<run>_r<time>/checkpoints/checkpoint_<i>)
new_ckpt() { find $WS -maxdepth 5 -path "$WS/$1*/checkpoints/checkpoint_*/model/model.safetensors" -newer $MARK 2>/dev/null | grep -q .; }
log "feeder start pid $$ on $(hostname -s)"
bad=0
while true; do
  d=$(decision)
  case "$d" in
    STOP) rm -f $HOLD; log "decision STOP: chain released, feeder stops"; exit 0 ;;
    EXTEND*) ;;
    *) log "decision '$d' unknown: waiting"; sleep 300; continue ;;
  esac
  k=${d#EXTEND }; run=ft_items_safe_x$k
  if [ "$k" = 2 ]; then prev=ft_items_safe; else prev=ft_items_safe_x$((k - 1)); fi
  if [ -f $WS/$run.DONE ]; then
    [ -f $HOLD ] && { rm -f $HOLD; log "$run DONE ($(cat $WS/$run.DONE)): chain released for the export and the A40 test"; }
    sleep 120; continue
  fi
  [ -f $HOLD ] || { echo "ft_x_feeder $$ $(date '+%F %T'): $run trains on ft_x_train.sbatch" > $HOLD; log "hold.ft_items set"; }
  # the chain must not train the same run at the same time: cancel its pending job, wait out a running one
  for jid in $(squeue -u $USER -h -n lang-chain -t PENDING -o %i); do
    grep -q "JOB ft_items -> $jid" $P/chain.log && { scancel $jid; log "cancelled the chain's pending ft_items job $jid"; }
  done
  while squeue -u $USER -h -n lang-chain -t RUNNING -o %i | grep -q .; do sleep 30; done
  sleep 90   # the chain logs the cancelled job as a FAIL: take it off the try count
  sed -i "/^ft_items|$(grep -o 'JOB ft_items -> [0-9]*' $P/chain.log | tail -1 | grep -o '[0-9]*$')\$/d" $P/state/failed.txt
  touch $MARK
  jid=$(RUN=$run PREV=$prev sbatch --export=ALL $SB | awk '{print $4}')
  [ -n "$jid" ] || { log "submit failed; retry in 5 min"; sleep 300; continue; }
  log "submitted $jid ($run from $prev)"
  while squeue -j $jid -h 2>/dev/null | grep -q .; do sleep 30; done
  rc=$(grep -o '^\[ftx\] rc=[0-9]*' /work/nvme/bifq/project/tmp/logs/vf_ftx_train_$jid.out 2>/dev/null | tail -1)
  step=$(grep -o 'GlobalStep\[[0-9]*/' /work/nvme/bifq/project/tmp/logs/vf_ftx_train_$jid.out 2>/dev/null | tail -1)
  log "$jid ended: ${rc:-no rc} ${step:-no step}"
  if new_ckpt $run || [ -f $WS/$run.DONE ]; then bad=0; else bad=$((bad + 1)); fi
  [ $bad -ge 3 ] && { log "3 jobs in a row without a new checkpoint: feeder stops (hold kept)"; exit 1; }
done
