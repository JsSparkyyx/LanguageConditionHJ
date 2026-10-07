#!/usr/bin/env bash
# Chained interactive GPU pipeline (generalized chain.sh): PARTITION and PIPE
# select the interactive partition (each has its own 1-job-per-user QOS) and
# the queue directory. Lines of $PIPE/queue.txt: `name|gpus|cmd` (cmd runs in
# the container, cwd = repo root). STAGE_OK -> done; STAGE_CONTINUE (exit 75 /
# stage timeout) -> resubmitted without counting a failure; else retried up
# to MAX_TRIES. Hot-reloads the queue. Simulation stages belong on A40
# (ray-traced rendering); torch-only stages can use gpuA100x4-interactive.
#   PARTITION=gpuA100x4-interactive PIPE=runs/lang_pipeline_a100 setsid nohup scripts/lang/chain2.sh >/dev/null 2>&1 &
set -u
ROOT=/work/nvme/bifq/project/LanguageConditionHJ
PARTITION=${PARTITION:-gpuA40x4-interactive}
PIPE=$(readlink -f "${PIPE:-$ROOT/runs/lang_pipeline}")
QUEUE=$PIPE/queue.txt
STATE=$PIPE/state
LOGS=/work/nvme/bifq/project/tmp/logs
MAX_TRIES=${MAX_TRIES:-3}
mkdir -p "$STATE" "$LOGS"
touch "$STATE/done.txt" "$STATE/failed.txt" "$QUEUE"
log() { echo "$(date '+%F %T') $*" >> "$PIPE/chain.log"; }
echo $$ > "$STATE/orchestrator.pid"
log "START pid $$ partition $PARTITION — queue: $(grep -cv '^#\|^$' "$QUEUE") stages"
while true; do
  QMTIME=$(stat -c %Y "$QUEUE")
  ran=0
  while IFS='|' read -r name gpus cmd; do
    case "$name" in ''|\#*) continue ;; esac
    grep -qx "$name" "$STATE/done.txt" && continue
    [ -f "$STATE/hold" ] && { log "HOLD file present — pausing"; break; }
    tries=$(grep -c "^$name|" "$STATE/failed.txt"); tries=${tries:-0}
    [ "$tries" -ge "$MAX_TRIES" ] && continue
    while squeue -u "$USER" -h -p "$PARTITION" 2>/dev/null | grep -q .; do sleep 30; done
    log "SUBMIT $name gpus=$gpus (try $((tries + 1)))"
    out="$LOGS/lang_${name}_%A.out"
    if [[ "$gpus" == *x* ]]; then   # NODESxGPUS, e.g. 1-4x1: one worker per node (gpu_stage_mn.sbatch)
      nodes=${gpus%x*}; gpn=${gpus#*x}
      jid=$(STAGENAME="$name" STAGECMD="$cmd" sbatch --export=ALL --partition="$PARTITION" --nodes="$nodes" \
            --gpus-per-node="$gpn" --cpus-per-task=$((${CPUS_PER_GPU:-16} * gpn)) --mem=$((${MEM_PER_GPU:-60} * gpn))g \
            -o "$out" "$ROOT/scripts/lang/gpu_stage_mn.sbatch" | awk '{print $4}')
    else
      jid=$(STAGENAME="$name" STAGECMD="$cmd" sbatch --export=ALL --partition="$PARTITION" --gpus-per-node="$gpus" \
            --cpus-per-task=$((${CPUS_PER_GPU:-16} * gpus)) --mem=$((${MEM_PER_GPU:-60} * gpus))g -o "$out" \
            "$ROOT/scripts/lang/gpu_stage.sbatch" | awk '{print $4}')
    fi
    if [ -z "${jid:-}" ]; then log "SUBMIT-FAIL $name; sleeping 5 min"; sleep 300; ran=1; break; fi
    log "JOB $name -> $jid"
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 30; done
    if grep -q "STAGE_OK $name" "$LOGS/lang_${name}_${jid}.out" 2>/dev/null; then
      echo "$name" >> "$STATE/done.txt"; log "DONE $name ($jid)"
    elif grep -q "STAGE_CONTINUE $name" "$LOGS/lang_${name}_${jid}.out" 2>/dev/null; then
      log "CONTINUE $name ($jid): resumable stage paused, resubmitting"
    else
      echo "$name|$jid" >> "$STATE/failed.txt"; log "FAIL $name ($jid): $LOGS/lang_${name}_${jid}.out"
    fi
    ran=1
    [ "$(stat -c %Y "$QUEUE")" != "$QMTIME" ] && { log "QUEUE CHANGED — new pass"; break; }
  done < "$QUEUE"
  [ $ran -eq 0 ] && sleep 60
done
