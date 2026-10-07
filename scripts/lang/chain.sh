#!/usr/bin/env bash
# Chained interactive GPU pipeline for main/lang (model: envs/scripts/item_envs/
# interactive_chain.sh). The interactive QOS allows ONE job per user (running or
# queued), up to 4 GPUs on a node for 1 hour; the batch partition can queue for
# days. Stages run one at a time from queue.txt, lines `name|gpus|cmd`
# (cmd runs in the container, cwd = repo root). Finished stages are skipped
# (state/done.txt); failed ones retry up to MAX_TRIES. Hot-reloads the queue.
#   setsid nohup scripts/lang/chain.sh >/dev/null 2>&1 &
set -u
ROOT=/work/nvme/bifq/project/LanguageConditionHJ
PIPE=$ROOT/runs/lang_pipeline
QUEUE=$PIPE/queue.txt
STATE=$PIPE/state
LOGS=/work/nvme/bifq/project/tmp/logs
MAX_TRIES=${MAX_TRIES:-3}
mkdir -p "$STATE" "$LOGS"
touch "$STATE/done.txt" "$STATE/failed.txt" "$QUEUE"
log() { echo "$(date '+%F %T') $*" >> "$PIPE/chain.log"; }
echo $$ > "$STATE/orchestrator.pid"
log "START pid $$ — queue: $(grep -cv '^#\|^$' "$QUEUE") stages"
while true; do
  QMTIME=$(stat -c %Y "$QUEUE")
  ran=0
  while IFS='|' read -r name gpus cmd; do
    case "$name" in ''|\#*) continue ;; esac
    grep -qx "$name" "$STATE/done.txt" && continue
    [ -f "$STATE/hold" ] && { log "HOLD file present — pausing"; break; }
    tries=$(grep -c "^$name|" "$STATE/failed.txt"); tries=${tries:-0}
    [ "$tries" -ge "$MAX_TRIES" ] && continue
    while squeue -u "$USER" -h -p gpuA40x4-interactive 2>/dev/null | grep -q .; do sleep 30; done
    log "SUBMIT $name gpus=$gpus (try $((tries + 1)))"
    out="$LOGS/lang_${name}_%A.out"
    jid=$(STAGENAME="$name" STAGECMD="$cmd" sbatch --export=ALL --gpus-per-node="$gpus" \
          --cpus-per-task=$((16 * gpus)) --mem=$((60 * gpus))g -o "$out" \
          "$ROOT/scripts/lang/gpu_stage.sbatch" | awk '{print $4}')
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
  [ $ran -eq 0 ] && sleep 60   # idle: wait for new stages
done
