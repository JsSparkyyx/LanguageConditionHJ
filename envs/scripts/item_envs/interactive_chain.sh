#!/usr/bin/env bash
# Chained interactive GPU pipeline: runs queue.txt stages one at a time through
# the 1-h interactive slot (QOS = 1 job/user). Idempotent — safe to relaunch;
# finished stages are skipped via state/done.txt, failed stages retry up to
# MAX_TRIES. Run detached:
#   setsid nohup envs/scripts/item_envs/interactive_chain.sh >/dev/null 2>&1 &
# Status: envs/runs/pipeline/state/{done,failed}.txt + chain.log
set -u
ROOT=/work/nvme/bifq/project/LanguageConditionHJ
PIPE=$ROOT/envs/runs/pipeline
QUEUE=$PIPE/queue.txt
STATE=$PIPE/state
LOGS=/work/nvme/bifq/project/tmp/logs
MAX_TRIES=${MAX_TRIES:-25}
mkdir -p "$STATE" "$LOGS"
touch "$STATE/done.txt" "$STATE/failed.txt"

log() { echo "$(date '+%F %T') $*" >> "$PIPE/chain.log"; }
log "START — queue: $(grep -cv '^#\|^$' "$QUEUE") stages"

while IFS='|' read -r name cmd; do
  case "$name" in ''|\#*) continue ;; esac
  grep -qx "$name" "$STATE/done.txt" 2>/dev/null && continue
  tries=$(grep -c "^$name|" "$STATE/failed.txt" 2>/dev/null)
  tries=${tries:-0}
  if [ "$tries" -ge "$MAX_TRIES" ]; then
    log "SKIP $name — $tries tries >= MAX_TRIES=$MAX_TRIES"
    continue
  fi
  # wait for a free interactive slot (a stray job of ours may be running)
  while squeue -u "$USER" -h -p gpuA40x4-interactive 2>/dev/null | grep -q .; do sleep 60; done
  log "SUBMIT $name (try $((tries + 1)))"
  jid=$(STAGENAME="$name" STAGECMD="$cmd" sbatch --export=ALL -o "$LOGS/chain_${name}_%A.out" \
        "$ROOT/envs/scripts/item_envs/stage_runner.sbatch" | awk '{print $4}')
  if [ -z "${jid:-}" ]; then log "SUBMIT-FAIL $name — sbatch error; sleeping 5 min"; sleep 300; continue; fi
  log "JOB $name -> $jid"
  while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 60; done
  if grep -q "STAGE_OK $name" "$LOGS/chain_${name}_${jid}.out" 2>/dev/null; then
    echo "$name" >> "$STATE/done.txt"
    log "DONE $name ($jid)"
  else
    echo "$name|$jid" >> "$STATE/failed.txt"
    log "FAIL $name ($jid) — will retry; see $LOGS/chain_${name}_${jid}.out"
  fi
done < "$QUEUE"
log "CHAIN COMPLETE"
