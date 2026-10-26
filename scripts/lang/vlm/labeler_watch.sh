#!/bin/bash
# Start and pause a VLM labeler chain stage on demand (VLM_FAILURE_PLAN; login node, a few file checks a minute).
# The labeler (vlm_label serve --idle-exit) stops itself when idle; this loop lifts the stage's hold file
# ($PIPE/state/hold.$STAGE, scripts/lang/chain6.sh) when there is work:
#   - an episode with complete agent-view frames and no label file of $VLM in one of $ROOTS;
#   - a fresh WANTED marker (a run collecting or waiting for labels, hsource.want_labels);
#   - $DONE_FILE still missing although $DONE_AFTER (the data's READY) exists;
# and sets it otherwise. It ends (stage held) once $UNTIL exists (the run's final checkpoint).
#   PIPE=<pipe> STAGE=<stage> VLM=<name> ROOTS="<ring> <warmup>/shard_g*" [DONE_FILE=.. DONE_AFTER=..] [UNTIL=..] \
#     setsid nohup scripts/lang/vlm/labeler_watch.sh >/dev/null 2>&1 &
set -u
: "${PIPE:?}" "${STAGE:?}" "${VLM:?}" "${ROOTS:?}"
hold=$PIPE/state/hold.$STAGE
log() { echo "$(date '+%F %T') $*" >> "$PIPE/watch_$STAGE.log"; }
mkdir -p "$PIPE/state"
echo $$ > "$PIPE/state/watch_$STAGE.pid"
log "START pid $$ roots $ROOTS"
while true; do
  if [ -n "${UNTIL:-}" ] && [ -e "$UNTIL" ]; then touch "$hold"; log "END: $UNTIL exists"; exit 0; fi
  need=""
  for r in $ROOTS; do   # globs expanded on every pass
    for d in "$r"/agent_view/*/done.json; do
      [ -e "$d" ] || continue
      e=$(basename "$(dirname "$d")")
      [ -e "$r/vlm_labels/$VLM/$e.json" ] || { need="pending $r episode $e"; break 2; }
    done
    w=$r/vlm_labels/$VLM/WANTED
    if [ -e "$w" ] && [ $(( $(date +%s) - $(stat -c %Y "$w") )) -lt 1800 ]; then need="wanted by $r"; break; fi
  done
  if [ -z "$need" ] && [ -n "${DONE_FILE:-}" ] && [ ! -e "$DONE_FILE" ] && [ -e "${DONE_AFTER:-/nonexistent}" ]; then
    need="$DONE_FILE to write"
  fi
  if [ -n "$need" ]; then
    [ -e "$hold" ] && { rm -f "$hold"; log "release: $need"; }
  else
    [ -e "$hold" ] || { touch "$hold"; log "hold: nothing to label"; }
  fi
  sleep 60
done
