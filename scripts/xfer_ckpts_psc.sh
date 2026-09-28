#!/usr/bin/env bash
# Push last-3 checkpoints + config.json of every run to PSC Bridges-2.
# Usage:
#   bash scripts/xfer_ckpts_psc.sh                    # all runs
#   bash scripts/xfer_ckpts_psc.sh 'loo_feature|loo_geom|loo_xvla'   # matching runs only
#   bash scripts/xfer_ckpts_psc.sh holobrain          # HoloBrain encoder ckpt
set -euo pipefail

DEST_HOST="yyang69@bridges2.psc.edu"
DEST_BASE="/ocean/projects/cis260270p/yyang69"
TMP=/work/nvme/bifq/project/tmp

if [ "${1:-}" = "holobrain" ]; then
  echo "[xfer] HoloBrain ckpt -> $DEST_BASE/tmp/holobrain"
  tar czf - -C "$TMP" holobrain | ssh "$DEST_HOST" \
    "mkdir -p '$DEST_BASE/tmp' && tar xzf - -C '$DEST_BASE/tmp'"
  echo "[xfer] done"
  exit 0
fi

PATTERN="${1:-}"
cd "$TMP/cehj/runs"
LIST=$(mktemp /tmp/ckpt_xfer.XXXXXX)
for d in */; do
  d=${d%/}
  if [ -n "$PATTERN" ] && ! echo "$d" | grep -qE "$PATTERN"; then continue; fi
  [ -f "$d/config.json" ] && echo "$d/config.json" >> "$LIST"
  [ -f "$d/checkpoint.pt" ] && echo "$d/checkpoint.pt" >> "$LIST"
  ls -1v "$d"/checkpoint_epoch*.pt 2>/dev/null | tail -2 >> "$LIST" || true
done
if [ ! -s "$LIST" ]; then echo "[xfer] no files matched '$PATTERN'"; rm -f "$LIST"; exit 1; fi
echo "[xfer] $(wc -l < "$LIST") files:"
cat "$LIST"
tar czf - -T "$LIST" | ssh "$DEST_HOST" \
  "mkdir -p '$DEST_BASE/EVAL/runs' && tar xzf - -C '$DEST_BASE/EVAL/runs'"
rm -f "$LIST"
echo "[xfer] done -> $DEST_HOST:$DEST_BASE/EVAL/runs"
