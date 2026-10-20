#!/bin/bash
# Training data, prepared once for both fine-tuning jobs (inside the container; PLAN.md "Fine-tuning"):
#   1. wait (at most WAIT_MIN minutes, default 90) for the collection stages of runs/lang_pipeline_vf
#      (vf_collect_safe, vf_collect_nominal, vf_collect_nominal2) to be done;
#   2. the first job to get here builds the matched sets (make_matched: per task the layouts both demonstrators
#      kept) and packs each with HoloBrain's packer into data/lmdb/matched_{safe,nominal}; a job arriving while
#      that runs waits for data/PREP_DONE. Later jobs reuse the same LMDBs, so both arms train on one set.
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
D=$VF/data
DONE_TXT=/work/nvme/bifq/project/LanguageConditionHJ/runs/lang_pipeline_vf/state/done.txt
cd /work/nvme/bifq/project/LanguageConditionHJ
if [ ! -f "$D/PREP_DONE" ]; then
  end=$(( $(date +%s) + ${WAIT_MIN:-90} * 60 ))
  for s in vf_collect_safe vf_collect_nominal vf_collect_nominal2; do
    until grep -qx "$s" "$DONE_TXT" 2>/dev/null || [ "$(date +%s)" -ge "$end" ]; do sleep 60; done
    grep -qx "$s" "$DONE_TXT" && echo "[prep] $s done" || echo "[prep] $s NOT done after the wait: using the demos collected so far"
  done
fi
if mkdir "$D/.prep.lock" 2>/dev/null; then
  if [ ! -f "$D/PREP_DONE" ]; then
    rm -rf "$D/matched_safe" "$D/matched_nominal" "$D/lmdb/matched_safe" "$D/lmdb/matched_nominal"
    python -m vla_finetuning.make_matched --safe-root "$D/demos_safe" --nominal-root "$D/demos_nominal" \
      --out-safe "$D/matched_safe" --out-nominal "$D/matched_nominal" || { rmdir "$D/.prep.lock"; exit 1; }
    mkdir -p "$D/lmdb"
    for arm in safe nominal; do
      python -m vla_finetuning.pack --root "$D/matched_$arm" --out "$D/lmdb/matched_$arm" || { rmdir "$D/.prep.lock"; exit 1; }
    done
    date '+%F %T' > "$D/PREP_DONE"
  fi
  rmdir "$D/.prep.lock"
else
  echo "[prep] another job is preparing the data: waiting for $D/PREP_DONE"
  until [ -f "$D/PREP_DONE" ]; do
    [ -d "$D/.prep.lock" ] || { echo "[prep] the other job stopped without PREP_DONE"; exit 1; }
    sleep 60
  done
fi
echo "[prep] data ready ($(cat "$D/PREP_DONE")): $(cat "$D/matched_safe_summary.json" 2>/dev/null | tr -d '\n ' | cut -c1-400)"
