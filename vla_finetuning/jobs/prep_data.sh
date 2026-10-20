#!/bin/bash
# Training data for the fine-tuning job (inside the container; PLAN.md "Fine-tuning"):
#   1. wait (at most WAIT_MIN minutes, default 90) for the safe collection stage of runs/lang_pipeline_vf
#      (vf_collect_safe) to be done; after the wait the demos collected so far are used;
#   2. pack every kept safe demo (data/demos_safe) with HoloBrain's packer into data/lmdb/safe, once
#      (data/PREP_DONE marks it; a rerun reuses it).
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
D=$VF/data
DONE_TXT=/work/nvme/bifq/project/LanguageConditionHJ/runs/lang_pipeline_vf/state/done.txt
cd /work/nvme/bifq/project/LanguageConditionHJ
if [ -f "$D/PREP_DONE" ]; then
  echo "[prep] data ready since $(cat "$D/PREP_DONE")"; exit 0
fi
end=$(( $(date +%s) + ${WAIT_MIN:-90} * 60 ))
until grep -qx vf_collect_safe "$DONE_TXT" 2>/dev/null || [ "$(date +%s)" -ge "$end" ]; do sleep 60; done
grep -qx vf_collect_safe "$DONE_TXT" && echo "[prep] safe collection done" \
  || echo "[prep] safe collection NOT done after the wait: using the demos collected so far"
mkdir -p "$D/lmdb"
rm -rf "$D/lmdb/safe"
python -m vla_finetuning.pack --root "$D/demos_safe" --out "$D/lmdb/safe" || exit 1
date '+%F %T' > "$D/PREP_DONE"
echo "[prep] packed: $(tr -d '\n ' < "$D/lmdb/safe/vf_pack.json" | cut -c1-500)"
