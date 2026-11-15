#!/bin/bash
# Training data of the sliding-tray fine-tune (inside the container; items/collect_tray.py, Yuxuan 2026-11-13): exit 0
# when every shard is DONE and the demos are packed (data/lmdb/tray_slide/PACK_DONE); 75 while shards are unfinished
# (the caller waits or resubmits), 1 when packing failed. The prep script of ft_c_train.sbatch (PREP=...).
#   bash vla_finetuning/jobs/prep_tray.sh
set -uo pipefail
R=/work/nvme/bifq/project/LanguageConditionHJ
VF=$R/vla_finetuning
ROOT=${TRAY_ROOT:-$VF/data/demos_tray_slide}; OUT=${TRAY_LMDB:-$VF/data/lmdb/tray_slide}
cd $R
[ -f "$OUT/PACK_DONE" ] && { echo "[prep_tray] packed: $OUT"; exit 0; }
shards=$(python -m vla_finetuning.items.collect_tray --shards) && [ -n "$shards" ] \
  || { echo "[prep_tray] shard lookup failed"; exit 75; }
short=0
for sh in $shards; do
  name=${sh%%:*}
  [ -f "${ROOT}_logs/tray_slide/$name/DONE" ] || { echo "[prep_tray] shard $name not done"; short=1; }
done
[ $short -eq 0 ] || exit 75
# demos per kind (a shard also exits 0 when its seeds ran out: a shortfall shows here)
for k in T TC C; do
  n=$(cat $ROOT/tray_slide/*_$k/attempts.jsonl 2>/dev/null | grep -c . || true)
  echo "[prep_tray] kind $k: $n demos"
done
rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"
python -m vla_finetuning.pack --root "$ROOT" --out "$OUT" --tasks tray_slide || exit 1
date '+%F %T' > "$OUT/PACK_DONE"
echo "[prep_tray] packed: $(tr -d '\n ' < "$OUT/vf_pack.json" | cut -c1-300)"
