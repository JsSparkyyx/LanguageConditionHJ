#!/bin/bash
# Training data of the item-scene fine-tune (inside the container; Yuxuan 2026-10-27/28: T1 / T3 / T5 / T6 / T7 safe
# demos, 100 kept each, T1 20 per host; ONE model for all five tasks):
#   exit 0 when every task has its kept demos and they are packed (data/lmdb/items_safe, once: data/PREP_ITEMS_DONE);
#   exit 75 when some task is still short (the caller waits or resubmits), nothing packed.
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
D=$VF/data; ROOT=$D/demos_items_safe; OUT=$D/lmdb/items_safe
cd /work/nvme/bifq/project/LanguageConditionHJ
if [ -f "$D/PREP_ITEMS_DONE" ]; then
  echo "[prep_items] data ready since $(cat "$D/PREP_ITEMS_DONE")"; exit 0
fi
python - "$ROOT" <<'EOF' || exit 75
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
need = {"push_zone": 100, "tray_transport": 100, "clean_pour": 100, "gap_place": 100}
hosts = ("place_container_plate", "place_burger_fries", "stack_blocks_two", "stack_bowls_two", "place_bread_basket")
def kept(d):
    p = d / "attempts.jsonl"
    return sum(1 for l in p.read_text().splitlines() if l.strip() and json.loads(l).get("kept")) if p.exists() else 0
short = []
for t, n in need.items():
    k = sum(kept(c) for c in (root / t).iterdir() if c.is_dir()) if (root / t).exists() else 0
    if k < n:
        short.append(f"{t} {k}/{n}")
for h in hosts:
    k = kept(root / "selective_avoid" / h)
    if k < 20:
        short.append(f"selective_avoid/{h} {k}/20")
print("[prep_items] short: " + ", ".join(short) if short else "[prep_items] every task has its demos")
sys.exit(1 if short else 0)
EOF
mkdir -p "$D/lmdb"
rm -rf "$OUT"
python -m vla_finetuning.pack --root "$ROOT" --out "$OUT" || exit 1
date '+%F %T' > "$D/PREP_ITEMS_DONE"
echo "[prep_items] packed: $(tr -d '\n ' < "$OUT/vf_pack.json" | cut -c1-500)"
