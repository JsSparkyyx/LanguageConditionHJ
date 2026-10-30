#!/bin/bash
# Training data of the item-scene fine-tune (inside the container; Yuxuan 2026-10-27/28: T1 / T3 / T5 / T6 / T7 safe
# demos, 100 kept each, T1 20 per host; ONE model for all five tasks; 10-29: the GPU pilot's and the T1 debug run's
# kept demos, same collector and settings, are reused as extra config directories, counted with their task / host):
#   exit 0 when every task has its kept demos and they are packed (data/lmdb/items_safe_<task>, once: data/PREP_ITEMS_DONE);
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
def host_of(c):   # a T1 config's host (its settings.json): the host's own shard and reused pilot / debug configs
    try:
        return json.loads((c / "settings.json").read_text()).get("host")
    except (OSError, ValueError):
        return None
t1 = [c for c in (root / "selective_avoid").iterdir() if c.is_dir()] if (root / "selective_avoid").exists() else []
for h in hosts:
    k = sum(kept(c) for c in t1 if host_of(c) == h)
    if k < 20:
        short.append(f"selective_avoid/{h} {k}/20")
print("[prep_items] short: " + ", ".join(short) if short else "[prep_items] every task has its demos")
sys.exit(1 if short else 0)
EOF
# one LMDB per task (train.sh / the config take a comma-separated list: $D/lmdb/items_safe_<task>), each packed once
# (PACK_DONE), so the packing (~1 s per 80-step episode, depth re-encoded) resumes across 1-hour jobs: a job with
# under 25 min left after a task exits 75 and the next one goes on
mkdir -p "$D/lmdb"
for t in selective_avoid push_zone tray_transport clean_pour gap_place; do
  o="${OUT}_$t"
  [ -f "$o/PACK_DONE" ] && continue
  rm -rf "$o"
  python -m vla_finetuning.pack --root "$ROOT" --out "$o" --tasks "$t" || exit 1
  date '+%F %T' > "$o/PACK_DONE"
  echo "[prep_items] packed $t: $(tr -d '\n ' < "$o/vf_pack.json" | cut -c1-300)"
  if [ -n "${STAGE_DEADLINE:-}" ] && [ $(( STAGE_DEADLINE - $(date +%s) )) -lt 1500 ]; then
    echo "[prep_items] packed up to $t; the rest in the next job"; exit 75
  fi
done
date '+%F %T' > "$D/PREP_ITEMS_DONE"
