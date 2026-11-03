#!/bin/bash
# Training data of the constraint-following fine-tune (inside the container; Yuxuan 2026-11-03,
# items/collect_constraint.py): exit 0 when every shard of the active tasks is DONE and every active task is packed
# (data/lmdb/constraint_<task>/PACK_DONE, checked per task: a task added later gets packed then); exit 75 while
# shards are unfinished or a lookup failed (the caller waits or resubmits), 1 when packing failed.
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
D=$VF/data; ROOT=$D/demos_constraint; OUT=$D/lmdb/constraint
cd /work/nvme/bifq/project/LanguageConditionHJ
tasks=$(python -m vla_finetuning.items.constraint --active) && [ -n "$tasks" ] \
  || { echo "[prep_constraint] task lookup failed"; exit 75; }
table=$(python -m vla_finetuning.items.constraint) && [ -n "$table" ] \
  || { echo "[prep_constraint] shard lookup failed"; exit 75; }
short=0
while read -r task host shard target start step stop; do
  [ -f "${ROOT}_logs/$task/$shard/DONE" ] || { echo "[prep_constraint] $task $shard not done"; short=1; }
done <<< "$table"
[ $short -eq 0 ] || exit 75
# the demos per task and kind (the collector exits 0 also when its seeds ran out: a shortfall shows here)
python - "$ROOT" $tasks <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for t in sys.argv[2:]:
    n = {}
    for cfg in sorted((root / t).glob("*_*")):
        kind = "A" if cfg.name.endswith("_A") else "B"
        p = cfg / "attempts.jsonl"
        k = sum(1 for l in p.read_text().splitlines() if l.strip()) if p.exists() else 0
        n[kind] = n.get(kind, 0) + k
    print(f"[prep_constraint] {t}: {n.get('A', 0)} careless demos, {n.get('B', 0)} careful demos")
EOF
mkdir -p "$D/lmdb"
for t in $tasks; do
  o="${OUT}_$t"
  [ -f "$o/PACK_DONE" ] && continue
  rm -rf "$o"
  python -m vla_finetuning.pack --root "$ROOT" --out "$o" --tasks "$t" || exit 1
  date '+%F %T' > "$o/PACK_DONE"
  echo "[prep_constraint] packed $t: $(tr -d '\n ' < "$o/vf_pack.json" | cut -c1-300)"
  if [ -n "${STAGE_DEADLINE:-}" ] && [ $(( STAGE_DEADLINE - $(date +%s) )) -lt 1500 ]; then
    echo "[prep_constraint] packed up to $t; the rest in the next job"; exit 75
  fi
done
echo "[prep_constraint] every active task packed: $tasks"
