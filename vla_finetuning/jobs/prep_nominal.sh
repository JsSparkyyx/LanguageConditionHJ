#!/bin/bash
# Training data of the safety-agnostic VLA (inside the container; Yuxuan 2026-11-21, items/collect_nominal.py): the
# view of the reused careless demos (T1 / T3 / T7, items/nominal_view.py) and the collected ones (T6 / T8) in one root,
# packed per task into data/lmdb/nominal_<task> (PACK_DONE per task). Exit 0 when every task is packed, 75 while a
# collection shard is unfinished or a lookup failed (the caller waits or resubmits), 1 when packing failed. The prep
# script of ft_c_train.sbatch (PREP=...).
set -uo pipefail
R=/work/nvme/bifq/project/LanguageConditionHJ
VF=$R/vla_finetuning
ROOT=${NOMINAL_ROOT:-$VF/data/demos_nominal}; OUT=$VF/data/lmdb/nominal
TASKS="selective_avoid push_zone clean_pour gap_place tray_slide"
cd $R
shards=$(python -m vla_finetuning.items.collect_nominal --shards) && [ -n "$shards" ] \
  || { echo "[prep_nominal] shard lookup failed"; exit 75; }
short=0
for sh in $shards; do
  IFS=: read -r t n _ <<< "$sh"
  [ -f "${ROOT}_logs/$t/$n/DONE" ] || { echo "[prep_nominal] $t $n not done"; short=1; }
done
[ $short -eq 0 ] || exit 75
python -m vla_finetuning.items.nominal_view --out "$ROOT" || exit 1
for t in $TASKS; do
  n=$(cat $ROOT/$t/*/attempts.jsonl 2>/dev/null | grep -c . || true)
  echo "[prep_nominal] $t: $n demos"
done
mkdir -p "$VF/data/lmdb"
for t in $TASKS; do
  o="${OUT}_$t"
  [ -f "$o/PACK_DONE" ] && continue
  rm -rf "$o"
  python -m vla_finetuning.pack --root "$ROOT" --out "$o" --tasks "$t" || exit 1
  date '+%F %T' > "$o/PACK_DONE"
  echo "[prep_nominal] packed $t: $(tr -d '\n ' < "$o/vf_pack.json" | cut -c1-300)"
  if [ -n "${STAGE_DEADLINE:-}" ] && [ $(( STAGE_DEADLINE - $(date +%s) )) -lt 1500 ]; then
    echo "[prep_nominal] packed up to $t; the rest in the next job"; exit 75
  fi
done
echo "[prep_nominal] every task packed: $TASKS"
