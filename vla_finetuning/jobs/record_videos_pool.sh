#!/bin/bash
# One node's workers of record_videos.sbatch (inside the container): SLOTS loops on this node's GPU, each claiming the
# next (controller, task) of the 25 by mkdir in this job's claim dir and running record_videos on it until it finishes
# (0) or reaches the stage deadline (75). Each (controller, task) resumes from its attempts.jsonl.
#   bash vla_finetuning/jobs/record_videos_pool.sh <claim dir> <slots> <out> [items]
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
CL=$1; SLOTS=${2:-3}; OUT=$3
cd $R
# heavy first: the safe T1 expert (grasp search, ~14.5 GB), then the long model episodes (T3 1200 actions, T6 / T7 800)
ITEMS=${4:-"nominal_safe:selective_avoid nominal_careless:selective_avoid \
base:push_zone ft_items_safe:push_zone ft_items_safe_x2:push_zone \
base:clean_pour ft_items_safe:clean_pour ft_items_safe_x2:clean_pour \
base:selective_avoid ft_items_safe:selective_avoid ft_items_safe_x2:selective_avoid \
base:gap_place ft_items_safe:gap_place ft_items_safe_x2:gap_place \
base:tray_transport ft_items_safe:tray_transport ft_items_safe_x2:tray_transport \
nominal_safe:push_zone nominal_careless:push_zone nominal_safe:clean_pour nominal_careless:clean_pour \
nominal_safe:gap_place nominal_careless:gap_place nominal_safe:tray_transport nominal_careless:tray_transport"}
mkdir -p $OUT/_logs
slot() {
  local it c t rc
  for it in $ITEMS; do
    mkdir "$CL/$it" 2> /dev/null || continue
    c=${it%%:*}; t=${it#*:}
    echo "[record $(hostname -s) s$1] $(date '+%T') $c $t"
    python -m vla_finetuning.items.record_videos --controller $c --tasks $t --out $OUT \
      >> $OUT/_logs/${c}_${t}.log 2>&1
    rc=$?
    echo "[record $(hostname -s) s$1] $(date '+%T') $c $t rc=$rc: $(grep -c '^\[record\].*-> ' $OUT/_logs/${c}_${t}.log) scenes run so far"
  done
}
for s in $(seq 1 $SLOTS); do slot $s & sleep 20; done
wait
