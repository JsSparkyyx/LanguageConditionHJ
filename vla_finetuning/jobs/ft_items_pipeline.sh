#!/bin/bash
# The item-scene fine-tune end to end, one interactive job at a time (stage ft_items of runs/lang_pipeline_ft; inside the
# container, STAGE_DEADLINE set by the stage): train (ft_chain_step.sh, resumed from the newest checkpoint) until
# ws/ft_items_safe.DONE, export it to models/ft_items_safe/model, then test the base model and the fine-tuned one on the
# same item scenes (items/evaluate_items.py, 20 episodes per task, resumable). Exit 0 when both tests are complete,
# 75 to go on in the next job, anything else = a failure.
#   bash vla_finetuning/jobs/ft_items_pipeline.sh
set -uo pipefail
R=/work/nvme/bifq/project/LanguageConditionHJ; VF=$R/vla_finetuning; WS=$VF/ws
cd "$R"
L=$(printf "$VF/data/lmdb/items_safe_%s," selective_avoid push_zone tray_transport clean_pour gap_place); L=${L%,}
if [ ! -f "$WS/ft_items_safe.DONE" ]; then
  bash vla_finetuning/jobs/ft_chain_step.sh ft_items_safe "$L" "${MAX_STEP:-15000}" "${SAVE_EVERY:-500}" 1 \
    vla_finetuning/jobs/prep_items.sh
  rc=$?
  [ $rc -eq 0 ] || exit $rc
fi
CK=$(head -1 "$WS/ft_items_safe.DONE")
[ -n "$CK" ] && [ -f "$CK/model/model.safetensors" ] || { echo "[pipeline] DONE names no checkpoint: '$CK'"; exit 1; }
M=$VF/models/ft_items_safe
if [ ! -f "$M/model/model.safetensors" ]; then
  rm -rf "$M"                                      # an export cut by a deadline
  bash vla_finetuning/jobs/export.sh ft_items_safe "$CK/model/model.safetensors" || exit 1
fi
for spec in "base:$VF/models/base_post_training_robotwin" "ft_items_safe:$M/model"; do
  name=${spec%%:*}; dir=${spec#*:}
  [ -f "$VF/results_items/$name/summary.json" ] && continue
  python -m vla_finetuning.items.evaluate_items --model-dir "$dir" --model-name "$name" \
    --out "$VF/results_items/$name" --episodes "${EVAL_EPISODES:-20}" --video-every 10
  rc=$?
  [ $rc -eq 0 ] || exit $rc
done
echo "[pipeline] done: $VF/results_items/{base,ft_items_safe}/summary.json"
exit 0
