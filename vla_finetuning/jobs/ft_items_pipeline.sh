#!/bin/bash
# The item-scene fine-tune end to end, one interactive job at a time (stage ft_items of runs/lang_pipeline_ft; inside the
# container, STAGE_DEADLINE set by the stage; every step resumes where the last job stopped):
#   1. train ft_items_safe (15k steps, ft_chain_step.sh), export it, test the base model and it (20 scenes per task,
#      items/evaluate_items.py);
#   2. unless it is already good, test its step-7500 checkpoint on the same first 10 scenes per task;
#   3. while ft_items_decide.py says EXTEND k: train ft_items_safe_x<k> (15k more steps from the previous model's
#      weights, a fresh LR schedule; Yuxuan 10-29: "if it's not good, do a longer training from the trained ckpt"),
#      export, test (20 per task), decide again.
# The rule and every model's numbers: results_items/decision.json. Exit 0 when the decision is STOP, 75 to go on in the
# next job, anything else = a failure.
#   bash vla_finetuning/jobs/ft_items_pipeline.sh
set -uo pipefail
R=/work/nvme/bifq/project/LanguageConditionHJ; VF=$R/vla_finetuning; WS=$VF/ws; RES=$VF/results_items
cd "$R"
L=$(printf "$VF/data/lmdb/items_safe_%s," selective_avoid push_zone tray_transport clean_pour gap_place); L=${L%,}
DEC="python vla_finetuning/jobs/ft_items_decide.py"

train_run() {   # <run> [initial weights]: train to <run>.DONE
  [ -f "$WS/$1.DONE" ] && return 0
  $DEC prune "$1"                                  # keep every 2500th step and the newest (2 GB each)
  INIT_CKPT="${2:-}" bash vla_finetuning/jobs/ft_chain_step.sh "$1" "$L" "${MAX_STEP:-15000}" "${SAVE_EVERY:-500}" 1 \
    vla_finetuning/jobs/prep_items.sh
}
final_ckpt() {  # <run>: the checkpoint dir its DONE names
  local ck; ck=$(head -1 "$WS/$1.DONE" 2>/dev/null)
  [ -n "$ck" ] && [ -f "$ck/model/model.safetensors" ] && echo "$ck"
}
export_model() {   # <name> <checkpoint dir> -> models/<name>/model
  [ -f "$VF/models/$1/model/model.safetensors" ] && return 0
  rm -rf "$VF/models/$1"                           # an export cut by a deadline
  bash vla_finetuning/jobs/export.sh "$1" "$2/model/model.safetensors"
}
eval_model() {     # <name> <model dir> <scenes per task>
  [ -f "$RES/$1/summary.json" ] && return 0
  python -m vla_finetuning.items.evaluate_items --model-dir "$2" --model-name "$1" --out "$RES/$1" --episodes "$3" \
    --video-every 10
}

# 1. the 15k-step run
train_run ft_items_safe || exit $?
CK=$(final_ckpt ft_items_safe) || { echo "[pipeline] ft_items_safe.DONE names no checkpoint"; exit 1; }
export_model ft_items_safe "$CK" || exit 1
eval_model base "$VF/models/base_post_training_robotwin" 20 || exit $?
eval_model ft_items_safe "$VF/models/ft_items_safe/model" 20 || exit $?
v=$($DEC next)
# 2. still improving? its step-7500 checkpoint on the same first scenes
if [ "$v" = WAIT ]; then
  CKM=$($DEC ckpt-at-step ft_items_safe 7500) || { echo "[pipeline] no complete step-7500 checkpoint"; exit 1; }
  export_model ft_items_safe_s7500 "$CKM" || exit 1
  eval_model ft_items_safe_s7500 "$VF/models/ft_items_safe_s7500/model" 10 || exit $?
  v=$($DEC next)
fi
# 3. longer training, 15k steps at a time from the previous model
while [[ "$v" == EXTEND* ]]; do
  k=${v#EXTEND }
  if [ "$k" = 2 ]; then prev=ft_items_safe; else prev=ft_items_safe_x$((k - 1)); fi
  run=ft_items_safe_x$k
  PCK=$(final_ckpt "$prev") || { echo "[pipeline] no final checkpoint of $prev"; exit 1; }
  echo "[pipeline] $v: $run from $prev ($PCK)"
  train_run "$run" "$PCK/model/model.safetensors" || exit $?
  CK=$(final_ckpt "$run") || { echo "[pipeline] $run.DONE names no checkpoint"; exit 1; }
  export_model "$run" "$CK" || exit 1
  eval_model "$run" "$VF/models/$run/model" 20 || exit $?
  v=$($DEC next)
done
[ "$v" = STOP ] || { echo "[pipeline] unexpected decision '$v'"; exit 1; }
echo "[pipeline] done: STOP, best $(python -c "import json; print(json.load(open('$RES/decision.json'))['best'])") ($RES/decision.json)"
exit 0
