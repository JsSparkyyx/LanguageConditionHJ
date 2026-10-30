#!/bin/bash
# Check 4: score every retrain checkpoint (checkpoint_step*.pt, checkpoint_final.pt) and the vanilla run's checkpoint
# on check 2's T5 branch states that are not scored yet (inside the container; one scorer at a time via a lock dir).
#   scripts/lang/in_container.sh bash scripts/lang/diag_1030/score_new.sh
set -u
cd /work/nvme/bifq/project/LanguageConditionHJ
R=runs/lang/diag_1030/retrain
OUT=$R/scores_t5
ST=runs/lang/diag_1030/t5_branch/states
mkdir -p $OUT
LOCK=$OUT/score.lock
if ! mkdir $LOCK 2> /dev/null; then
  # a lock older than 50 min is stale (a killed job)
  [ $(( $(date +%s) - $(stat -c %Y $LOCK) )) -gt 3000 ] && rm -rf $LOCK && mkdir $LOCK || { echo "[score_new] busy"; exit 0; }
fi
trap 'rm -rf $LOCK' EXIT
shopt -s nullglob
states=($ST/*.pt)
[ ${#states[@]} -eq 0 ] && { echo "[score_new] no states yet"; exit 0; }
ckpts=(runs/lang/rounds_ur5_g095_1023/checkpoint.pt $R/*/checkpoint_step*.pt $R/*/checkpoint_final.pt)
for ck in "${ckpts[@]}"; do
  tag="$(basename $(dirname $ck))__$(basename $ck .pt)"
  todo=()
  for s in "${states[@]}"; do [ -f "$OUT/$(basename $s .pt)__$tag.npz" ] || todo+=("$s"); done
  [ ${#todo[@]} -eq 0 ] && continue
  echo "[score_new] $(date '+%T') $tag on ${#todo[@]} states file(s)"
  python scripts/lang/diag_1030/score_states.py --states "${todo[@]}" --gt V95 --ckpt "$ck" --out $OUT 2>&1 \
    | grep -E "^\[score\]|Error|error" | tail -40
done
echo "[score_new] $(date '+%T') done"
