#!/bin/bash
# Experiment matrix (VANILLA_PLAN 9.3). Prints the commands; --submit appends them as stages to a
# chain queue (A100 for offline training, A40 for rounds and closed-loop grids). THE HUMAN DECIDES
# WHAT TO LAUNCH: run without --submit first.
#   bash scripts/lang/matrix.sh [--submit] [--only full,shuffled,...] [--steps 50000]
set -u
SUBMIT=0; ONLY=""; STEPS=50000
while [ $# -gt 0 ]; do case "$1" in
  --submit) SUBMIT=1;; --only) ONLY=$2; shift;; --steps) STEPS=$2; shift;; *) echo "unknown $1"; exit 2;; esac; shift; done
ROOT=/work/nvme/bifq/project/LanguageConditionHJ
DATA=$ROOT/data/lang/warmup; VAL=$ROOT/data/lang/val
STAMP=$(date +%m%d-%H%M)
declare -A FLAGS=(
  [full]=""
  [fixedrule]="--relabel fixed"
  [randrelabel]="--relabel random"
  [shuffled]="--shuffle-text"
  [noscene]="--no-scene" [nobody]="--no-body" [nofilm]="--no-film" [noqueries]="--no-queries" [nortoken]="--no-rtoken"
  [bertraw]="--text-backend bert_raw" [generic]="--text-backend sentence_generic"
)
for t in selective_avoid granular_pour push_zone tray_transport gap_place; do   # T4 / T6 dropped 2026-10-22
  FLAGS[pertask_$t]="--tasks $t"
done
ORDER="full fixedrule randrelabel shuffled noscene nobody nofilm noqueries nortoken bertraw generic pertask_selective_avoid pertask_granular_pour pertask_push_zone pertask_tray_transport pertask_gap_place"
for name in $ORDER; do
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then continue; fi
  run=runs/lang/lang_${name}_off_${STAMP}
  extra=""
  case "$name" in
    bertraw) extra="--text-cache $DATA/text_cache_bert_raw.pt $VAL/text_cache_bert_raw.pt";;
    generic) extra="--text-cache $DATA/text_cache_sentence_generic.pt $VAL/text_cache_sentence_generic.pt";;
  esac
  cmd="python -m main.lang.train offline --data $DATA --val $VAL --run $run --grad-steps $STEPS --probe-every 5000 ${FLAGS[$name]} $extra --resume"
  probe="python -m main.lang.eval_offline --run $run --ckpt checkpoint_final.pt --val $VAL --states-per-task 2000 --paraphrases 3"
  echo "# $name"
  echo "$cmd && $probe"
  if [ $SUBMIT = 1 ]; then
    echo "mx_${name}|1|$cmd && $probe" >> $ROOT/runs/lang_pipeline_a100/queue.txt
  fi
done
echo "# rounds (only the variants the human picks; compare among rounds runs):"
echo "python -m main.lang.train rounds --data $DATA --val $VAL --run runs/lang/lang_full_paired_rounds_${STAMP} --collect-rounds 100 --episodes-per-round 4 --grad-steps-per-round 2048 --probe-every-rounds 10 --resume"
echo "# closed-loop grid per offline checkpoint (A40):"
echo "python -m main.lang.eval_closed --run <run> --ckpt checkpoint_final.pt --val $VAL --scenes-per-combo 20 --axes train,heldout_template"
