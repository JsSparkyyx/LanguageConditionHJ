#!/bin/bash
# Smoke test B4 (A40, under 1 h): the safe demonstrator with the start-collision fallback (one plan retried
# without the held object, then without the hand, when cuRobo finds its start in collision): 6 attempts per
# task, yield and reasons.
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
S=$VF/data/smoke
cd /work/nvme/bifq/project/LanguageConditionHJ
step() { echo; echo "=== $1 $(date +%T)"; }
for T in place_container_plate stack_blocks_two place_burger_fries stack_bowls_two place_bread_basket; do
  step "safe_diag $T"
  python -m vla_finetuning.collect_demos --task $T --mode safe --root $S/demos_safe_b4 --config-name diag \
    --target 6 --max-attempts 6 --video-every 2 --rejected-videos 2
done
step done
