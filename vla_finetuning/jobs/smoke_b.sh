#!/bin/bash
# Smoke test B (inside the container, one A40, under 1 h): the base HoloBrain policy on one test scene
# (the local policy adapter), and a few safe-demonstrator attempts per task with the hand model in cuRobo
# (yield and rejection reasons), then one safe-planner test scene.
#   scripts/lang/in_container.sh bash vla_finetuning/jobs/smoke_b.sh
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
S=$VF/data/smoke
cd /work/nvme/bifq/project/LanguageConditionHJ
step() { echo; echo "=== $1 $(date +%T)"; }
step eval_base
python -m vla_finetuning.evaluate --agent vla --model-dir $VF/models/base_post_training_robotwin \
  --out $S/results_b/base --tasks place_container_plate --episodes 1 --video-every 1
for T in place_container_plate stack_blocks_two place_burger_fries stack_bowls_two place_bread_basket; do
  step "safe_diag $T"
  python -m vla_finetuning.collect_demos --task $T --mode safe --root $S/demos_safe_b --config-name diag \
    --target 2 --max-attempts 3 --video-every 1 --rejected-videos 3
done
step eval_safe
python -m vla_finetuning.evaluate --agent safe --out $S/results_b/safe --tasks stack_blocks_two --episodes 1 --video-every 1
step done
