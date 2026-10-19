#!/bin/bash
# Smoke test B3 (A40, under 1 h): the safe demonstrator with the box-only world and inscribed hand/payload
# spheres (4 attempts per task: yield, reasons, cuRobo statuses), then the base-model sanity check on
# plain RoboTwin layouts (1 scene each for 3 tasks).
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
S=$VF/data/smoke
cd /work/nvme/bifq/project/LanguageConditionHJ
step() { echo; echo "=== $1 $(date +%T)"; }
for T in place_container_plate stack_blocks_two place_burger_fries stack_bowls_two place_bread_basket; do
  step "safe_diag $T"
  python -m vla_finetuning.collect_demos --task $T --mode safe --root $S/demos_safe_b3 --config-name diag \
    --target 4 --max-attempts 4 --video-every 2 --rejected-videos 2
done
step sanity
python -m vla_finetuning.evaluate --agent vla --scene standard --model-dir $VF/models/base_post_training_robotwin \
  --out $S/results_s/base_standard --tasks place_container_plate stack_blocks_two place_burger_fries \
  --episodes 1 --video-every 1
step done
