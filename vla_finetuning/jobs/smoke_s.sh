#!/bin/bash
# Sanity check (A40): the base HoloBrain model through our policy adapter on plain RoboTwin layouts
# (no stretch, no box), 2 scenes for each of 3 tasks: should behave as HoloBrain does on RoboTwin.
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
cd /work/nvme/bifq/project/LanguageConditionHJ
python -m vla_finetuning.evaluate --agent vla --scene standard --model-dir $VF/models/base_post_training_robotwin \
  --out $VF/data/smoke/results_s/base_standard --tasks place_container_plate stack_blocks_two place_burger_fries \
  --episodes 2 --video-every 1
echo "=== done $(date +%T)"
