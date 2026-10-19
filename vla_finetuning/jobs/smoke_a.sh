#!/bin/bash
# Smoke test A (inside the container, one A40, under 1 h): setup, the pack round trip, one safe and one
# nominal demonstration, packing, and one test scene for the base model and both planners.
#   scripts/lang/in_container.sh bash vla_finetuning/jobs/smoke_a.sh
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
S=$VF/data/smoke
T=place_container_plate
cd /work/nvme/bifq/project/LanguageConditionHJ
step() { echo; echo "=== $1 $(date +%T)"; }
step setup;           python -m vla_finetuning.setup_models || exit 1
step roundtrip;       python -m pytest -q -p no:cacheprovider vla_finetuning/tests/test_demo_io.py
step collect_safe;    python -m vla_finetuning.collect_demos --task $T --mode safe --root $S/demos_safe \
                        --config-name smoke --target 1 --max-attempts 4 --video-every 1 --rejected-videos 4
step collect_nominal; python -m vla_finetuning.collect_demos --task $T --mode nominal --root $S/demos_nominal \
                        --config-name smoke --target 1 --max-attempts 3 --video-every 1
step pack;            rm -rf "$S/lmdb_safe"; python -m vla_finetuning.pack --root $S/demos_safe --out $S/lmdb_safe
step eval_base;       python -m vla_finetuning.evaluate --agent vla --model-dir $VF/models/base_post_training_robotwin \
                        --out $S/results/base --tasks $T --episodes 1 --video-every 1
step eval_nominal;    python -m vla_finetuning.evaluate --agent nominal --out $S/results/nominal --tasks $T --episodes 1 --video-every 1
step eval_safe;       python -m vla_finetuning.evaluate --agent safe --out $S/results/safe --tasks $T --episodes 1 --video-every 1
step done
