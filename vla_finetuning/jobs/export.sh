#!/bin/bash
# Export a fine-tuned checkpoint with HoloBrain's export.py (inside the container; cwd = vla_finetuning/).
#
#   scripts/lang/in_container.sh bash vla_finetuning/jobs/export.sh <name> <ws/<run>/checkpoints/checkpoint_<k>/model/model.safetensors>
#
# Model directory for evaluate.py: vla_finetuning/models/<name>/model (weights, configs, urdf/, ckpt/bert).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1
CKPT=$2
RL=${ROBO_ORCHARD_LAB:-/work/nvme/bifq/project/RoboOrchardLab}
export PYTHONPATH="$PWD/pyshim:$RL:${PYTHONPATH:-}"   # pyshim: accelerate 1.14 DummyOptim re-export
export VF_LMDB=${VF_LMDB:-}
test -f "$CKPT" || { echo "no checkpoint $CKPT"; exit 2; }
test ! -e "models/$NAME/model" || { echo "models/$NAME/model exists"; exit 2; }
mkdir -p "models/$NAME"
cp -r configs "models/$NAME/vf_configs"
echo "{\"checkpoint\": \"$CKPT\", \"git\": \"$(git -C .. rev-parse --short HEAD 2>/dev/null || echo ?)\", \"at\": \"$(date '+%F %T')\"}" \
  > "models/$NAME/vf_export.json"
exec python "$RL/projects/holobrain/scripts/export.py" --config configs/config_holobrain_gd_common.py \
  --workspace "models/$NAME" --kwargs "{\"checkpoint\": \"$CKPT\"}"
