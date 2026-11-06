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
# the run's model config: VF_CONFIG, else the "config" its workspace recorded (ws/<run>/vf_run.json, 4 levels above
# checkpoints/checkpoint_<k>/model/model.safetensors), else the GD copy (runs trained before the field)
if [ -z "${VF_CONFIG:-}" ]; then
  RJ=$(dirname "$(dirname "$(dirname "$(dirname "$CKPT")")")")/vf_run.json
  VF_CONFIG=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("config", ""))' "$RJ" 2>/dev/null || true)
fi
CFG=${VF_CONFIG:-configs/config_holobrain_gd_common.py}
test -f "$CFG" || { echo "no config $CFG"; exit 2; }
test ! -e "models/$NAME/model" || { echo "models/$NAME/model exists"; exit 2; }
mkdir -p "models/$NAME"
cp -r configs "models/$NAME/vf_configs"
echo "{\"checkpoint\": \"$CKPT\", \"config\": \"$CFG\", \"git\": \"$(git -C .. rev-parse --short HEAD 2>/dev/null || echo ?)\", \"at\": \"$(date '+%F %T')\"}" \
  > "models/$NAME/vf_export.json"
echo "[export] $NAME config $CFG"
exec python "$RL/projects/holobrain/scripts/export.py" --config "$CFG" \
  --workspace "models/$NAME" --kwargs "{\"checkpoint\": \"$CKPT\"}"
