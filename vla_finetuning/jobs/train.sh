#!/bin/bash
# Fine-tune HoloBrain with its own train.py (inside the container; cwd = vla_finetuning/).
#
#   scripts/lang/in_container.sh bash vla_finetuning/jobs/train.sh <run> <lmdb[,lmdb...]> <max_step> <save_every> [n_gpu] [resume_checkpoint_dir]
#
# Workspace: vla_finetuning/ws/<run>/ (checkpoints/checkpoint_<k>/model/model.safetensors, the last 3 kept).
# A resume writes to a fresh workspace ws/<run>_r<unix time> with resume_from = the old checkpoint
# directory: accelerate's checkpoint numbering restarts at 0 after load_state, and reusing the workspace
# deletes the newest checkpoints, then crashes (review 10-15, train A1).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=$1
export VF_LMDB=$2
MAX=$3
SAVE=$4
N=${5:-1}
RESUME=${6:-}
RL=${ROBO_ORCHARD_LAB:-/work/nvme/bifq/project/RoboOrchardLab}
export PYTHONPATH="$PWD/pyshim:$RL:${PYTHONPATH:-}"   # pyshim: accelerate 1.14 DummyOptim re-export
# GRAD_ACCUM (default 1): micro-batches per optimizer step (accelerate reads the variable in Accelerator(); its backward
# divides the loss by it; HoloBrain's trainer counts max_step / save_step_freq and steps the LR schedule in committed
# optimizer steps), so one GPU with GRAD_ACCUM=4 trains the 4-GPU recipe's batch of 32 (Yuxuan 2026-10-28: one GPU)
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=${GRAD_ACCUM:-1}
KW="{\"max_step\": $MAX, \"save_step_freq\": $SAVE"
# INIT_CKPT: start from these weights instead of the config's post_training_robotwin (a longer training continued
# from a fine-tuned model with a fresh LR schedule; 10-29); a resume still restores the run's own state over them
if [ -n "${INIT_CKPT:-}" ]; then
  test -f "$INIT_CKPT" || { echo "no INIT_CKPT $INIT_CKPT"; exit 2; }
  KW="$KW, \"checkpoint\": \"$INIT_CKPT\""
fi
WS="ws/$RUN"
if [ -n "$RESUME" ]; then
  test -d "$RESUME" || { echo "no checkpoint dir $RESUME"; exit 2; }
  KW="$KW, \"resume_from\": \"$RESUME\""
  WS="ws/${RUN}_r$(date +%s)"
elif [ -d "$WS/checkpoints" ]; then
  echo "$WS already has checkpoints: pass a resume checkpoint dir or use another run name"; exit 2
fi
KW="$KW}"
mkdir -p "$WS"
cp -r configs "$WS/vf_configs"                     # train.py copies HoloBrain's upstream configs, not ours
cat > "$WS/vf_run.json" <<EOF
{"run": "$RUN", "lmdb": "$VF_LMDB", "kwargs": $KW, "n_gpu": $N, "grad_accum": ${GRAD_ACCUM:-1}, "init_ckpt": "${INIT_CKPT:-}", "resume": "$RESUME",
 "git": "$(git -C .. rev-parse --short HEAD 2>/dev/null || echo ?)", "host": "$(hostname)", "start": "$(date '+%F %T')"}
EOF
echo "[train] $WS lmdb $VF_LMDB kwargs $KW gpus $N grad_accum ${GRAD_ACCUM:-1} host $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [ "$N" -gt 1 ]; then
  exec accelerate launch --multi_gpu --num_processes "$N" --num_machines 1 --mixed_precision no \
    --dynamo_backend no "$RL/projects/holobrain/scripts/train.py" \
    --config configs/config_holobrain_gd_common.py --workspace "$WS" --kwargs "$KW"
else
  exec python "$RL/projects/holobrain/scripts/train.py" \
    --config configs/config_holobrain_gd_common.py --workspace "$WS" --kwargs "$KW"
fi
