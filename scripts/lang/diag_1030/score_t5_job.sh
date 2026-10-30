#!/bin/bash
# Score check (diag 2026-10-30): score every existing retrain checkpoint (checkpoint.pt = latest, checkpoint_step*.pt,
# checkpoint_final.pt) and the vanilla run's checkpoints on check 2's T5 branch states, then aggregate.
# Inside the container:  scripts/lang/in_container.sh bash scripts/lang/diag_1030/score_t5_job.sh
set -u
cd /work/nvme/bifq/project/LanguageConditionHJ
R=runs/lang/diag_1030/retrain
V=runs/lang/rounds_ur5_g095_1023
shopt -s nullglob
ck=($V/checkpoint.pt $V/checkpoint_round10.pt $V/checkpoint_round20.pt $V/checkpoint_round30.pt)
for v in stoch det data zero stoch_norm stoch_mlp; do
  ck+=($R/$v/checkpoint.pt $R/$v/checkpoint_step*.pt $R/$v/checkpoint_final.pt)
done
echo "[score_t5_job] $(date '+%T') ${#ck[@]} checkpoints"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/lang/diag_1030/score_t5.py score --ckpt "${ck[@]}"
echo "[score_t5_job] $(date '+%T') score rc=$?"
python scripts/lang/diag_1030/score_t5.py aggregate
echo "[score_t5_job] $(date '+%T') aggregate rc=$?"
