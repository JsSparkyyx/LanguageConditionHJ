#!/usr/bin/env bash
# Submit the 5 leave-one-out splits for the VANILLA baseline (--vanilla =
# --ablate-geometry + --ablate-injection: plain cross-attention over frozen
# HoloBrain tokens, same heads). Ablated runs need their own checkpoints, so
# these get separate run dirs (loo_vanilla_*); the warmup buffer is shared
# read-only as usual.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /work/nvme/bifq/project/tmp/cehj/slurm

for emb in piper franka-panda ARX-X5 ur5-wsg aloha-agilex; do
  case "$emb" in
    franka-panda)  short=franka ;;
    ARX-X5)        short=arx ;;
    ur5-wsg)       short=ur5 ;;
    aloha-agilex)  short=aloha ;;
    *)             short="$emb" ;;
  esac
  sbatch --job-name="loov_${short}" \
    scripts/slurm_train_standard.sbatch "loo_vanilla_${short}" \
    --leave-out "$emb" --vanilla
done
