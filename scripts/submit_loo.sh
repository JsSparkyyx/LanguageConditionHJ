#!/usr/bin/env bash
# Submit the 5 leave-one-out (phase 1) splits: train on 4 embodiments,
# cross-eval on the held-out one every 10 epochs. Each job seeds its own
# node-local /tmp buffer from the shared warmup (see slurm_train_standard.sbatch).
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
  sbatch --job-name="loos_${short}" \
    scripts/slurm_train.sbatch "loos_${short}" --leave-out "$emb"
done
