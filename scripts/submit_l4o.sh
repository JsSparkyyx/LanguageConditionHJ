#!/usr/bin/env bash
# Submit the 5 leave-4-out splits: train on ONE embodiment (from the shared
# warmup buffer, filtered at sample time), cross-eval on the other 4.
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
  sbatch --job-name="l4o_${short}" \
    scripts/slurm_train_standard.sbatch "l4o_${short}" --only-embodiment "$emb"
done
