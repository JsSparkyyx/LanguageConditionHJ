#!/usr/bin/env bash
# Submit the 5 leave-one-out splits for the FEATURE-INJECTION ablation
# (--ablate-injection: body tokens are the frozen HoloBrain state tokens
# alone, no analytic feature columns injected). Ablated runs need their own
# checkpoints, so these get separate run dirs (loo_feature_*); the warmup
# buffer is shared read-only as usual.
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
  sbatch --job-name="loof_${short}" \
    scripts/slurm_train_standard.sbatch "loo_feature_${short}" \
    --leave-out "$emb" --ablate-injection
done
