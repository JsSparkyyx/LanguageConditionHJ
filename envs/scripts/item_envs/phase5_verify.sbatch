#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=06:00:00
#SBATCH --job-name=p5-verify
#SBATCH -o /work/nvme/bifq/project/tmp/logs/p5verify_%A.out

# Phase 5 GPU verification (PLAN v3 §6 Phase 5):
# - test_phase5_regression.py: Phase-0 h traces must reproduce the
#   reference's reproducible contract — scene EXACT, 0.04 s cadence,
#   tick count +-2, deterministic head (>=15 ticks within 50 µm),
#   min/mean h within 3/1 cm, success parity (ur5-wsg 143 ticks
#   min_h=-0.0277 m; franka-panda 149 ticks min_h=-0.0388 m). NOT
#   byte-identity: same-node planner reruns diverge at tick ~30
#   (job 22624498, see the test docstring).
# - test_phase5_dryrun.py: 3 nominal episodes per item task x embodiment
#   through the real collect wiring; structural guards (lockstep teardown,
#   rule_valid width, ring parity) hard per episode; sidecar-vs-trace
#   margins compared as a trajectory contract (count +-8, divergence
#   onset >=3 ticks within 5 mm) with >=1 matching pair required —
#   strict equality is unachievable under measured planner nondeterminism
#   (job 22624498, see the test docstrings).

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c 'cd /work/nvme/bifq/project/LanguageConditionHJ/envs && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python -m pytest \
      tests/item_envs/test_phase5_regression.py tests/item_envs/test_phase5_dryrun.py \
      -m gpu -q --tb=short'
