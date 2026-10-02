#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=08:00:00
#SBATCH --job-name=p4-accept
#SBATCH --array=0-2
#SBATCH -o /work/nvme/bifq/project/tmp/logs/p4accept_%A_%a.out

# Phase 4 GPU acceptance wave (PLAN v3 §6 Phase 4 step 3): sim-marked tests.
# shard 0/1: T1 selective_avoid sim grid per embodiment; shard 2: T2/T3 task
# sim + the spawn/P1/P3-robot sim suites loosened by spawn.py Fix 1.

case "$SLURM_ARRAY_TASK_ID" in
  0) TESTS="tests/item_envs/test_selective_avoid_sim.py -k ur5-wsg" ;;
  1) TESTS="tests/item_envs/test_selective_avoid_sim.py -k franka-panda" ;;
  2) TESTS="tests/item_envs/test_push_zone_sim.py tests/item_envs/test_granular_pour_sim.py tests/item_envs/test_spawn_sim.py tests/item_envs/test_p1_sim.py tests/item_envs/test_p3_robot_sim.py" ;;
esac

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c "cd /work/nvme/bifq/project/LanguageConditionHJ/envs && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python -m pytest ${TESTS} -m sim -q --tb=short"
