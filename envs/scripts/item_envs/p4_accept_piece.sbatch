#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=01:00:00
#SBATCH --job-name=p4a-piece
#SBATCH -o /work/nvme/bifq/project/tmp/logs/p4apiece_%A.out

# Phase 4 acceptance, interactive-partition pieces (phase4_accept.sbatch split
# to fit the 1 h cap; previous batch runtimes: shard0 3m, shard1 2m, shard2 30m
# with early-failing tests — passing runs are slower, so shard 2 is halved).
# Submit sequentially, slot permitting:
#   sbatch --export=ALL,STAGE=t1-ur5    envs/scripts/item_envs/p4_accept_piece.sbatch
#   sbatch --export=ALL,STAGE=t1-franka envs/scripts/item_envs/p4_accept_piece.sbatch
#   sbatch --export=ALL,STAGE=t2t3      envs/scripts/item_envs/p4_accept_piece.sbatch
#   sbatch --export=ALL,STAGE=spawns    envs/scripts/item_envs/p4_accept_piece.sbatch
#   sbatch --export=ALL,STAGE=p5v       envs/scripts/item_envs/p4_accept_piece.sbatch  (phase5_verify confirm)

set -e
STAGE=${STAGE:?STAGE unset}
case "$STAGE" in
  t1-ur5)    TESTS="tests/item_envs/test_selective_avoid_sim.py -k ur5-wsg" ;;
  t1-franka) TESTS="tests/item_envs/test_selective_avoid_sim.py -k franka-panda" ;;
  t2t3)      TESTS="tests/item_envs/test_push_zone_sim.py tests/item_envs/test_granular_pour_sim.py" ;;
  spawns)    TESTS="tests/item_envs/test_spawn_sim.py tests/item_envs/test_p1_sim.py tests/item_envs/test_p3_robot_sim.py" ;;
  p5v)       TESTS="tests/item_envs/test_phase5_regression.py tests/item_envs/test_phase5_dryrun.py" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
MARK="sim"; [ "$STAGE" = p5v ] && MARK="gpu"
echo "[p4a-piece] stage=$STAGE tests='$TESTS'"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c "cd /work/nvme/bifq/project/LanguageConditionHJ/envs && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python -m pytest ${TESTS} -m ${MARK} -q --tb=short"
