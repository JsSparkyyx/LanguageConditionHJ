#!/bin/bash
# Run one command inside the robot-sim container, cwd = repo root.
#
#   scripts/lang/in_container.sh python -m main.lang.collect warmup ...
#
# The apptainer line of envs/scripts/item_envs/stage_runner.sbatch, with the
# repo root as cwd (main = <repo>/main; main/lang/__init__.py appends
# <repo>/envs/main for main.item_envs). `python` is the RoboTwin conda env.
# Arguments are passed through unchanged (quoting preserved).
set -u
REPO=/work/nvme/bifq/project/LanguageConditionHJ
PYENV=/work/nvme/bifq/conda/envs/RoboTwin
exec apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  PATH="$PYENV/bin:/usr/local/bin:/usr/bin:/bin" \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext \
  WANDB_MODE=offline PYTHONUNBUFFERED=1 \
  bash -c 'cd "$0" && exec "$@"' "$REPO" "$@"
