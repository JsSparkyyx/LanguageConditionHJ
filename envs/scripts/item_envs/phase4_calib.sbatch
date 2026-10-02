#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=08:00:00
#SBATCH --job-name=p4-calib
#SBATCH --array=0-13
#SBATCH -o /work/nvme/bifq/project/tmp/logs/p4calib_%A_%a.out

# Phase 4 calibration (PLAN v3 section 8): adaptive risk-grid sweeps in
# no-render mode. One array task per (task-variant, embodiment) shard; the
# sweep parent itself spawns one child process per embodiment.
#   0-4 : T1 selective_avoid x ur5-wsg      x host[0..4]
#   5-9 : T1 selective_avoid x franka-panda x host[0..4]
#   10  : T2 granular_pour x ur5-wsg
#   11  : T2 granular_pour x franka-panda
#   12  : T3 push_zone     x ur5-wsg
#   13  : T3 push_zone     x franka-panda
# Grid: time_scale (5 yaml values) x main task knob (3 points; T1 has 2),
# 5 episodes per cell + 15 adaptive in cells with primary violation rate in
# [0.1, 0.6]. Worst case per shard: 15 cells x 20 ep x ~50 s = 4.2 h.

HOSTS=(place_container_plate place_burger_fries stack_blocks_two stack_bowls_two place_bread_basket)
GRID_T1="time_scale=0.6,0.8,1.0,1.15,1.3 on_path_count=1,2"
GRID_T2="time_scale=0.5,0.75,1.0,1.15,1.3 tilt_rate=1,2.5,4"
GRID_T3="time_scale=0.5,0.75,1.0,1.15,1.3 overshoot=0,0.04,0.08"

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 10 ]; then
  TASK=selective_avoid; OUT=T1; GRID=$GRID_T1
  if [ "$i" -lt 5 ]; then EMB=ur5-wsg;        HOST=${HOSTS[$i]};      HOSTIDX=$i
  else                 EMB=franka-panda;  HOSTIDX=$((i - 5)); HOST=${HOSTS[$HOSTIDX]}; fi
  EXTRA="--host $HOST"
else
  case "$i" in
    10) TASK=granular_pour; OUT=T2; GRID=$GRID_T2; EMB=ur5-wsg ;;
    11) TASK=granular_pour; OUT=T2; GRID=$GRID_T2; EMB=franka-panda ;;
    12) TASK=push_zone;     OUT=T3; GRID=$GRID_T3; EMB=ur5-wsg ;;
    13) TASK=push_zone;     OUT=T3; GRID=$GRID_T3; EMB=franka-panda ;;
  esac
  EXTRA=""
fi

echo "[p4-calib] shard $i: task=$TASK emb=$EMB $EXTRA grid='$GRID'"

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c "cd /work/nvme/bifq/project/LanguageConditionHJ/envs && \
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python -m main.item_envs.probe sweep \
      --task $TASK --embodiment $EMB $EXTRA \
      --grid $GRID --episodes 5 --adaptive --adaptive-episodes 15 \
      --render none --out runs/item_envs/calib/$OUT"
