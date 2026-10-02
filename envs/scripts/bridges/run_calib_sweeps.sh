#!/usr/bin/env bash
# V4 — Phase 4 calibration sweeps (PLAN v3 section 8): adaptive risk grids,
# no-render mode. 14 sweeps total (T1: 5 hosts x 2 embodiments; T2/T3: 2
# embodiments each). Sequential by default.
#
# Shard across GPUs/terminals:  SHARD=i/n scripts/bridges/run_calib_sweeps.sh
#   e.g. 4 terminals: SHARD=0/4 ... SHARD=3/4, each with its own
#   CUDA_VISIBLE_DEVICES set. ALL sweeps MUST run on the same GPU model
#   (planner results are hardware-dependent at the cm level).
# Per sweep: cells x 5 episodes + 15 adaptive in mid-band cells; worst case
# ~4 h on A40-class hardware; total ~10-20 GPU-hours.
set -euo pipefail
ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${ENVS_DIR}/.." && pwd)"
source "${ENVS_DIR}/scripts/bridges/_env.sh"
cd "${ENVS_DIR}"

SHARD="${SHARD:-0/1}"
I="${SHARD%%/*}"; N="${SHARD##*/}"

GRID_T1="time_scale=0.6,0.8,1.0,1.15,1.3 on_path_count=1,2"
GRID_T2="time_scale=0.5,0.75,1.0,1.15,1.3 tilt_rate=1,2.5,4"
GRID_T3="time_scale=0.5,0.75,1.0,1.15,1.3 overshoot=0,0.04,0.08"

SWEEPS=()
for host in place_container_plate place_burger_fries stack_blocks_two stack_bowls_two place_bread_basket; do
  for emb in ur5-wsg franka-panda; do
    SWEEPS+=("selective_avoid|${emb}|--host ${host}|${GRID_T1}|T1")
  done
done
SWEEPS+=("granular_pour|ur5-wsg||${GRID_T2}|T2")
SWEEPS+=("granular_pour|franka-panda||${GRID_T2}|T2")
SWEEPS+=("push_zone|ur5-wsg||${GRID_T3}|T3")
SWEEPS+=("push_zone|franka-panda||${GRID_T3}|T3")

for idx in "${!SWEEPS[@]}"; do
  (( idx % N == I )) || continue
  IFS='|' read -r task emb extra grid out <<< "${SWEEPS[idx]}"
  echo "=== [calib] sweep $((idx + 1))/${#SWEEPS[@]}: ${task} ${emb} ${extra} (${grid})"
  # shellcheck disable=SC2086
  "${PY}" -m main.item_envs.probe sweep \
    --task "${task}" --embodiment "${emb}" ${extra} \
    --grid ${grid} --episodes 5 --adaptive --adaptive-episodes 15 \
    --render none --out "runs/item_envs/calib/${out}" \
    2>&1 | tee "${LOG_DIR}/calib_${out}_${task}_${emb}$(echo " ${extra}" | tr ' /' '__').log"
done
echo "=== [calib] shard ${SHARD} DONE"
