#!/usr/bin/env bash
# V2 — Phase 4 acceptance: sim test suites for T1/T2/T3 + spawn/P1/P3-robot.
# One GPU node. Expected runtime ~40-60 min total on A40-class hardware.
# Usage: scripts/bridges/run_accept.sh   (env: PY, HOLOBRAIN_CKPT, ROBOTWIN_ROOT)
set -euo pipefail
ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${ENVS_DIR}/.." && pwd)"
source "${ENVS_DIR}/scripts/bridges/_env.sh"
cd "${ENVS_DIR}"

run() {  # run <stage> <pytest args...>
  local stage="$1"; shift
  echo "=== [accept] ${stage}: $*"
  "${PY}" -m pytest "$@" -m sim -q --tb=short 2>&1 | tee "${LOG_DIR}/accept_${stage}.log"
}

run t1-ur5    tests/item_envs/test_selective_avoid_sim.py -k ur5-wsg
run t1-franka tests/item_envs/test_selective_avoid_sim.py -k franka-panda
run t2t3      tests/item_envs/test_push_zone_sim.py tests/item_envs/test_granular_pour_sim.py
run spawns    tests/item_envs/test_spawn_sim.py tests/item_envs/test_p1_sim.py tests/item_envs/test_p3_robot_sim.py
echo "=== [accept] ALL STAGES DONE — logs in ${LOG_DIR}"
