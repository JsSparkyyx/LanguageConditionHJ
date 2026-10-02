#!/usr/bin/env bash
# V3 — Phase 5 verification: robust regression contract + dry-run sidecar.
# NOTE: these tests are ROBUST TO planner nondeterminism by design (cuRobo
# plans differ run-to-run on ANY hardware). A failure with a chaos signature
# (count/onset stats, >=1-of-3 pairs) is a known flake — rerun once. A
# structural failure (ALL pairs diverge at tick 0/1, lockstep errors) is real.
# Expected runtime ~30 min. Usage: scripts/bridges/run_phase5_verify.sh
set -euo pipefail
ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${ENVS_DIR}/.." && pwd)"
source "${ENVS_DIR}/scripts/bridges/_env.sh"
cd "${ENVS_DIR}"

echo "=== [p5v] regression + dryrun"
"${PY}" -m pytest tests/item_envs/test_phase5_regression.py tests/item_envs/test_phase5_dryrun.py \
  -m gpu -q --tb=short 2>&1 | tee "${LOG_DIR}/phase5_verify.log"
echo "=== [p5v] DONE — log ${LOG_DIR}/phase5_verify.log"
