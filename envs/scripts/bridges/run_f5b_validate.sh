#!/usr/bin/env bash
# V1 — F5b re-validation (P4 visibility gate): 20 seeds x 5 hosts x both
# embodiments, setup-only; per-class gate frac(vis_max >= 0.3) >= 0.95.
# Existing validate rows are moved aside FIRST (resume keys would skip them).
# Expected runtime ~15 min. Usage: scripts/bridges/run_f5b_validate.sh
set -euo pipefail
ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${ENVS_DIR}/.." && pwd)"
source "${ENVS_DIR}/scripts/bridges/_env.sh"

ROWS="${ENVS_DIR}/main/item_envs/configs/validate"
mkdir -p "${ROWS}"
stamp="$(date +%Y%m%d_%H%M%S)"
shopt -s nullglob
for f in "${ROWS}"/visibility_rows_*.jsonl; do
  mv -v "${f}" "${f%.jsonl}_aside_${stamp}.jsonl"
done

cd "${REPO_ROOT}"
echo "=== [f5b] validate both embodiments"
"${PY}" envs/scripts/item_envs/visibility_band.py --embodiment both --validate \
  2>&1 | tee "${LOG_DIR}/f5b_validate.log"
echo "=== [f5b] DONE — summary at the end of ${LOG_DIR}/f5b_validate.log"
