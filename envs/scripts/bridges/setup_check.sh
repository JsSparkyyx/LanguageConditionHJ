#!/usr/bin/env bash
# Sanity checks before any GPU run on Bridges. No GPU work is performed.
set -euo pipefail
ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${ENVS_DIR}/.." && pwd)"
PY="${PY:-python}"

echo "== repo: ${REPO_ROOT}"
"${PY}" - <<'EOF'
import sys
print("python:", sys.version.split()[0])
import sapien
print("sapien:", sapien.__version__, "(need 3.0.0b1)")
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import curobo
print("curobo: OK")
EOF

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/../RoboTwin}"
[[ -d "${ROBOTWIN_ROOT}/envs" ]] || { echo "RoboTwin not found at ${ROBOTWIN_ROOT} (set ROBOTWIN_ROOT)"; exit 1; }
echo "== RoboTwin: ${ROBOTWIN_ROOT}"
"${ENVS_DIR}/scripts/item_envs/install_robotwin_tasks.sh"
echo "== task symlinks: OK"

[[ -n "${HOLOBRAIN_CKPT:-}" && -f "${HOLOBRAIN_CKPT}/model.config.json" ]] || {
  echo "HOLOBRAIN_CKPT must point at the HoloBrain_v0.0_GD dir (needs model.config.json)"; exit 1; }
echo "== ckpt: ${HOLOBRAIN_CKPT}"
echo "ALL CHECKS PASSED"
