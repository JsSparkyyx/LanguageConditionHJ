#!/usr/bin/env bash
# Shared setup for the language-filter scripts on a machine without Delta's container
# (Bridges-2 or any Linux GPU host). Sourced by the other scripts in this directory.
#
# Required:  HOLOBRAIN_CKPT  the HoloBrain_v0.0_GD directory (holds model.config.json)
# Optional:  PY              python of the RoboTwin env (default: python); may be a command,
#                            e.g. PY="apptainer exec --nv robot-sim.sif python"
#            LANG_TMP        scratch for caches (default: <repo>/runs/bridges_tmp)
#            HF_HOME         Hugging Face cache; needs bert-base-uncased (tokenizer + config) when the
#                            compute nodes have no internet (copy Delta's hf-cache/hub/models--bert-base-uncased)
#            WANDB_API_KEY   set it to log online (project cehj-lang); otherwise wandb writes offline files
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export REPO_ROOT
PY="${PY:-python}"
export HOLOBRAIN_CKPT="${HOLOBRAIN_CKPT:?set HOLOBRAIN_CKPT to the HoloBrain_v0.0_GD directory}"
LANG_TMP="${LANG_TMP:-${REPO_ROOT}/runs/bridges_tmp}"
mkdir -p "${LANG_TMP}/tmp"
# cuRobo's NVRTC needs a writable TMPDIR (else it silently falls back to the mplib planner)
export TMP="${LANG_TMP}" TMPDIR="${LANG_TMP}/tmp"
export HF_HOME="${HF_HOME:-${LANG_TMP}/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LANG_TMP}/xdg-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LANG_TMP}/triton-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LANG_TMP}/torch-ext}"
export PYTHONUNBUFFERED=1
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi
# simulator crashes write multi-GB core files into the RoboTwin directory (the env chdirs there)
ulimit -c 0
LOG_DIR="${REPO_ROOT}/runs/bridges_logs"
mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
