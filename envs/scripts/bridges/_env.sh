#!/usr/bin/env bash
# Shared preamble for Bridges runner scripts: env knobs + caches + ckpt.
# Sourced by run_*.sh — expects ENVS_DIR and REPO_ROOT to be set by the caller.

PY="${PY:-python}"
export HOLOBRAIN_CKPT="${HOLOBRAIN_CKPT:?set HOLOBRAIN_CKPT to the HoloBrain_v0.0_GD dir}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/../RoboTwin}"

# cuRobo NVRTC needs a writable TMPDIR; caches stay inside the repo tree.
BRIDGES_TMP="${BRIDGES_TMP:-${REPO_ROOT}/envs/runs/bridges_tmp}"
mkdir -p "${BRIDGES_TMP}"
export TMP="${BRIDGES_TMP}" TMPDIR="${BRIDGES_TMP}"
export HF_HOME="${HF_HOME:-${BRIDGES_TMP}/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BRIDGES_TMP}/xdg-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BRIDGES_TMP}/triton-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${BRIDGES_TMP}/torch-ext}"
export WANDB_MODE=offline

LOG_DIR="${REPO_ROOT}/envs/runs/bridges_logs"
mkdir -p "${LOG_DIR}"
