#!/usr/bin/env bash
# install_robotwin_tasks.sh — symlink this repo's robotwin_tasks/*.py into
# RoboTwin/envs/ (PLAN v3 §2). Idempotent: ln -sfn replaces existing links;
# absolute targets; every link is verified to resolve. NEVER modifies any
# existing RoboTwin file — it only ADDS links whose names come from our
# robotwin_tasks/ directory, and refuses to overwrite a real file.
#
# Usage: scripts/item_envs/install_robotwin_tasks.sh [--check-import]
#   ROBOTWIN_ROOT  default: <repo>/../RoboTwin
#   --check-import  also import envs.<task> from the RoboTwin root afterwards
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${ROBOTWIN_ROOT:-}" ]]; then
    # RoboTwin is a sibling of the LanguageConditionHJ checkout (../../RoboTwin
    # from envs/); ../RoboTwin covers a flat local layout.
    for cand in "${REPO_ROOT}/../../RoboTwin" "${REPO_ROOT}/../RoboTwin"; do
        if [[ -d "${cand}/envs" ]]; then
            ROBOTWIN_ROOT="$(cd "${cand}" && pwd)"
            break
        fi
    done
fi
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:?RoboTwin checkout not found; set ROBOTWIN_ROOT}"
TASK_SRC="${REPO_ROOT}/robotwin_tasks"
TASK_DST="${ROBOTWIN_ROOT}/envs"
CHECK_IMPORT=0
[[ "${1:-}" == "--check-import" ]] && CHECK_IMPORT=1

[[ -d "${TASK_SRC}" ]] || { echo "no robotwin_tasks dir at ${TASK_SRC}" >&2; exit 1; }
[[ -d "${TASK_DST}" ]] || { echo "no RoboTwin envs dir at ${TASK_DST}" >&2; exit 1; }

shopt -s nullglob
installed=()
for src in "${TASK_SRC}"/*.py; do
    name="$(basename "${src}")"
    dst="${TASK_DST}/${name}"
    if [[ -e "${dst}" && ! -L "${dst}" ]]; then
        echo "REFUSING to overwrite real RoboTwin file: ${dst}" >&2
        exit 1
    fi
    ln -sfn "${src}" "${dst}"
    if [[ ! -e "${dst}" ]]; then
        echo "broken symlink after install: ${dst}" >&2
        exit 1
    fi
    installed+=("${name}")
    echo "linked ${dst} -> ${src}"
done
echo "installed ${#installed[@]} task file(s) into ${TASK_DST}"

# asset variants (robotwin_assets/<model>/, e.g. 002_bowl_shallow — new
# model folders whose json rescales an existing mesh): linked into
# RoboTwin/assets/objects/, never over a real folder
ASSET_SRC="${REPO_ROOT}/robotwin_assets"
ASSET_DST="${ROBOTWIN_ROOT}/assets/objects"
if [[ -d "${ASSET_SRC}" ]]; then
    for src in "${ASSET_SRC}"/*/; do
        src="${src%/}"
        name="$(basename "${src}")"
        dst="${ASSET_DST}/${name}"
        if [[ -e "${dst}" && ! -L "${dst}" ]]; then
            echo "REFUSING to overwrite real RoboTwin asset folder: ${dst}" >&2
            exit 1
        fi
        ln -sfn "${src}" "${dst}"
        [[ -e "${dst}" ]] || { echo "broken asset symlink: ${dst}" >&2; exit 1; }
        echo "linked ${dst} -> ${src}"
    done
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
# login-node-safe syntax check (importing envs.<task> executes RoboTwin's
# robot/planner chain, which needs cuRobo — only --check-import does that)
for name in "${installed[@]}"; do
    "${PYTHON_BIN}" -m py_compile "${TASK_SRC}/${name}" \
        && echo "py_compile OK: ${name}"
done

if [[ "${CHECK_IMPORT}" == "1" ]]; then
    echo "note: import check needs the full RoboTwin stack (cuRobo / GPU node)"
    for name in "${installed[@]}"; do
        [[ "${name}" == _*.py ]] && continue  # helper modules import via their tasks
        mod="envs.${name%.py}"
        (cd "${ROBOTWIN_ROOT}" && "${PYTHON_BIN}" -c "import importlib; importlib.import_module('${mod}'); print('import OK: ${mod}')")
    done
fi
