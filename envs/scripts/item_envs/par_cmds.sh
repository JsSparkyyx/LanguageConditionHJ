#!/bin/bash
# Run arbitrary command lines in parallel inside ONE GPU job (e.g. sharded
# `probe sweep` calibration slices). cwd must be <repo>/envs.
#
#   bash scripts/item_envs/par_cmds.sh <cmds.txt> <log_dir> [parallel=12]
#
# cmds.txt: one shell command per line ('#' comments ok); "$PY" expands to the
# RoboTwin interpreter. Each line's output -> <log_dir>/<lineno>.log. Exit 0
# iff every line exited 0. Sweeps resume per episode (existing meta.json is
# skipped), so a 1-h slice killed mid-way is simply resubmitted.
set -u
CMDS=${1:?cmds file}
LOGDIR=$(readlink -f "${2:?log dir}")
PAR=${3:-12}
export PY=${PY:-/work/nvme/bifq/conda/envs/RoboTwin/bin/python}
export PYTHONUNBUFFERED=1
mkdir -p "$LOGDIR"
rm -f "$LOGDIR"/*.rc

run_line() {
  local n=$1; shift
  echo "[par-cmds] #$n start: $*"
  bash -c "$*" > "$LOGDIR/$n.log" 2>&1
  local rc=$?
  echo "$rc" > "$LOGDIR/$n.rc"
  echo "[par-cmds] #$n done rc=$rc"
}
export -f run_line
export LOGDIR

grep -v '^\s*#' "$CMDS" | grep -v '^\s*$' | nl -ba -w1 -s' ' | \
  xargs -P "$PAR" -L 1 bash -c 'run_line "$@"' _

fail=0
for f in "$LOGDIR"/*.rc; do [ "$(cat "$f")" = "0" ] || fail=1; done
exit $fail
