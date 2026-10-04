#!/bin/bash
# Run groups of pytest invocations in parallel inside ONE GPU job (no
# pytest-xdist in the RoboTwin env). cwd must be <repo>/envs.
#
#   bash scripts/item_envs/par_pytest.sh <groups.txt> <log_dir> [parallel=6]
#
# groups.txt: one pytest argument string per line ('#' comments ok), e.g.
#   tests/item_envs/test_push_zone_sim.py -k ur5-wsg -m sim
# Each group's full output -> <log_dir>/<lineno>.log (never piped through tail:
# tracebacks must reach the log). Exit 0 iff every group passed; the summary
# line of each group is printed at the end.
set -u
GROUPS_FILE=${1:?groups file}
mkdir -p "${2:?log dir}"; LOGDIR=$(readlink -f "$2")
PAR=${3:-6}
PY=${PY:-/work/nvme/bifq/conda/envs/RoboTwin/bin/python}
export PYTHONUNBUFFERED=1
mkdir -p "$LOGDIR"
rm -f "$LOGDIR"/*.rc

run_group() {
  local n=$1; shift
  echo "[par-pytest] #$n start: $*"
  # shellcheck disable=SC2086
  $PY -m pytest "$@" -q --tb=short -p no:cacheprovider > "$LOGDIR/$n.log" 2>&1
  local rc=$?
  echo "$rc" > "$LOGDIR/$n.rc"
  echo "[par-pytest] #$n done rc=$rc: $(grep -E '^[0-9]+ (passed|failed)|passed|failed|error' "$LOGDIR/$n.log" | tail -1)"
}
export -f run_group
export LOGDIR PY

grep -v '^\s*#' "$GROUPS_FILE" | grep -v '^\s*$' | nl -ba -w1 -s' ' | \
  xargs -P "$PAR" -L 1 bash -c 'run_group "$@"' _

fail=0
for f in "$LOGDIR"/*.rc; do
  rc=$(cat "$f")
  # rc 5 = no tests collected for that group (e.g. a -k filter matched nothing)
  if [ "$rc" != "0" ] && [ "$rc" != "5" ]; then fail=1; fi
done
echo "[par-pytest] summary:"
for f in "$LOGDIR"/*.log; do
  echo "  $(basename "$f"): $(grep -E 'passed|failed|error' "$f" | tail -1)"
done
exit $fail
