#!/bin/bash
# Run a list of `probe episode` invocations in parallel inside ONE GPU job
# (physics is CPU-bound PhysX; cuRobo shares the GPU — an A40 holds ~8
# concurrent planners comfortably). One process per episode (pybind11 TLS
# limit, AGENTS.md). cwd must be <repo>/envs.
#
#   bash scripts/item_envs/par_episodes.sh <jobs.txt> <out_root> [parallel=4]
#
# jobs.txt: one `probe episode` argument string per line ('#' comments ok),
# e.g.  --task push_zone --embodiment ur5-wsg --seed 0 --risk time_scale=0.5 --video
# Each line's stdout/stderr -> <out_root>/logs/<lineno>.log; the JSON summary
# printed by `probe episode` is collected into <out_root>/summary.jsonl.
set -u
JOBS=${1:?jobs file}
OUT=$(readlink -f "${2:?out root}")
PAR=${3:-4}
PY=${PY:-/work/nvme/bifq/conda/envs/RoboTwin/bin/python}
export PYTHONUNBUFFERED=1  # stream episode logs live (block-buffered when redirected)
mkdir -p "$OUT/logs"
: > "$OUT/summary.jsonl"

run_one() {
  local n=$1; shift
  local log="$OUT/logs/$n.log"
  echo "[par] #$n start: $*"
  # shellcheck disable=SC2086
  $PY -m main.item_envs.probe episode "$@" --out "$OUT" > "$log" 2>&1
  local rc=$?
  # the episode summary is the trailing JSON object of the log
  $PY - "$log" "$n" "$rc" "$*" >> "$OUT/summary.jsonl" <<'EOF'
import json, sys
log, n, rc, argv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
text = open(log, errors="replace").read()
i = text.rfind("\n{")
try:
    summ = json.loads(text[i + 1:]) if i >= 0 else {}
except Exception:
    summ = {}
print(json.dumps({"n": n, "rc": rc, "args": argv, **summ}))
EOF
  echo "[par] #$n done rc=$rc"
}
export -f run_one
export OUT PY

grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | nl -ba -w1 -s' ' | \
  xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "[par] all done: $(wc -l < "$OUT/summary.jsonl") summaries in $OUT/summary.jsonl"
