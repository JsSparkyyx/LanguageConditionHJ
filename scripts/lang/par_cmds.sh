#!/bin/bash
# Run the lines of a command file in parallel INSIDE one job (each line is a
# full shell command, quoting kept: read with IFS= read -r, run with bash -c).
#
#   bash scripts/lang/par_cmds.sh <cmds.txt> <log_dir> [parallel=4]
#
# Each line -> <log_dir>/<lineno>.log and <lineno>.rc. Exit 0 iff all lines
# exit 0. cwd = the caller's (the repo root inside in_container.sh).
set -u
CMDS=${1:?cmds file}; LOGDIR=${2:?log dir}; PAR=${3:-4}
mkdir -p "$LOGDIR"
rm -f "$LOGDIR"/*.rc
n=0; pids=()
run_line() { local i=$1; shift; bash -c "$1" > "$LOGDIR/$i.log" 2>&1; echo $? > "$LOGDIR/$i.rc"; }
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
  echo "[par] #$n start: $line"
  run_line "$n" "$line" &
done < "$CMDS"
wait
# 0 iff every line exited 0; 75 iff every failing line exited 75 (paused, resumable); else 1
fail=0
for f in "$LOGDIR"/*.rc; do
  rc=$(cat "$f"); [ "$rc" = 0 ] && continue
  echo "[par] $(basename "$f" .rc) rc=$rc"
  if [ "$rc" = 75 ]; then [ "$fail" = 1 ] || fail=75; else fail=1; fi
done
exit $fail
