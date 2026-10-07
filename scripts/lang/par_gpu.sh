#!/bin/bash
# Run the lines of a command file in parallel over the job's GPUs with
# CUDA_VISIBLE_DEVICES pinned. A line "@<g> <cmd>" runs on GPU g; other lines
# go round-robin. At most PER_GPU lines run at once on each GPU
# (scripts/lang/par_cmds.sh per GPU).
#   bash scripts/lang/par_gpu.sh <cmds.txt> <log_dir> <n_gpus> <per_gpu>
set -u
CMDS=${1:?cmds}; LOGDIR=${2:?log dir}; NG=${3:?n gpus}; PER=${4:?per gpu}
mkdir -p "$LOGDIR"
HERE=$(dirname "$(readlink -f "$0")")
for g in $(seq 0 $((NG - 1))); do
  grep -v '^\s*#' "$CMDS" | grep -v '^\s*$' | awk -v g=$g -v n=$NG '
    /^@[0-9]+ / { split($1, a, "@"); if (a[2] % n == g) { sub(/^@[0-9]+ /, ""); print } next }
    { if (rr % n == g) print; rr++ }' > "$LOGDIR/gpu$g.cmds"
  CUDA_VISIBLE_DEVICES=$g bash "$HERE/par_cmds.sh" "$LOGDIR/gpu$g.cmds" "$LOGDIR/gpu$g" "$PER" > "$LOGDIR/gpu$g.par.log" 2>&1 &
done
wait
fail=0
for g in $(seq 0 $((NG - 1))); do
  for f in "$LOGDIR"/gpu$g/*.rc; do [ -f "$f" ] || continue; rc=$(cat "$f")
    [ "$rc" = 0 ] && continue
    echo "[par_gpu] gpu$g/$(basename "$f" .rc) rc=$rc"
    if [ "$rc" = 75 ]; then [ "$fail" = 1 ] || fail=75; else fail=1; fi
  done
done
exit $fail
