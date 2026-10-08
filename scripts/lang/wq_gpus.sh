#!/bin/bash
# N workqueue workers on one node, worker k pinned to GPU k (main.lang.workqueue does not pin GPUs itself).
#   bash scripts/lang/wq_gpus.sh <n_gpus> <cmds> <state>        (inside the container, cwd = the code checkout)
# Exit 0 when every worker exits 0, 75 when any paused (and none failed otherwise), else 1.
set -u
NG=${1:?n gpus}; CMDS=${2:?cmds}; STATE=${3:?state}
pids=()
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g python -m main.lang.workqueue --cmds "$CMDS" --state "$STATE" --slots 1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do
  wait "$p"; r=$?
  if [ "$r" -eq 75 ]; then [ "$rc" -eq 0 ] && rc=75; elif [ "$r" -ne 0 ]; then rc=1; fi
done
exit $rc
