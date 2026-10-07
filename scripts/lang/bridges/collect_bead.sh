#!/usr/bin/env bash
# Collect the bead-task shards (T2 / T4 / T6 under the bead-count h: 30 warmup shards of 6 episodes and 10
# validation shards of 3, 210 episodes) on ONE GPU, PAR shards at a time (<= 7 simulators per 48 GB GPU).
#
#   HOLOBRAIN_CKPT=<dir> PY=<python> bash scripts/lang/bridges/collect_bead.sh [PAR=4]
#
# Same commands and seeds as the Delta queue (runs/lang_pipeline_bead/work.cmds), so shard folders copied
# from Delta are reused: a complete shard is skipped, a partial one resumes after its last finished episode.
# Resumable: after a stop (job end, crash), run the same command again. Several GPUs: run one copy per GPU
# with distinct CUDA_VISIBLE_DEVICES; they share the queue state (atomic claims) and never run a shard twice.
# Do NOT also run the Delta queue on the same folders (stop it, or copy only finished shards).
# About 1 episode per minute per GPU at PAR=4: ~3.5 GPU-hours in all.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
PAR=${1:-4}
Q="${REPO_ROOT}/runs/lang_bead_bridges"
mkdir -p "$Q"
T=granular_pour,granular_carry,clean_pour
{
  for e in ur5-wsg franka-panda; do for s in $(seq 0 14); do
    g=$([ $e = ur5-wsg ] && echo $s || echo $((s + 15)))
    echo "$PY -m main.lang.collect warmup --out ${REPO_ROOT}/data/lang/warmup/shard_g${g}_bead --embodiment $e --shard $s --n-shards 15 --episodes-per-task 30 --tasks $T --no-masks"
  done; done
  for e in ur5-wsg franka-panda; do for s in $(seq 0 4); do
    g=$([ $e = ur5-wsg ] && echo $s || echo $((s + 5)))
    echo "$PY -m main.lang.collect validation --out ${REPO_ROOT}/data/lang/val/shard_g${g}_bead --embodiment $e --shard $s --n-shards 5 --episodes-per-task 5 --tasks $T"
  done; done
} > "$Q/work.cmds"
echo "[collect_bead] $(wc -l < "$Q/work.cmds") shard commands, $PAR at a time; logs in $Q/work_state/logs"
$PY -m main.lang.workqueue --cmds "$Q/work.cmds" --state "$Q/work_state" --slots "$PAR"
rc=$?
echo "[collect_bead] workqueue exit $rc (0 = every shard done; 75 = stopped early, rerun to continue)"
exit $rc
