#!/usr/bin/env bash
# The safety filter's data from scratch, on the GPU node this runs on (Bridges-2 or any Linux GPU host). The same 29
# shard commands as Delta's 2026-10-23 collection (runs/lang_1023_data, data/lang_1023): ur5-wsg, the five tasks
# T1 T3 T5 T6 T7 (main/lang/config.py TRAIN_TASKS), same seeds:
#
#   warmup         15 shards, 30 episodes per task (150), no masks              the training data
#   val             5 shards,  5 episodes per task (25)                         validation during the rounds
#   val20           5 shards, 20 episodes per task of T3 / T5 / T6 (60)         grounding stream, g-base 4
#   val_grounding   4 shards, 20 episodes per task of T1 / T7 (40)              grounding stream
#
# then the text caches (HoloBrain BERT, with the held-out item words), check_shards, the collect reports and
# <ROOT>/READY. About 1 episode per minute per GPU at PAR=4 (~5 GPU-hours); ~52 GB on disk (sparse files).
#
#   bash scripts/lang/bridges/collect.sh [vanilla|qwen|all]     (normally through collect.sbatch)
#
#   vanilla  the shards above under ROOT (default data/lang)
#   qwen     the HoloBrain-Qwen replays of the same shards under QROOT (default data/lang_qwen): the Qwen runs'
#            data (another ~128 GB); needs ROOT/READY, the Qwen checkpoints and flash-attn (README section 1)
#   all      vanilla, then qwen
#
# Env: PAR (default 4) simulators per GPU (each ~8-14 GB; <= 4 on a 48 GB card next to nothing else). Every GPU
# this job sees gets its own queue worker; the workers share the queue state (atomic claims), so several jobs can
# also work on it at once. Resumable: rerun the same command after any stop; a finished shard is skipped, a
# partial one resumes after its last finished episode. Stops by itself before $STAGE_DEADLINE (collect.sbatch sets
# it). Exit 0 = READY, 75 = not finished (run again), else a failure (see the logs it names).
# Refuses a GPU without ray tracing cores (V100 / A100 / H100: the simulator would render with rasterization, so
# the images, and HoloBrain's tokens, would differ from Delta's); ALLOW_RASTER=1 overrides.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
WHAT=${1:-vanilla}
PAR=${PAR:-4}
abs() { realpath -m "$1"; }
ROOT=$(abs "${ROOT:-data/lang}"); QROOT=$(abs "${QROOT:-data/lang_qwen}")
STATE="${REPO_ROOT}/runs/lang_collect/$(basename "$ROOT")"
QSTATE="${REPO_ROOT}/runs/lang_collect/$(basename "$QROOT")"
case $WHAT in vanilla|qwen|all) ;; *) echo "usage: collect.sh [vanilla|qwen|all]"; exit 2 ;; esac

# the GPU must ray trace (main/envs/env.py picks the renderer by the GPU's name / compute capability)
if ! $PY -c "import sys; from main.envs.env import gpu_supports_ray_tracing as rt, _gpu_name as n; print('[collect] GPU:', n() or 'none'); sys.exit(0 if rt() else 3)"; then
  if [[ "${ALLOW_RASTER:-0}" != 1 ]]; then
    echo "[collect] this GPU has no ray tracing cores: refusing (ALLOW_RASTER=1 to collect with rasterization anyway)"
    exit 3
  fi
  echo "[collect] WARNING: rasterization (ALLOW_RASTER=1): these images will not match ray-traced data"
fi

vanilla_cmds() {   # the 29 commands, one per line (cwd = the repo root)
  local L="$ROOT/logs" s
  local P="mkdir -p $L && $PY -m main.lang.collect"
  for s in $(seq 0 14); do
    echo "$P warmup --out $ROOT/warmup/shard_g$s --embodiment ur5-wsg --shard $s --n-shards 15 --episodes-per-task 30 --no-masks >> $L/warmup_shard_g$s.log 2>&1"
  done
  for s in $(seq 0 4); do
    echo "$P validation --out $ROOT/val/shard_g$s --embodiment ur5-wsg --shard $s --n-shards 5 --episodes-per-task 5 >> $L/val_shard_g$s.log 2>&1"
  done
  for s in $(seq 0 4); do
    echo "$P validation --stream grounding --g-base 4 --out $ROOT/val20/shard_g$((s + 4)) --embodiment ur5-wsg --shard $s --n-shards 5 --episodes-per-task 20 --tasks push_zone,tray_transport,clean_pour >> $L/val20_shard_g$((s + 4)).log 2>&1"
  done
  for s in $(seq 0 3); do
    echo "$P validation --stream grounding --out $ROOT/val_grounding/shard_g$s --embodiment ur5-wsg --shard $s --n-shards 4 --episodes-per-task 20 --tasks selective_avoid,gap_place >> $L/val_grounding_shard_g$s.log 2>&1"
  done
}

qwen_cmds() {      # the Qwen replays of the same 29 shards (flash_attention_2: the Qwen runs' kernel)
  local L="$QROOT/logs" k d
  for k in warmup val val20 val_grounding; do
    for d in "$ROOT/$k"/shard_g*; do
      echo "mkdir -p $L && $PY -m main.lang.collect warmup --embodiment ur5-wsg --encoder holobrain_qwen --qwen-attn flash_attention_2 --replay $d --out $QROOT/$k/$(basename "$d") >> $L/${k}_$(basename "$d").log 2>&1"
    done
  done
}

run_queue() {      # cmds-file state-dir: one worker per visible GPU, PAR commands each -> 0 done / 75 paused / 1
  local cmds=$1 state=$2 gpus g pids=() rc=0 r
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
  else mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi
  [[ ${#gpus[@]} -gt 0 ]] || gpus=(0)
  echo "[collect] $(wc -l < "$cmds") commands, GPUs ${gpus[*]}, $PAR per GPU; queue $state (logs $state/logs)"
  for g in "${gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=$g $PY -m main.lang.workqueue --cmds "$cmds" --state "$state" --slots "$PAR" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p"; r=$?
    if [[ $r -eq 75 ]]; then [[ $rc -eq 0 ]] && rc=75; elif [[ $r -ne 0 ]]; then rc=1; fi
  done
  return $rc
}

finished() {       # cmds-file state-dir -> 0 when every command is done
  local n_cmd n_done
  n_cmd=$(grep -cv '^\s*\(#\|$\)' "$1"); n_done=$(ls "$2"/*.done 2>/dev/null | wc -l)
  echo "[collect] $n_done / $n_cmd commands done, $(ls "$2"/*.fail 2>/dev/null | wc -l) with failures"
  [[ $n_done -ge $n_cmd ]]
}

collect_vanilla() {
  [[ -f "$ROOT/READY" ]] && { echo "[collect] $ROOT is READY"; return 0; }
  mkdir -p "$STATE"
  vanilla_cmds > "$STATE/cmds"
  run_queue "$STATE/cmds" "$STATE"; local rc=$?
  if ! finished "$STATE/cmds" "$STATE"; then
    [[ $rc -eq 1 ]] && { echo "[collect] a command failed 3 times: see $STATE/logs and $ROOT/logs"; return 1; }
    return 75
  fi
  local k
  for k in warmup val val20 val_grounding; do
    [[ -s "$ROOT/$k/text_cache.pt" ]] && continue
    echo "[collect] text cache $k"
    $PY -m main.lang.text build --backend holobrain_bert --episodes "$ROOT/$k/shard_g*/item_buffer/episodes" \
        --out "$ROOT/$k/text_cache.pt" --with-heldout-item || return 1
  done
  $PY scripts/lang/check_shards.py "$ROOT/warmup" "$ROOT/val" "$ROOT/val20" "$ROOT/val_grounding" > "$STATE/check_shards.log" 2>&1 \
    || { echo "[collect] check_shards failed: $STATE/check_shards.log"; return 1; }
  for k in warmup val val20 val_grounding; do $PY -m main.lang.collect_report "$ROOT/$k" > /dev/null 2>&1 || true; done
  date > "$ROOT/READY"
  echo "[collect] $ROOT READY"
}

collect_qwen() {
  [[ -f "$QROOT/READY" ]] && { echo "[collect] $QROOT is READY"; return 0; }
  [[ -f "$ROOT/READY" ]] || { echo "[collect] $ROOT is not READY: collect the vanilla shards first"; return 1; }
  mkdir -p "$QSTATE"
  qwen_cmds > "$QSTATE/cmds"
  run_queue "$QSTATE/cmds" "$QSTATE"; local rc=$?
  if ! finished "$QSTATE/cmds" "$QSTATE"; then
    [[ $rc -eq 1 ]] && { echo "[collect] a command failed 3 times: see $QSTATE/logs and $QROOT/logs"; return 1; }
    return 75
  fi
  $PY scripts/lang/check_shards.py "$QROOT/warmup" "$QROOT/val" "$QROOT/val20" "$QROOT/val_grounding" > "$QSTATE/check_shards.log" 2>&1 \
    || { echo "[collect] check_shards failed: $QSTATE/check_shards.log"; return 1; }
  date > "$QROOT/READY"
  echo "[collect] $QROOT READY"
}

case $WHAT in
  vanilla) collect_vanilla ;;
  qwen) collect_qwen ;;
  all) collect_vanilla && collect_qwen ;;
esac
rc=$?
echo "[collect] exit $rc (0 = READY, 75 = not finished: run again)"
exit $rc
