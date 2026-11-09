#!/usr/bin/env bash
# Start one of the 2026-10-23 round trainings on Bridges-2 (or any Slurm cluster), with Delta's settings
# (runs/lang/rounds_1023/launch.sh): ur5-wsg, gamma 0.95, softmin critic readout, MLP actor, 100 rounds x 4 episodes
# x 2048 steps, probes every 10 rounds. The job is rounds.sbatch: it resubmits itself until the run is done.
#
#   cd <repo>
#   export HOLOBRAIN_CKPT=<dir> PY=<python> SBATCH_FLAGS="-A <account> -p GPU-shared --gpus=<rtx type>:1"
#   bash scripts/lang/bridges/launch.sh vanilla|t1|qwen|t1qwen
#
#   vanilla   the five tasks (T1 T3 T5 T6 T7)        runs/lang/rounds_ur5_g095_<TAG>
#   t1        T1 (selective_avoid) only               runs/lang/rounds_t1_ur5_g095_<TAG>
#   qwen      the five tasks, HoloBrain-Qwen front end runs/lang/rounds_ur5_qwen_g095_<TAG>
#   t1qwen    T1 only, HoloBrain-Qwen                  runs/lang/rounds_t1_ur5_qwen_g095_<TAG>
#
# Paths (any location; relative ones are taken from the repo root and passed on as absolute paths):
#   DATA_ROOT   the vanilla data, with warmup/ val/ val_grounding/ (default data/lang)
#   QDATA_ROOT  the Qwen replays of it, same layout (default data/lang_qwen; qwen and t1qwen only)
#   RUNS_DIR    where the run directory goes (default runs/lang);  TAG  the run-name suffix (default MMDD of today)
# WANDB_API_KEY set = online wandb (project cehj-lang), else offline files in the run.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
WHICH=${1:?vanilla, t1, qwen or t1qwen}
: "${SBATCH_FLAGS:?set SBATCH_FLAGS, e.g. \"-A <account> -p GPU-shared --gpus=<rtx type>:1\"}"
D=$(realpath -m "${DATA_ROOT:-data/lang}"); Q=$(realpath -m "${QDATA_ROOT:-data/lang_qwen}")
RUNS=$(realpath -m "${RUNS_DIR:-runs/lang}"); TAG=${TAG:-$(date +%m%d)}
SET_GD='gamma=0.95 gamma_final=0.95 critic_readout="softmin"'
SET_Q='gamma=0.95 gamma_final=0.95 encoder="holobrain_qwen" text_backend="holobrain_qwen" critic_readout="softmin" qwen_attn="flash_attention_2"'

need() {   # data root, splits... : every split has shards (and, for the vanilla data, its text cache)
  local root=$1 k; shift
  for k in "$@"; do
    ls -d "$root/$k"/shard_g* > /dev/null 2>&1 || { echo "[launch] no shards in $root/$k"; exit 3; }
    if [[ "$root" == "$D" && ! -s "$root/$k/text_cache.pt" ]]; then echo "[launch] $root/$k/text_cache.pt missing"; exit 3; fi
  done
  [[ -f "$root/READY" ]] || echo "[launch] note: $root has no READY file (copied data?): run check.sh on it first"
}
submit() {   # name, data root, extra args, set
  local R="$RUNS/$1"
  [[ ! -e "$R" ]] || { echo "[launch] $R exists (a resubmission of it: sbatch rounds.sbatch with the same RUN and variables)"; exit 2; }
  echo "[launch] $R on $2"
  env -u CHAIN_PIPE -u CHAIN_STAGE -u GAMMA -u LANG_REPO RUN="$R" DATA="$2/warmup" VAL="$2/val" EXTRA_ARGS="$3" SET="$4" \
    sbatch --export=ALL $SBATCH_FLAGS -J "lang-$WHICH" scripts/lang/bridges/rounds.sbatch
}
case $WHICH in
  vanilla) need "$D" warmup val
           submit "rounds_ur5_g095_$TAG" "$D" "--embodiments ur5-wsg" "$SET_GD" ;;
  t1)      need "$D" warmup val val_grounding
           submit "rounds_t1_ur5_g095_$TAG" "$D" "--tasks selective_avoid --embodiments ur5-wsg --val $D/val $D/val_grounding" "$SET_GD" ;;
  qwen)    need "$Q" warmup val
           submit "rounds_ur5_qwen_g095_$TAG" "$Q" "--embodiments ur5-wsg" "$SET_Q" ;;
  t1qwen)  need "$Q" warmup val val_grounding
           submit "rounds_t1_ur5_qwen_g095_$TAG" "$Q" "--tasks selective_avoid --embodiments ur5-wsg --val $Q/val $Q/val_grounding" "$SET_Q" ;;
  *) echo "unknown $WHICH (vanilla, t1, qwen or t1qwen)"; exit 2 ;;
esac
