#!/bin/bash
# Check 4 (diag 2026-10-30): run critic-retrain variants on THIS node, inside the container (cwd = repo root).
#   bash scripts/lang/diag_1030/retrain_node.sh <variants a,b,..> <budget s per lane> <grad steps> <run root> [extra args]
# Variant i goes to node (i mod SLURM_NNODES); on a node, variant j goes to GPU lane (j mod n_gpus), lanes run in
# parallel, the variants of one lane one after the other (each with budget / its lane's length). A run dir with a
# state.pt is resumed. One log per variant and job: <run root>/<variant>/job_<jobid>.log
set -u
VARS=${1:?variants}; BUDGET=${2:?budget}; STEPS=${3:?steps}; ROOT=${4:?run root}; shift 4
EXTRA="$*"
IFS=, read -ra V <<< "$VARS"
NN=${SLURM_NNODES:-1}; NID=${SLURM_NODEID:-0}
mine=()
for i in "${!V[@]}"; do [ $((i % NN)) -eq "$NID" ] && mine+=("${V[$i]}"); done
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"; else
  G=(); for k in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do G+=("$k"); done; fi
NG=${#G[@]}
echo "[node $NID $(hostname)] variants ${mine[*]:-none} on GPUs ${G[*]} ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1))"
for g in $(seq 0 $((NG - 1))); do
  lane=()
  for j in "${!mine[@]}"; do [ $((j % NG)) -eq "$g" ] && lane+=("${mine[$j]}"); done
  [ ${#lane[@]} -eq 0 ] && continue
  (
    b=$((BUDGET / ${#lane[@]}))
    for v in "${lane[@]}"; do
      R=$ROOT/$v; mkdir -p "$R"
      res=""; [ -f "$R/state.pt" ] && res="--resume"
      CUDA_VISIBLE_DEVICES=${G[$g]} python scripts/lang/diag_1030/critic_retrain.py train --target "$v" --run "$R" \
        --grad-steps "$STEPS" --time-budget "$b" $res $EXTRA > "$R/job_${SLURM_JOB_ID:-local}.log" 2>&1
      echo "[node $NID gpu ${G[$g]}] $v rc=$? $(date '+%T') last: $(grep -E '^\[(retrain|lang-train)\]' "$R/job_${SLURM_JOB_ID:-local}.log" | tail -1)"
    done
  ) &
done
wait
