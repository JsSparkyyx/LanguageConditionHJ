#!/bin/bash
# One node's workers of eval_nominal.sbatch (inside the container): SLOTS loops on this node's GPU, each claiming the
# next (mode, task) of the 10 by mkdir in a claim dir of this job and running evaluate_nominal's child on it until it
# finishes (0) or reaches the stage deadline (75). The writer lock per (mode, task) output keeps two jobs apart.
#   bash vla_finetuning/jobs/eval_nominal_pool.sh <claim dir> <slots>
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; RES=$R/vla_finetuning/results_items
CL=$1; SLOTS=${2:-3}
cd $R
# T1 first: the safe T1 expert is the slowest (grasp search) and the heaviest (~14.5 GB of GPU memory)
ITEMS="safe:selective_avoid careless:selective_avoid safe:push_zone careless:push_zone safe:clean_pour \
careless:clean_pour safe:gap_place careless:gap_place safe:tray_transport careless:tray_transport"
slot() {
  local it mode task rc
  for it in $ITEMS; do
    mkdir "$CL/$it" 2> /dev/null || continue
    mode=${it%%:*}; task=${it#*:}
    echo "[nominal $(hostname -s) s$1] $(date '+%T') $mode $task"
    python -m vla_finetuning.items.evaluate_nominal --mode $mode --out $RES/nominal_$mode --ref $RES/base \
      --tasks $task --child > $RES/nominal_$mode/log_${task}_${SLURM_JOB_ID}.txt 2>&1
    rc=$?
    echo "[nominal $(hostname -s) s$1] $(date '+%T') $mode $task rc=$rc: $(grep -c '^\[eval_nominal\].*ep[0-9]' \
      $RES/nominal_$mode/log_${task}_${SLURM_JOB_ID}.txt) episodes this job"
  done
}
mkdir -p $RES/nominal_safe $RES/nominal_careless
for s in $(seq 1 $SLOTS); do slot $s & sleep 20; done
wait
