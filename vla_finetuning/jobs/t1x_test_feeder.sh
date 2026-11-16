#!/bin/bash
# The T1 data test's closed-loop tests as 1-hour pieces on gpuA40x4-interactive (Yuxuan 2026-11-16: "let's finish the
# collision avoidance on a40 first"; the long A40 queue started nothing in two days). Each piece runs
# jobs/constraint_eval.sbatch for every model with a part left (scripted, the 100- and 300-scene GD / Qwen runs) and
# resumes where the last stopped; parts are writer-locked, so the long-queue jobs (vf-ceval, vf-t1x300-eval) may run
# at the same time. A refused sbatch (the user's one interactive job is taken) is retried after 2 minutes. Stops when
# every (model, part) has its DONE marker, or after MAX_PIECES pieces. t1x_feeder.sh writes the summaries.
#   ssh dt-login01 'cd /work/nvme/bifq/project/LanguageConditionHJ && setsid nohup bash vla_finetuning/jobs/t1x_test_feeder.sh \
#     > /dev/null 2>> vla_finetuning/results_constraint_t1/_logs/test_feeder.err < /dev/null &'
R=/work/nvme/bifq/project/LanguageConditionHJ
RES=$R/vla_finetuning/results_constraint_t1; LOG=$RES/_logs/test_feeder.log
T=selective_avoid; NPARTS=5; NSCENES=100; SLOTS=${SLOTS:-4}; MAX_PIECES=${MAX_PIECES:-14}
ALL="scripted qwen_constraint_t1 ft_constraint_t1 ft_constraint_t1_300 qwen_constraint_t1_300"
cd $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
left_names() { local n k out=""; for n in $ALL; do for k in $(seq 0 $((NPARTS - 1))); do
  [ -f $RES/$n/DONE_$T.p$k ] || { out="$out $n"; break; }; done; done; echo $out; }
log "test feeder start (pid $$ on $(hostname -s))"
piece=0
while [ $piece -lt $MAX_PIECES ]; do
  names=$(left_names)
  [ -z "$names" ] && { log "all parts DONE"; exit 0; }
  out=$(NAMES="$names" RES=$RES EVAL_TASKS=$T SCENES=$NSCENES PARTS=$NPARTS SLOTS=$SLOTS sbatch --parsable \
        --partition=gpuA40x4-interactive --time=01:00:00 --mem=96g --nodes=1 --export=ALL \
        --job-name=vf-ceval-int vla_finetuning/jobs/constraint_eval.sbatch 2>&1)
  if ! [[ "$out" =~ ^[0-9]+$ ]]; then sleep 120; continue; fi
  piece=$((piece + 1)); log "piece $piece job $out: $names"
  while squeue -h -j $out 2>/dev/null | grep -q .; do sleep 60; done
  log "piece $piece job $out ended: $(sacct -j $out -X -n -o State,Elapsed | tr -s ' ')"
done
log "stopped after $MAX_PIECES pieces"
