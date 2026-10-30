#!/bin/bash
# Check 4 (diag 2026-10-30): the body of every critic-retrain job (sbatch files under runs/lang/diag_1030/retrain/jobs/).
# Runs on the batch node, outside the container:
#   1. the probe set (built once; a job that finds another building it waits),
#   2. the smoke (5 variants x 150 steps, 100 then a resume to 150) unless retrain_smoke/PASSED exists,
#   3. the production pool (5 variants x 20,000 steps, resumable) until the job's end.
#   SLOTS=<workers per GPU> PHASES=smoke,prod bash scripts/lang/diag_1030/retrain_job.sh
set -u
REPO=/work/nvme/bifq/project/LanguageConditionHJ
cd $REPO
S=$REPO/scripts/lang/in_container.sh
D=scripts/lang/diag_1030
OUT=runs/lang/diag_1030/retrain
SM=runs/lang/diag_1030/retrain_smoke
VARS=${VARS:-det,stoch,stoch_mlp:5000,zero,data,stoch_norm}
SLOTS=${SLOTS:-1}
PHASES=${PHASES:-smoke,prod}
PROD_STEPS=${PROD_STEPS:-20000}
mkdir -p $OUT $SM
left_s() { awk -v t="$(squeue -h -j "$SLURM_JOB_ID" -o %L)" 'BEGIN{n=split(t,a,":"); s=0; for(i=1;i<=n;i++) s=s*60+a[i]; print s}'; }
END=$(( $(date +%s) + $(left_s) ))
echo "[job $SLURM_JOB_ID] $SLURM_JOB_PARTITION nodes $SLURM_JOB_NODELIST ($SLURM_NNODES), slots/GPU $SLOTS," \
     "phases $PHASES, ends $(date -d @$END '+%T')"
P1="srun --nodes=1 --ntasks=1 --kill-on-bad-exit=0"
PN="srun --ntasks-per-node=1 --kill-on-bad-exit=0"

# 1. probe set
if [ ! -f $OUT/probe_index.json ]; then
  if mkdir $OUT/probe.lock 2> /dev/null; then
    echo "$SLURM_JOB_ID" > $OUT/probe.lock/owner
    $P1 $S python $D/critic_retrain.py build-probe
    echo "[job] build-probe rc=$? $(date '+%T')"
    rm -rf $OUT/probe.lock
  else
    echo "[job] another job builds the probe set ($(cat $OUT/probe.lock/owner 2>/dev/null)); waiting"
    for i in $(seq 1 60); do [ -f $OUT/probe_index.json ] && break; sleep 15; done
  fi
fi
[ -f $OUT/probe_index.json ] || { echo "[job] no probe set: stop"; exit 1; }

# 2. smoke
smoke_ok() {
  local v
  for v in ${VARS//,/ }; do
    [ -f $SM/$v/checkpoint_final.pt ] || return 1
    grep -q '"step": 150,' $SM/$v/diag.jsonl 2> /dev/null || return 1
  done
}
if [[ $PHASES == *smoke* ]] && [ ! -f $SM/PASSED ]; then
  for ST in 100 150; do
    $PN bash $D/retrain_pool.sh $VARS $ST $SM $SLOTS $END 1200 --probe-every 50 --state-every 50 --ckpt-every 0
    echo "[job] smoke to $ST done $(date '+%T')"
  done
  for i in $(seq 1 40); do smoke_ok && break; sleep 15; done   # another job may hold some smoke variants
  if smoke_ok; then date > $SM/PASSED; echo "[job] smoke PASSED"; else echo "[job] smoke NOT passed: stop"; exit 1; fi
fi

# 3. production (+ in the background on node 0, sharing its GPU: score new interim checkpoints on check 2's T5 states)
if [[ $PHASES == *prod* ]]; then
  [ -f $SM/PASSED ] || { echo "[job] smoke not passed: no production"; exit 1; }
  if [ "${SCORE:-1}" = 1 ]; then
    ( sleep 120; timeout $(( END - $(date +%s) - 300 )) srun --nodes=1 --ntasks=1 --overlap --kill-on-bad-exit=0 \
        env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $S bash $D/score_new.sh \
        >> $OUT/scoring_logs/score_new_${SLURM_JOB_ID}.log 2>&1 ) &
  fi
  $PN bash $D/retrain_pool.sh $VARS $PROD_STEPS $OUT $SLOTS $END 0 --probe-every 250 --state-every 250 --ckpt-every 2500
  echo "[job] production pool done $(date '+%T')"
  wait
  $S python $D/retrain_table.py --quiet && echo "[job] results.json refreshed"
fi
