#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4,gpuA40x4-preempt --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=08:00:00
#SBATCH --job-name=tray-eval -o /work/nvme/bifq/project/tmp/logs/tray_eval_%j.out
# Closed-loop test of the sliding-tray fine-tune (items/evaluate_tray.py) for NAMES (env, default "scripted careless"):
# "scripted" / "careless" / "hold" are agents, any other name is a model (exported from ws/<name>.DONE first when
# models/<name>/model is missing). PARTS (env, default 2) parts per name run SLOTS (env, default 2) at a time on the
# one A40 (the cameras need ray tracing). HELDOUT=1 (env) adds the held-out-template prompts; SCENES (env, default 20)
# scenes, drawn once into $RES/scenes/. Resubmitting resumes.
#   NAMES="scripted careless ft_tray" sbatch --export=ALL vla_finetuning/jobs/tray_eval.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
NAMES=${NAMES:-"scripted careless"}; PARTS=${PARTS:-2}; SLOTS=${SLOTS:-2}
RES=${RES:-$R/vla_finetuning/results_tray}
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 240 ))
mkdir -p $RES/_logs
scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_tray --out $RES/_scenes_only --make-scenes --scenes ${SCENES:-20} --scenes-dir $RES/scenes \
  > $RES/_logs/scenes.log 2>&1 || { echo "[tray-eval] scene list failed"; tail -5 $RES/_logs/scenes.log; exit 1; }
for n in $NAMES; do
  case $n in scripted|careless|hold) continue ;; esac
  if [ ! -d vla_finetuning/models/$n/model ]; then
    [ -f vla_finetuning/ws/$n.DONE ] || { echo "[tray-eval] $n: no model and no ws/$n.DONE"; exit 1; }
    ck=$(head -1 vla_finetuning/ws/$n.DONE)/model/model.safetensors
    echo "[tray-eval] export $n from $ck"
    scripts/lang/in_container.sh bash vla_finetuning/jobs/export.sh $n $ck || exit 1
  fi
done
one() {   # <name> <part>
  local n=$1 p=$2 agent args rc
  case $n in scripted|careless|hold) agent=$n; args="" ;;
             *) agent=model; args="--model-dir $R/vla_finetuning/models/$n/model" ;; esac
  [ "${HELDOUT:-0}" = 1 ] && args="$args --heldout"
  while true; do
    scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_tray --agent $agent $args --model-name $n \
      --out $RES/$n --parts $PARTS --part $p --scenes ${SCENES:-20} --scenes-dir $RES/scenes >> $RES/_logs/${n}_p$p.log 2>&1
    rc=$?
    [ $rc -eq 75 ] && [ $(( STAGE_DEADLINE - $(date +%s) )) -gt 600 ] && continue
    break
  done
  echo "[tray-eval] $(date '+%T') $n part $p rc=$rc"
}
for n in $NAMES; do
  for p in $(seq 0 $((PARTS - 1))); do
    while [ $(jobs -rp | wc -l) -ge $SLOTS ]; do sleep 20; done
    one $n $p & sleep 20
  done
done
wait
for n in $NAMES; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_tray --summary --out $RES/$n | tail -12
done
