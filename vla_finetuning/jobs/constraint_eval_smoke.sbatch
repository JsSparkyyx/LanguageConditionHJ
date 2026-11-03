#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=00:40:00
#SBATCH --job-name=vf-cesmoke -o /work/nvme/bifq/project/tmp/logs/vf_cesmoke_%j.out
# Smoke test of evaluate_constraint.py: 2 scenes of T5 and T7 (own scenes dir), the scripted agent on both, the
# 10-27 model on T5's scenes (5 prompts each).
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; S=$R/vla_finetuning/results_constraint/_smoke_eval
cd $R
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 120 ))
mkdir -p $S/_logs
for t in tray_transport gap_place; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_constraint --make-scenes --tasks $t --child \
    --scenes 2 --scenes-dir $S/scenes --out $S/scenes_only >> $S/_logs/scenes_$t.log 2>&1 &
done
wait
for t in tray_transport gap_place; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_constraint --agent scripted --model-name scripted \
    --tasks $t --child --scenes 2 --scenes-dir $S/scenes --out $S/scripted >> $S/_logs/scripted_$t.log 2>&1 &
  sleep 15
done
scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_constraint --agent model \
  --model-dir $R/vla_finetuning/models/ft_items_safe_x2/model --model-name ft_items_safe_x2 --tasks tray_transport \
  --child --scenes 2 --scenes-dir $S/scenes --out $S/ft_items_safe_x2 >> $S/_logs/model_tray_transport.log 2>&1 &
wait
for n in scripted ft_items_safe_x2; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_constraint --summary --out $S/$n --tasks tray_transport gap_place
done
