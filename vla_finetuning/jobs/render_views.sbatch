#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-preempt --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=8 --mem=40g --time=00:30:00
#SBATCH --job-name=vf-render -o /work/nvme/bifq/project/tmp/logs/vf_render_%j.out
# The three camera views of one scene per item task (items/render_views.py), 320x240 and 640x480, into imgs/.
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
cd $R
for c in D435 Large_D435; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.render_views --out $R/imgs --cam $c
done
rm -rf $R/imgs/_runner
