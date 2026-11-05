#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=8 --mem=40g --time=00:45:00
#SBATCH --job-name=vf-render -o /work/nvme/bifq/project/tmp/logs/vf_render_%j.out
# The agent view (head camera) of ur5-wsg and franka-panda in the same T1 scene, every joint at 0 (Yuxuan 11-05;
# items/render_views.py --zero-qpos), 320x240 and 640x480, into imgs/zero_qpos/.
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
cd $R
for c in D435 Large_D435; do
  scripts/lang/in_container.sh python -m vla_finetuning.items.render_views --out $R/imgs/zero_qpos --cam $c \
    --tasks selective_avoid --embodiments ur5-wsg franka-panda --zero-qpos --cams head_camera
done
rm -rf $R/imgs/zero_qpos/_runner
