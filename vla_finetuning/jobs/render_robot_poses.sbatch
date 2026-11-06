#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=8 --mem=40g --time=00:45:00
#SBATCH --job-name=vf-render -o /work/nvme/bifq/project/tmp/logs/vf_render_%j.out
# ur5-wsg and piper in one original RoboTwin scene (place_container_plate), each in its own natural pose with no
# self contact and both arms fully in the head camera picture (Yuxuan 11-05; items/render_robot_poses.py)
# -> imgs/robot_poses/.
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
cd $R
export TMPDIR=/tmp
scripts/lang/in_container.sh python -m vla_finetuning.items.render_robot_poses --out $R/imgs/robot_poses "$@"
