#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=8 --mem=40g --time=01:00:00
#SBATCH --job-name=vf-render -o /work/nvme/bifq/project/tmp/logs/vf_render_%j.out
# ur5-wsg and piper in one original RoboTwin scene (place_container_plate), both at the same arm joint values,
# searched for a natural pose on both (Yuxuan 11-05; items/render_same_joints.py) -> imgs/same_joints/.
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
cd $R
scripts/lang/in_container.sh python -m vla_finetuning.items.render_same_joints --out $R/imgs/same_joints "$@"
