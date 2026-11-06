#!/bin/bash
#SBATCH --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1 --cpus-per-task=8 --mem=40g --time=00:30:00
#SBATCH --job-name=vf-render -o /work/nvme/bifq/project/tmp/logs/vf_render_%j.out
# ur5-wsg and piper in one original RoboTwin scene (place_container_plate), all four arms at the same joint
# values, ur5 and piper alike (Yuxuan 11-05), no contact, both arms in the observer view (
# items/render_shared_pose.py) -> imgs/shared_pose/.
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ
cd $R
export TMPDIR=/tmp
scripts/lang/in_container.sh python -m vla_finetuning.items.render_shared_pose --out $R/imgs/shared_pose "$@"
