#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA100x4-interactive,gpuA100x8-interactive,gpuH200x8-interactive,gpuA40x4-interactive
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 --mem=60g --time=01:00:00
#SBATCH --job-name=vf-ftx-train -o /work/nvme/bifq/project/tmp/logs/vf_ftx_train_%j.out
# The training part of the item-scene fine-tune's extension run <RUN> (ft_items_pipeline.sh step 3) on whichever
# 1-GPU interactive slot starts first (Yuxuan 10-30: the training needs no A40, only the sim evaluation does). Same
# recipe as the chain: one GPU, GRAD_ACCUM 4 (batch 32), 15k steps from <PREV>'s final weights, resumed from the
# newest complete checkpoint each hour. The chain holds ft_items meanwhile (runs/lang_pipeline_ft/state/hold.ft_items)
# and does the export, the A40 evaluation and the decision once <RUN>.DONE exists. Submitted by ft_x_feeder.sh:
#   RUN=ft_items_safe_x2 PREV=ft_items_safe sbatch --export=ALL vla_finetuning/jobs/ft_x_train.sbatch
set -u
R=/work/nvme/bifq/project/LanguageConditionHJ; WS=$R/vla_finetuning/ws
cd $R
RUN=${RUN:?}; PREV=${PREV:?}
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print s}')
export STAGE_DEADLINE=$(( $(date +%s) + left - 240 ))
export GRAD_ACCUM=4
PCK=$(head -1 "$WS/$PREV.DONE")
[ -f "$PCK/model/model.safetensors" ] || { echo "[ftx] no final checkpoint of $PREV ($PCK)"; exit 1; }
L=$(printf "$R/vla_finetuning/data/lmdb/items_safe_%s," selective_avoid push_zone tray_transport clean_pour gap_place)
L=${L%,}
echo "[ftx] $RUN from $PREV ($PCK) on $(hostname) $SLURM_JOB_PARTITION $(nvidia-smi --query-gpu=name --format=csv,noheader)," \
     "deadline $(date -d @$STAGE_DEADLINE +%T)"
scripts/lang/in_container.sh bash -c "python vla_finetuning/jobs/ft_items_decide.py prune $RUN; \
  INIT_CKPT=$PCK/model/model.safetensors bash vla_finetuning/jobs/ft_chain_step.sh $RUN $L 15000 ${SAVE_EVERY:-250} 1 \
  vla_finetuning/jobs/prep_items.sh"
rc=$?
echo "[ftx] rc=$rc $(date '+%F %T')"
exit $rc
