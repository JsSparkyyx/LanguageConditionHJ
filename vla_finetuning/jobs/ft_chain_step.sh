#!/bin/bash
# One step of a fine-tune run as a chain of 1-hour interactive jobs (inside the container; a stage command of
# scripts/lang/chain6.sh, which sets STAGE_DEADLINE): the batch A100 queue was 10+ days long on 2026-10-28, the
# interactive one hours. Each step resumes train.sh from the run's newest COMPLETE checkpoint (HoloBrain's trainer
# restores its step count, so max_step stays the total), trains until 2 min before the stage deadline and exits 75
# (the chain resubmits); the step that reaches max_step writes ws/<run>.DONE (the final checkpoint's path) and exits 0.
#   bash vla_finetuning/jobs/ft_chain_step.sh <run> <lmdb> <max_step> <save_every> <n_gpu> [prep script]
# A checkpoint is complete when model, optimizer, scheduler and trainer-progress files are all there and it is older
# than 2 min (a save cut part-way misses its last file, model/model.safetensors). The prep script (prep_items.sh) packs the data first; its 75
# (demos still short) passes through, so the chain tries again later.
set -uo pipefail
RUN=$1; LMDB=$2; MAX=$3; SAVE=$4; N=$5; PREP=${6:-}
R=/work/nvme/bifq/project/LanguageConditionHJ; WS=$R/vla_finetuning/ws
cd "$R"
[ -f "$WS/$RUN.DONE" ] && { echo "[ft_chain] $RUN done: $(cat "$WS/$RUN.DONE")"; exit 0; }
if [ -n "$PREP" ]; then
  bash "$PREP"; rc=$?
  [ $rc -eq 0 ] || exit $rc
fi
newest() {   # [min age s, default 120]
  python - "$WS" "$RUN" "${1:-120}" <<'EOF'
import sys, time
from pathlib import Path
ws, run, age = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
need = ("model.safetensors", "optimizer.bin", "scheduler.bin", "custom_checkpoint_0.pkl", "model/model.safetensors")
cands = []
for d in [*ws.glob(f"{run}/checkpoints/checkpoint_*"), *ws.glob(f"{run}_r*/checkpoints/checkpoint_*")]:
    if all((d / f).exists() for f in need):
        t = max((d / f).stat().st_mtime for f in need)
        if time.time() - t > age:
            cands.append((t, str(d)))
print(max(cands)[1] if cands else "")
EOF
}
CK=$(newest)
if [ -z "$CK" ] && [ -d "$WS/$RUN" ]; then   # a first step cut before its first save: keep it aside, start again
  mv "$WS/$RUN" "$WS/${RUN}_cut_$(date +%s)"
fi
left=$(( ${STAGE_DEADLINE:?} - $(date +%s) - 120 ))
[ $left -gt 600 ] || { echo "[ft_chain] only ${left}s left: next job"; exit 75; }
echo "[ft_chain] $RUN from ${CK:-scratch}, ${left}s of training"
t0=$(date +%s)
timeout --signal=INT --kill-after=60 $left bash vla_finetuning/jobs/train.sh "$RUN" "$LMDB" "$MAX" "$SAVE" "$N" $CK
rc=$?
el=$(( $(date +%s) - t0 ))
echo "[ft_chain] train.sh exit $rc after ${el}s $(date)"
if [ $rc -eq 0 ]; then
  newest 0 > "$WS/$RUN.DONE"
  echo "[ft_chain] $RUN reached max_step: $(cat "$WS/$RUN.DONE")"; exit 0
fi
# a deadline cut: 124 (timeout's INT worked), or 137 when the window really ran out (its KILL 60 s later). An earlier
# 137 is an outside SIGKILL (e.g. the cgroup OOM killer) and fails the job instead of resubmitting it (review 10-28);
# a cut that left no new complete checkpoint stops too, so a run that cannot progress never loops
if [ $rc -eq 124 ] || { [ $rc -eq 137 ] && [ $el -ge $(( left - 5 )) ]; }; then
  CK2=$(newest 0)
  if [ "$CK2" = "$CK" ]; then
    echo "[ft_chain] cut after ${el}s with no new complete checkpoint (still ${CK:-none}): stopping, not resubmitting"
    exit 1
  fi
  exit 75
fi
echo "[ft_chain] train.sh died early (rc $rc after ${el}s of ${left}s): not a deadline cut"
exit $rc
