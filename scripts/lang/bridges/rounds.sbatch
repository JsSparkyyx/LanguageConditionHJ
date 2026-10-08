#!/bin/bash
# The rounds training run (VANILLA_PLAN 6.2) as a self-resubmitting Slurm job.
#
#   cd <repo>
#   export RUN=runs/lang/<name> HOLOBRAIN_CKPT=<dir> PY=<python> SBATCH_FLAGS="-A <account> --gpus=<type>:1"
#   sbatch $SBATCH_FLAGS scripts/lang/bridges/rounds.sbatch
#
# Site flags go in SBATCH_FLAGS (or edit the #SBATCH lines below): the resubmissions reuse them, while
# flags typed only on the first sbatch command line would be lost. The GPU must render
# with ray tracing (RTX class: A40 / L40S / RTX 6000 ...; V100, A100, H100 have no RT cores) and hold the
# trainer (27 GB peak in rounds mode on Delta) plus one simulator and cuRobo: use a 48 GB card.
#
# The job stops the trainer 10 min before its end (state saved inside the round, exit 75) and submits
# itself again (RESUBMIT=0 to disable); the next job resumes where it stopped. A restart for the
# per-process environment-init limit (also exit 75) relaunches inside the same job, and so does a crash
# after progress (state.pt newer than at the launch; at most 3 per job). Defaults are the plan's
# (ROUNDS=100, EPISODES=4, STEPS=2048, PROBE=10); EXTRA_ARGS goes to `train rounds` unchanged
# (e.g. "--no-wandb"). Keep the same values on every resubmission of one run.
#SBATCH --job-name=lang-rounds
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=runs/bridges_logs/lang-rounds_%j.out
set -u
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
source scripts/lang/bridges/_env.sh
RUN="${RUN:?set RUN=runs/lang/<name>}"
ROUNDS=${ROUNDS:-100}; EPISODES=${EPISODES:-4}; STEPS=${STEPS:-2048}; PROBE=${PROBE:-10}
end=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null || echo 0)
[[ $end -gt 0 ]] || end=$(( $(date +%s) + 47 * 3600 ))
export STAGE_DEADLINE=$(( end - 600 ))
echo "[rounds.sbatch] job $SLURM_JOB_ID on $(hostname), run $RUN, stop at $(date -d @"$STAGE_DEADLINE")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
state_mtime() { stat -c %Y "$RUN/state.pt" 2>/dev/null || echo 0; }
crashes=0
while true; do
  before=$(state_mtime)
  # shellcheck disable=SC2086
  $PY -m main.lang.train rounds --data data/lang/warmup --val data/lang/val --run "$RUN" \
      --collect-rounds "$ROUNDS" --episodes-per-round "$EPISODES" --grad-steps-per-round "$STEPS" \
      --probe-every-rounds "$PROBE" --resume ${EXTRA_ARGS:-}
  rc=$?
  echo "[rounds.sbatch] train rounds exit $rc at $(date)"
  if [[ $rc -ne 0 && $rc -ne 75 ]]; then
    # a crash: resume from the last saved point if the run got further since the last launch (up to 3
    # times); a crash that repeats without progress stops the chain for a human to look at
    if (( $(state_mtime) > before && crashes < 3 )) && (( $(date +%s) < STAGE_DEADLINE - 1800 )); then
      crashes=$((crashes + 1)); echo "[rounds.sbatch] crash $crashes after progress: resuming"; continue
    fi
    echo "[rounds.sbatch] crash without progress (or 3 already): stopping; see the log above"
    exit $rc
  fi
  [[ $rc -eq 75 ]] || break
  if (( $(date +%s) > STAGE_DEADLINE - 1800 )); then   # out of job time (not the init limit)
    if [[ "${RESUBMIT:-1}" == 1 ]]; then
      # shellcheck disable=SC2086
      sbatch --export=ALL ${SBATCH_FLAGS:-} "${SLURM_SUBMIT_DIR}/scripts/lang/bridges/rounds.sbatch" \
        && echo "[rounds.sbatch] resubmitted"
    fi
    exit 75
  fi
done
[[ $rc -eq 0 ]] && echo "[rounds.sbatch] run complete: $RUN/checkpoint_final.pt"
exit $rc
