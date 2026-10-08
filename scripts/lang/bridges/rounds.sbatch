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
# (ROUNDS=100, EPISODES=4, STEPS=2048, PROBE=10); GAMMA (default 0.9, the plan's) sets the HJ discount
# (gamma and gamma_final); EXTRA_ARGS goes to `train rounds` unchanged (e.g. "--no-wandb"). Keep the same
# values on every resubmission of one run. Two gamma runs = two RUN directories, e.g.
#   RUN=runs/lang/rounds_g090 sbatch ...   and   RUN=runs/lang/rounds_g095 GAMMA=0.95 sbatch ...
#   Bridges-2:  SBATCH_FLAGS="-A <account> -p GPU-shared --gpus=<rtx type>:1"
#   Delta:      SBATCH_FLAGS="-A bifq-delta-gpu -p gpuA40x4 --gpus-per-node=1 --mem=64g"
#               PY="bash scripts/lang/in_container.sh python"   (the container wrapper)
#SBATCH --job-name=lang-rounds
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=runs/bridges_logs/lang-rounds_%j.out
set -u
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
source scripts/lang/bridges/_env.sh
# online wandb: WANDB_KEY_FILE names a file with an `export WANDB_API_KEY=...` line (read here, never printed)
if [[ -n "${WANDB_KEY_FILE:-}" ]]; then source scripts/lang/wandb_env.sh; fi
RUN="${RUN:?set RUN=runs/lang/<name>}"
ROUNDS=${ROUNDS:-100}; EPISODES=${EPISODES:-4}; STEPS=${STEPS:-2048}; PROBE=${PROBE:-10}; GAMMA=${GAMMA:-0.9}
DATA=${DATA:-data/lang/warmup}; VAL=${VAL:-data/lang/val}   # absolute when the code runs from another checkout
# the run's config is fixed by its first start (lang_config.json); a resumed run keeps it
# more LangConfig overrides go in SET="key=value ...", never in EXTRA_ARGS: a second --set would replace
# this one (argparse keeps the last)
# shellcheck disable=SC2206
GAMMA_ARGS=(--set "gamma=$GAMMA" "gamma_final=$GAMMA" ${SET:-})
end=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null || echo 0)
[[ $end -gt 0 ]] || end=$(( $(date +%s) + 47 * 3600 ))
export STAGE_DEADLINE=$(( end - 600 ))
echo "[rounds.sbatch] job $SLURM_JOB_ID on $(hostname), run $RUN, stop at $(date -d @"$STAGE_DEADLINE")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
state_mtime() { stat -c %Y "$RUN/state.pt" 2>/dev/null || echo 0; }
crashes=0; no_progress_retries=0
while true; do
  before=$(state_mtime)
  # shellcheck disable=SC2086
  $PY -m main.lang.train rounds --data "$DATA" --val "$VAL" --run "$RUN" \
      --collect-rounds "$ROUNDS" --episodes-per-round "$EPISODES" --grad-steps-per-round "$STEPS" \
      --probe-every-rounds "$PROBE" "${GAMMA_ARGS[@]}" --resume ${EXTRA_ARGS:-}
  rc=$?
  echo "[rounds.sbatch] train rounds exit $rc at $(date)"
  if [[ $rc -ne 0 && $rc -ne 75 ]]; then
    # a crash: resume from the last saved point if the run got further since the last launch (up to 3
    # times per job; near the job's end, by resubmitting); a crash that repeats without progress stops the
    # chain for a human to look at
    # one retry without progress per job too: the first save of a launch comes only after its first
    # episode (~5 min) or its first 500 steps (~17 min), and a transient error there must not end the run
    progressed=$(( $(state_mtime) > before ? 1 : 0 ))
    if (( crashes < 3 && (progressed || no_progress_retries < 1) )); then
      crashes=$((crashes + 1)); (( progressed )) || no_progress_retries=$((no_progress_retries + 1))
      if (( $(date +%s) < STAGE_DEADLINE - 1800 )); then
        echo "[rounds.sbatch] crash $crashes (progress: $progressed): resuming"; continue
      fi
      echo "[rounds.sbatch] crash $crashes (progress: $progressed) near the job's end: resubmitting"
      rc=75
    else
      echo "[rounds.sbatch] crash again without progress (or 3 already): stopping; see the log above"
      exit $rc
    fi
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
