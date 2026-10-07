# Phase 4: rollout integration and rounds

## Built

- `LangRolloutController` (`main/lang/rollout.py`): s0 captured before the first physics step outside
  collect mode; `trace["status"]` in {ok, timeout, exception} from a wrapped `play_once`; per-tick
  timing; per-tick `h_all`, `h_named`, unnamed minimum, `action_source`, named item distance, `held`.
- `LangPolicyShim`: CrossSafe's 7-argument call. Frames are pushed into the history only during a
  capture (`_in_capture`); frames [t, t-k, 0] with t-k = frame 0 until k+1 frames exist; the
  cache-miss path uses the fresh frame as t without pushing; a counterfactual encoding when a second
  sentence is set.
- `set_instruction(sentence, cf_sentence, rule_id)`: text store lookup + text adapter once per episode.
- `_tick_hook`: Q_cf = qmin(enc_cf, a_nom) appended to the trace before the base hook.
- Rounds (`main/lang/rounds.py`): round 0 trains on the warmup; each later round collects
  `episodes_per_round` episodes (80% filter) into the run's own ring with the current weights, refreshes
  the sampler, trains. `state.pt` at every round start; exit 75 at the init limit; `--resume` rolls both
  rings back and drops the round's lines, then collects the round again with the same ids; a resume
  fingerprint (step, alpha, round, schedule cursors, RNG hashes, ring n) is written before the exit and
  after the resume. `scripts/lang/run_rounds.sh` relaunches on 75.
- `LangRoundMetrics` (`main/lang/metrics.py`): per filter episode, keyed by (round, task, host,
  embodiment, consequence, designation), heatmaps of tasks x (embodiment x designation) per metric.
- `LangPanelVideoWriter` (`main/lang/video.py`): Section 8.4 panel in h units.

## Tests

CPU (`tests/lang/test_rollout_shim.py`, `test_rounds.py`): history order [t, t-k, 0] at captures 0-5,
cache misses never grow the history, counterfactual encoding present, schedule balance and state
round trip.

GPU / simulator (`python -m main.lang.phase4_checks`, outputs in `runs/lang_phase4/`):

<!-- PHASE4_RESULTS -->

## Smoke run (3 rounds x 2 episodes x 256 grad steps, exit 75 forced before round 3)

<!-- ROUNDS_SMOKE -->
