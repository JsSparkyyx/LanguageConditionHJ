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

| check | result |
|---|---|
| one filter episode per task and embodiment with a random-init language model (`filter`) | **14 / 14** ran to completion with the filter engaged (named rule harm- or benign-designated, a training sentence, Q_cf from the other designation when it exists), StepBuffer + sidecar + `lang_episodes.jsonl` in lockstep (h parity 0.0, rows match the line), one panel video each. A random critic gives Q < 0 everywhere, so the actor drives the whole episode and every one hits the 750-tick cap. T3 ur5's first scene had no sampleable rule (a held-out box), which the loop correctly ran nominal; the check now retries seeds until a filter episode runs (`filter_v2`) |
| deployment frame 0 == collection's s0 row, same seed (`frame0`, T5, both embodiments) | **max abs difference 0.0** (fp16), frame scale ~3.5 |
| exactly one capture before the first physics step | **1** in collect mode and **1** in deployment mode, both embodiments |
| history never grows on cache misses | CPU test (`test_rollout_shim.py`) |
| `trace["status"]` | forced timeout -> `timeout`; forced exception in physics -> `exception` (`RuntimeError: forced failure for the status test`) |
| resume after exit 75 | rounds smoke below (`resume_check_pre.json` vs `resume_check_post.json`) |

Panel check (one frame, T7 harm run): instruction, rule id, h_named = -1.000 with a red border (the random
actor tipped the tall block), Q, Q_cf, the most at-risk unnamed rule, FILTER, intervened ticks, block counter.

## Smoke run (3 rounds x 2 episodes x 256 grad steps, exit 75 forced before round 3)

<!-- ROUNDS_SMOKE -->
