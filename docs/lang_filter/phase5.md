# Phase 5: closed-loop evaluation and report tooling

## Built

- `main/lang/eval_closed.py`: the Section 9.2 protocol. Per scene: 2 nominal runs (cached in
  `runs/lang/_nominal_cache/`, keyed by task, host, embodiment, seed, episode cap, env code hash and bank
  hash, shared by all checkpoints; redraws on failed builds), then a harm- and a benign-designated rule
  per axis (eligible for the axis, chosen by a hash of seed and axis), one sentence each, one filtered
  episode each with Q_cf from the other rule's sentence. Rule table and designations are checked against
  the nominal runs. Per-run metrics: harm_named, intervention rate, lead time, unnamed harm, v_le_h,
  hq_corr, q_cf_gap. Aggregates per group and axis with Wilson intervals: harm / benign rates nominal and
  filtered, protect rate, benign intervention, false alarm, success cost, selectivity, nominal
  disagreement. Children per (embodiment, shard) run in parallel, restart at the init limit, and resume
  per row.
- `main/lang/report.py`: one table per research question (Q1-Q4) plus the closed-loop table, from each
  run's latest `eval_offline/*/metrics.json` and `eval_closed/*/summary.json`; variants read from the
  run's config.

## Cost

The full grid at 20 scenes per (task, embodiment) on axes train + heldout_template is up to 1120 filtered
episodes per checkpoint plus 560 nominal runs, about 90 h of simulation (~5 min per filtered episode
with 5 per A40), i.e. ~4.5 h on 20 parallel children (one 4-GPU node) per checkpoint.

## Smoke grid (train-smoke checkpoint, 1 scene per combo, train axis)

<!-- EVAL_SMOKE -->
