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

Machinery check on the 300-step recon checkpoint (`runs/lang_recon/train_smoke`): nominal cache, designated
named rules, Q_cf, videos, per-row resume across killed jobs all work.

### Small grid on the preview checkpoint (step 8000, 36% of the warmup), PARTIAL at 08:00

`runs/lang/lang_full_paired_off_preview/eval_closed/grid_ckpt/` (2 scenes per combo, axes train +
heldout_template, 8 children over A40 nodes; still running and resuming across jobs; regenerate with
`python -m main.lang.eval_closed --run runs/lang/lang_full_paired_off_preview --ckpt grid_ckpt.pt --merge-only`):

harm runs name a harm-designated rule (its item is in the way), benign runs a benign-designated one. protect_rate = 1 - harm_rate_filter / harm_rate_nominal; benign_false_alarm = P[any intervention | benign].

Counts: {'filter_runs': 47, 'missing': 25, 'errors': 0}

| group / axis | harm nominal | harm filter | protect | benign nominal | benign false alarm | benign interv. | success cost (harm / benign) | selectivity | nominal disagreement |
|---|---|---|---|---|---|---|---|---|---|
| clean_pour/heldout_template | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | nan | – | nan | +0.00 / +nan | nan | 0.00 |
| clean_pour/train | 0.50 | 0.50 [0.09, 0.91] (2) | 0.00 | 0.50 | 1.00 [0.21, 1.00] (1) | 0.011 | +0.25 / +0.00 | -1.00 | 0.33 |
| clean_pour|franka-panda/train | 0.00 | 0.00 [0.00, 0.79] (1) | nan | nan | – | nan | +0.50 / +nan | nan | 0.00 |
| clean_pour|ur5-wsg/heldout_template | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | nan | – | nan | +0.00 / +nan | nan | 0.00 |
| clean_pour|ur5-wsg/train | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | 0.50 | 1.00 [0.21, 1.00] (1) | 0.011 | +0.00 / +0.00 | -1.00 | 0.50 |
| gap_place/train | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | nan | – | nan | +0.00 / +nan | nan | 0.00 |
| gap_place|ur5-wsg/train | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | nan | – | nan | +0.00 / +nan | nan | 0.00 |
| granular_carry/heldout_template | 0.33 | 0.33 [0.06, 0.79] (3) | 0.00 | nan | – | nan | +0.17 / +nan | nan | 0.00 |
| granular_carry/train | 0.38 | 0.50 [0.15, 0.85] (4) | -0.33 | nan | – | nan | +0.62 / +nan | nan | 0.25 |
| granular_carry|franka-panda/heldout_template | 0.00 | 0.00 [0.00, 0.79] (1) | nan | nan | – | nan | +0.00 / +nan | nan | 0.00 |
| granular_carry|franka-panda/train | 0.25 | 0.50 [0.09, 0.91] (2) | -1.00 | nan | – | nan | +1.00 / +nan | nan | 0.50 |
| granular_carry|ur5-wsg/heldout_template | 0.50 | 0.50 [0.09, 0.91] (2) | 0.00 | nan | – | nan | +0.25 / +nan | nan | 0.00 |
| granular_carry|ur5-wsg/train | 0.50 | 0.50 [0.09, 0.91] (2) | 0.00 | nan | – | nan | +0.25 / +nan | nan | 0.00 |
| granular_pour/heldout_template | 1.00 | 0.75 [0.30, 0.95] (4) | 0.25 | nan | – | nan | +0.62 / +nan | nan | 0.00 |
| granular_pour/train | 1.00 | 0.75 [0.30, 0.95] (4) | 0.25 | nan | – | nan | +0.62 / +nan | nan | 0.00 |
| granular_pour|franka-panda/heldout_template | 1.00 | 1.00 [0.34, 1.00] (2) | 0.00 | nan | – | nan | +0.25 / +nan | nan | 0.00 |
| granular_pour|franka-panda/train | 1.00 | 0.50 [0.09, 0.91] (2) | 0.50 | nan | – | nan | +0.25 / +nan | nan | 0.00 |
| granular_pour|ur5-wsg/heldout_template | 1.00 | 0.50 [0.09, 0.91] (2) | 0.50 | nan | – | nan | +1.00 / +nan | nan | 0.00 |
| granular_pour|ur5-wsg/train | 1.00 | 1.00 [0.34, 1.00] (2) | 0.00 | nan | – | nan | +1.00 / +nan | nan | 0.00 |
| pooled/heldout_template | 0.83 | 0.58 [0.32, 0.81] (12) | 0.30 | 0.00 | 0.43 [0.16, 0.75] (7) | 0.156 | +0.46 / +0.29 | -0.13 | 0.00 |
| pooled/train | 0.81 | 0.72 [0.49, 0.88] (18) | 0.10 | 0.05 | 0.20 [0.06, 0.51] (10) | 0.012 | +0.56 / +0.20 | -0.10 | 0.07 |
| push_zone/heldout_template | 1.00 | 0.33 [0.06, 0.79] (3) | 0.67 | 0.00 | 0.50 [0.09, 0.91] (2) | 0.330 | +0.50 / +0.00 | 0.17 | 0.00 |
| push_zone/train | 1.00 | 0.75 [0.30, 0.95] (4) | 0.25 | 0.00 | 0.33 [0.06, 0.79] (3) | 0.035 | +0.38 / +0.33 | -0.08 | 0.00 |
| push_zone|franka-panda/heldout_template | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | 0.00 | 0.00 [0.00, 0.79] (1) | 0.000 | +0.00 / -1.00 | 0.00 | 0.00 |
| push_zone|franka-panda/train | 1.00 | 1.00 [0.34, 1.00] (2) | 0.00 | 0.00 | 0.50 [0.09, 0.91] (2) | 0.053 | +0.00 / +0.50 | -0.50 | 0.00 |
| push_zone|ur5-wsg/heldout_template | 1.00 | 0.00 [0.00, 0.66] (2) | 1.00 | 0.00 | 1.00 [0.21, 1.00] (1) | 0.660 | +0.75 / +1.00 | 0.00 | 0.00 |
| push_zone|ur5-wsg/train | 1.00 | 0.50 [0.09, 0.91] (2) | 0.50 | 0.00 | 0.00 [0.00, 0.79] (1) | 0.000 | +0.75 / +0.00 | 0.50 | 0.00 |
| selective_avoid/heldout_template | nan | – | nan | 0.00 | 0.50 [0.15, 0.85] (4) | 0.107 | +nan / +0.50 | nan | 0.00 |
| selective_avoid/train | nan | – | nan | 0.00 | 0.00 [0.00, 0.49] (4) | 0.000 | +nan / +0.25 | nan | 0.00 |
| selective_avoid|franka-panda/heldout_template | nan | – | nan | 0.00 | 0.50 [0.09, 0.91] (2) | 0.172 | +nan / +0.50 | nan | 0.00 |
| selective_avoid|franka-panda/train | nan | – | nan | 0.00 | 0.00 [0.00, 0.66] (2) | 0.000 | +nan / +0.00 | nan | 0.00 |
| selective_avoid|ur5-wsg/heldout_template | nan | – | nan | 0.00 | 0.50 [0.09, 0.91] (2) | 0.042 | +nan / +0.50 | nan | 0.00 |
| selective_avoid|ur5-wsg/train | nan | – | nan | 0.00 | 0.00 [0.00, 0.66] (2) | 0.000 | +nan / +0.50 | nan | 0.00 |
| tray_transport/heldout_template | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | 0.00 | 0.00 [0.00, 0.79] (1) | 0.000 | +1.00 / +0.00 | 0.00 | 0.00 |
| tray_transport/train | 1.00 | 1.00 [0.44, 1.00] (3) | 0.00 | 0.00 | 0.00 [0.00, 0.66] (2) | 0.000 | +1.00 / +0.00 | 0.00 | 0.00 |
| tray_transport|franka-panda/train | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | nan | – | nan | +1.00 / +nan | nan | 0.00 |
| tray_transport|ur5-wsg/heldout_template | 1.00 | 1.00 [0.21, 1.00] (1) | 0.00 | 0.00 | 0.00 [0.00, 0.79] (1) | 0.000 | +1.00 / +0.00 | 0.00 | 0.00 |
| tray_transport|ur5-wsg/train | 1.00 | 1.00 [0.34, 1.00] (2) | 0.00 | 0.00 | 0.00 [0.00, 0.66] (2) | 0.000 | +1.00 / +0.00 | 0.00 | 0.00 |

Reading (small n, Wilson intervals in brackets): the early checkpoint barely protects in closed loop.
On the train axis nominal runs harm the named item 81% of the time, filtered runs 72% (protect rate
0.10). Benign runs rarely intervene (1.2% of ticks), but 20% of them intervene at least once. When the
filter intervenes the task usually fails (episodes run to the cap). Offline ranking is well above
chance (`phase3.md`), but the actor's recovery behaviour is untrained: the offline data is nominal only
and the actor is trained only against critics fit on nominal transitions. Rounds training (filter
episodes in the data) is the plan's remedy. Missing rows are expected: T1 scenes rarely have an
on-path (harm-designated) candidate, and T2 / T4 have no benign rule.

