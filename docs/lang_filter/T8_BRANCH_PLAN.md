# T8 branch data: the same drag state continued with different actions (Yuxuan, 2026-11-17)

## 1. Why

The T8 warmup is planner-only: every state appears with exactly one action. In the action-sensitivity study (T5,
`ACTION_SENSITIVITY_PLAN.md`), a critic's Q started to respond to the action only when its batches also held
**branch rows**: one saved simulator state continued with several different actions, each with its own outcome.
The 3-tick critic with the revised injection and normal rows only fell from a Q spread of 0.032 over random actions
to 0.003; with branch rows it rose to 0.30 by step 6,000. The T8 3-tick runs of 2026-11-17 have the 3-tick decision
and the revised injection but no branch rows.

## 2. Collection (`scripts/lang/action_sens/collect_branches.py --task tray_slide`)

- **Parent episodes**: 90 planner-only T8 drags, ur5-wsg, no kicks: seeds 7001-7090, behaviours in the cycle medium,
  medium, fast, medium, slow, medium. Every fifth seed is held out (18 episodes): its states are kept for probes and
  neither its branches nor its planner episode train.
- **Branch states**: 3 per drag, at about 0.75, 0.5 and 0.25 s before the planner's brake (shifted per episode by
  up to 0.08 s). The boundary check of 2026-11-17 puts the decisions that matter in this window: stopping dead saves
  the glass only 0.7 s or more before the brake, a gentle slow-down from 0.2 to 0.65 s.
- **Eight continuations per state**, each run twice from the restored state (physics only for the truth, then with a
  capture every tick, stored as an episode of a normal shard behind 3 context rows):

  | name | what the arm does | command type |
  |---|---|---|
  | `plan` | the planner | planner |
  | `t3_c0`, `t3_c50` | the plan at 0 / 0.5 x its rate for 3 ticks, then at full rate | planner |
  | `half` | the plan at half rate to its end | planner |
  | `slow25` | the rate falls linearly to 0 over 25 ticks (1 s), then holds | planner |
  | `stop` | the zero command from the first tick | actor-style |
  | `d_slow25`, `d_slow12` | per-tick displacement commands a0 x g held in 3-tick blocks, g falling to 0 over 25 / 12 ticks | actor-style |

  a0 = the arm's joint displacement over the last tick (its current speed). The actor-style commands go through
  `env.step_dtheta`, as the filter's actor does, and are stored with action source 1. They show what the actor's own
  command space can do; the planner-style ones are stored as the planner's commands are (the target at the tick's end
  minus the live joint state).
- **Truth per continuation**: V95, min h and "fell" of the glass rule over the whole continuation (the can's h is in
  the stored rows too). Example: a state 0.45 s before the brake where `plan` and `stop` tip the glass (V95 0.3 and
  -0.4) and `slow25` keeps it up (V95 0.97).
- Size: at most 60 ticks are stored per continuation (they cover the brake and any fall); about 120,000 rows,
  90 GB. The first plan (120 drags x 4 states x 10 continuations, stored to rest) would have taken about 270 GB of
  the 590 GB free on 2026-11-17.
- Output: `data/lang_t8_branch/parents/shard_g*` (planner episodes), `branches/shard_g*` (branch episodes),
  `states/<episode>.pt` (encoder inputs, the first 12 commands of every continuation, the truths),
  `branch_table.json`.

### 2.1 Recovery continuations (Yuxuan, 2026-11-17)

The continuations above that slow down then hold the arm: the tray stops short of the pad and the task never
finishes. Two more continuations slow down with actor-style commands and then hand back to the planner, which replans
from the live state and finishes the drag (`main/lang/controller.py` ResumeViaController: the waypoints ahead, the
remaining length, from the arm's current speed), as the filter does when it engages and releases:

| name | what the arm does |
|---|---|
| `d_slow12_resume` | actor-style slow-down to rest over 12 ticks (0.48 s), then the planner finishes the drag |
| `d_half_resume` | actor-style half speed for 6 ticks, quarter speed for 6, then the planner takes over while moving |

Up to 100 ticks are stored per recovery continuation, so the planner's brake at the pad is in the stored rows. In the
release check (`scripts/lang/t8_release_check.py`) the resumed medium drag still tipped the glass at its own hard
brake or when it sped back up: releasing during a medium drag is harmful unless the filter takes over again before
the pad. These rows show the critic exactly that. They are collected for the main data as a supplement over the same
90 drags (`data/lang_t8_branch_resume`, `--decisions d_slow12_resume,d_half_resume`; only its `branches/` train,
its re-collected parents duplicate the main ones); the YOLO branch data collects all 10 continuations in one pass.
Branch episode ids come from a fixed registry of the 10 names, so a subset run gives the same ids.

### 2.2 Finishing continuations: slow down, then finish slowly (Yuxuan, 2026-11-18)

The recovery continuations of 2.1 save the task, not the glass: over the 90 drags the glass fell in 96 % (medium) and
100 % (fast) of `d_slow12_resume` and `d_half_resume`, as often as under the plan (97 %, 100 %). The resumed planner
climbs back to the profile's cruise speed and brakes hard at the pad, which is what tips the glass; the slow-down
alone works when it starts early (`d_slow12`, held: 9 of 60 medium glasses fell 0.75 s before the brake, the plan
58 of 60). The continuations that kept the glass up all stop short of the pad. So no branch showed a real recovery.
The recovery warmup does (`recover_glass`: 0.45 -> 0.15 m/s over 12 cm, then slow to the pad: 2 of 14 fell, all
finished), but as planner-only episodes, one action per state, its slow-down planned from the start.

Three more continuations slow down with actor-style commands, then hand back to the planner capped at
`FINISH_SPEED` = 0.15 m/s (the `slow` behaviour's cruise speed, `recover_glass`'s second speed;
`controller.capped_profile`: the original peak acceleration, the original 4 cm stop distance, so the pad brake peaks
at 0.42 m/s^2 instead of 3.8; an arm handed over faster than 0.15 m/s first slows to it):

| name | what the arm does |
|---|---|
| `d_slow12_finish` | actor-style slow-down to rest over 12 ticks, then the planner finishes at <= 0.15 m/s |
| `d_slow25_finish` | the same over 25 ticks (1 s) |
| `d_half_finish` | actor-style half speed for 6 ticks, quarter speed for 6, then the planner finishes at <= 0.15 m/s |

Up to 130 ticks are stored per finishing continuation (to the pad at 0.15 m/s). Collected as a supplement over the
same 90 drags (`data/lang_t8_branch_finish`, `--decisions d_slow12_finish,d_slow25_finish,d_half_finish`, with the
YOLO detector's target masks so the YOLO run can use the same rows; only `branches/` trains), after a check on 4
drags with videos (`data/lang_t8_branch_finish_check`).

## 3. Training mix (`main/lang/data.py`, `LangConfig.branch_cont_frac`, `branch_decision_frac`)

- Rows of a branch episode: the 3 context rows are never sample rows; the first 3 rows from the branch state are
  **decision rows** (with `decision_ticks` 3 their action window is the branch's first 3 commands); the rest are
  **continuation rows**.
- Each batch of 64: 32 normal rows (warmup, recovery warmup, the planner parents, the run's ring), 16 continuation
  rows, 16 decision rows (`branch_cont_frac = branch_decision_frac = 0.25`, the study's mix). Pairs (one transition
  with a harmful and a benign rule) are drawn inside each group.

## 4. Monitors

- `collapse/*` and `collapse_round/*` (2026-11-17) as before.
- Planned: the held-out states as a fixed probe with ground truth: over the three continuations that share the
  planner as their continuation (`plan`, `t3_c0`, `t3_c50`), does Q order them as their V95 does (rank agreement),
  and does Q move as much as V95 (spread ratio), the study's two scores.

## 5. Caveat

The sustained continuations (`half`, `slow*`, `d_*`) differ from `plan` mostly AFTER their first 3 ticks, so their
outcome is not caused by the 3-tick decision alone. They add states in which the tray is already slowing and the
glass survives, and rows whose commands are slow-downs. Whether the critic then credits the first decision depends on
the actor in its target, as before.
