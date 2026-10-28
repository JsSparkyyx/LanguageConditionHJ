# Safe controllers for the item tasks (for safe-demonstration VLA fine-tuning)

2026-10-22. Investigation only: no code, config or job was changed or run. Numbers come from the docs, the
code and the saved 2026-10-05 runs (read with `jq`, nothing re-simulated). Updated 2026-10-23 (text only)
for Yuxuan's decisions of that day: T2 replaced by a redesigned T6, and tilt limits of 45 deg in T3, T5, T7.

Question: to fine-tune HoloBrain on SAFE demonstrations of every item task (not only collision avoidance),
we need an expert per task that finishes the job without harming the items. What does each task's nominal
do wrong, what would a safe expert have to guarantee, what already exists, what must be written, and how do
we decide which demos to keep?

Tasks in use (Yuxuan, 2026-10-23): T1 `selective_avoid`, T3 `push_zone`, T5 `tray_transport`, T6 `clean_pour`
(redesigned: three bowls, section 8.1), T7 `gap_place`. T2 `granular_pour` was replaced by the new T6 that day,
because its one rule is not text-conditioned, so T2 is no longer a safe-demo task; section 5 stays as the
record, and its pour is the one T6 uses. T4 was dropped on 2026-10-22 (section 8.2). When this was written
(2026-10-22) the tasks were T1, T2, T3, T5, T7, with T4 and the old T6 dropped.

## Decisions (Yuxuan, 2026-10-22)

- **Robot**: ur5-wsg only.
- **T1 keep rule**: no physical contact with any extra object (PhysX separation <= 0 at any physics step), as in
  the collision-avoidance demos. The filter's 1 cm safety buffer is not required.
- **T2 keep rule**: no bead out, at all (zero beads on the table). A safe pour still has to be found first.
  (T2 is no longer a safe-demo task from 2026-10-23.)
- **T3 demos** (clarified the same day): every demo names one rule, and every demo keeps it.
  - 80%: the named rule is about an element **in the way** (a mat or cylinder on the push path). The demo is the
    SAFE trajectory: the push goes around that element.
  - 20%: the named rule is about an element **not in the way**. The demo is the plain NOMINAL trajectory (it
    already keeps that rule; other elements may be harmed).
- **The same 80 / 20 split for T1, T5, T7** (Yuxuan, yes): 80% the named item is in the way and the demo is the
  safe trajectory (T1 obstacle-aware plan, T5 slow drag when the tall block is named, T7 centred slide when the
  scraped neighbour is named); 20% the named item is not in the way and the demo is the nominal trajectory.
- **Where the code goes**: into `envs/`, as an additional mode of the nominal controller (the default careless
  mode stays as it is), not a separate package.
- **T6 (Yuxuan, 2026-10-23)**: the redesigned T6 replaces T2. Its safe expert is the safe mode of the nominal
  (`nominal_mode="safe"`): the pour aimed at the target bowl's centre, low and slow. Keep rule: no bead in the
  named bowl (section 8.1).

## Decisions (Yuxuan, 2026-10-27): the demo set for HoloBrain fine-tuning

These replace the 80 / 20 named-rule split of 2026-10-22 for the HoloBrain demos ("setting similar to collision
avoidance"):

- **All items protected, plain instruction**: the instruction is the task only (no constraint sentence), and every
  demo keeps every item safe, as in the collision-avoidance set: T1 touches no extra object, T3 goes around every
  element in the way, T5 and T7 disturb no block, T6 puts no bead in either non-target bowl.
- **100 kept demos per task**: T1, T3, T5, T6, T7; T1's 100 are 20 per host (its 5 hosts rotate).
- **Keep rule**: the task succeeds and no protected rule is broken, h >= 0 at every tick (tilt under 45 deg, moved
  less than the stay limit, the box off the mats, no bead in a protected bowl); for T1, no physical contact
  (PhysX separation <= 0) between any extra object and anything but the table at any physics step.
- ur5-wsg only (2026-10-22).

## Status (2026-10-23)

Sections 1 to 10 are the plan of 2026-10-22. Since then the safe modes were built into `envs/`, as decided:
`ItemEpisodeRunner(..., nominal_mode="safe")` pins each task yaml's `safe_mode` section (careless stays the
default, and a seed builds the same scene in both modes).

| task | safe mode | first check (ur5-wsg) |
|---|---|---|
| T5 tray | slow drag (`time_scale` 0.5), commit 4c6a158 | 8/8 clean |
| T7 gap | centred slide (`place_offset` 0), commit 4c6a158 | 8/8 clean |
| T1 avoid | obstacle-aware cuRobo: the protected objects in its world (+1 cm), the WSG hand and the held object as attached spheres, no mplib fallback (`envs/main/item_envs/safe_plan.py`), commit 22b5b3a | 4/8 finished the job, all 8 with no touch; careless touched the protected object in 5/8 |
| T6 bowls | the pour aimed at the target's centre, low and slow (`T6_clean_pour.yaml` `safe_mode`), with the ur5 aim lead fit to the plain bowls | 21/24 finished; 1/24 put beads in the named bowl (section 8.1) |
| T3 push | built 2026-10-27, branch `safe-demos/t3`: the push goes around every protected element along a route planned from the live scene (`envs/robotwin_tasks/_push_route.py`, pure numpy), pushed by `_item_utils.push_via` with every move re-checked first, cuRobo only, the ur5 wrist_3 limit kept (`T3_push_zone.yaml` `safe_mode`, `plan: route`); a layout with no route is not pushed (`safe_fail` "no_route") | no simulator run yet. Off-GPU on replayed layouts (`envs/scripts/item_envs/t3_route_feasibility.py`): a route for 107/120 with every element protected (52/60 on seeds 0-59); each named element alone 210/210 on the path, 79/79 off it. GPU pilot pending (`envs/scripts/item_envs/t3_safe_pilot.py`) |

Still missing: the T3 GPU pilot (success, cylinder touches at every physics step, joint travel, physics steps), and the demo collector for the item scenes.

## 1. Short answer

| task | safe expert | status |
|---|---|---|
| T5 tray | drag slowly (`time_scale` 0.5) | exists: a knob. 40/40 clean in the safe-knob runs |
| T7 gap | slide centred (`place_offset` 0) | exists: a knob. 40/40 clean in the safe-knob runs |
| T1 avoid | plan around the objects (cuRobo world + the hand model of `vla_finetuning/`) | mostly exists for ONE box; needs a multi-object version |
| T3 push | push along a detour (via points around the elements) | missing: a route planner and a pusher that follows it |
| T6 bowls (2026-10-23) | pour aimed at the target bowl's centre, low and slow; keep rule: no bead in the named bowl | exists: the safe nominal mode (`T6_clean_pour.yaml` `safe_mode`). Calibrated 2026-10-23: 21/24 finished, 1/24 broke the keep rule |
| T2 pour (not a safe-demo task from 2026-10-23) | a pour that loses no bead | missing, and unclear if it can be done: even the gentlest pour spills (ur5 0/20 clean) |

Everything also needs one shared piece: a demo collector for the item runner (frames, contact checks, keep
rule, HoloBrain files, task instructions). The CrossSafe one (`vla_finetuning/collect_demos.py`) only knows
the one-box scenes.

## 2. Words and symbols used here

- **tick**: one control step, 1/25 s = 10 physics steps (physics runs at 250 Hz).
- **rule r**: one safety rule of the scene, for example "don't touch the red can".
- **h_r(t)**: the safety number of rule r at tick t. `h = clip(margin / h_unit, -1, 1)`
  (`docs/lang_filter/DESIGN_CONTEXT.md:77`, `envs/main/item_envs/margins.py:69-133`). h >= 0 is safe,
  h < 0 is harmed. `margin` is the rule's own measure (a gap in metres, a tilt allowance, ...), `h_unit`
  is 5 cm for distances and zones, 1 cm for beads, the tilt limit for "upright" (45 deg in T3, T5, T7 from
  2026-10-23; 10 deg when this was written), the movement limit for "stay".
  Example: a 3 cm gap under a 1 cm avoid buffer gives margin 0.02 m, h = 0.02 / 0.05 = 0.4.
  Bead rules count beads once broken: one bead out reads -0.5, 20 or more read -1
  (`DESIGN_CONTEXT.md:85`).
- **all-items safe**: every rule of the scene keeps h >= 0 for the whole episode. This is what a demo needs
  when the VLA gets no constraint sentence (it cannot know which item matters, so it must protect all).
- **named-rule safe**: only the one named rule keeps h >= 0; the others may be broken. This is what a demo
  needs when the instruction carries the constraint sentence ("... and don't touch the red can").
- **nominal**: the task's own scripted expert (`play_once` in the RoboTwin task file).
- **time_scale (s)**: speed multiplier for the "risky" parts of the nominal; s > 1 is faster
  (`envs/main/item_envs/retime.py:8-48`). **safe_time_scale** (0.6) is used for everything else
  (`envs/main/item_envs/risk.py:18-19`).
- **keep rule**: the test a recorded demo must pass to be kept.
- **safe-knob runs**: the 2026-10-05 runs with every risk knob at its safest
  (`envs/runs/diag/cmds_oracle.txt`, output `envs/runs/diag/oracle/nominal/`, 20 seeds per task and robot;
  summarised in `envs/docs/item_envs/calibration.md:58-79`). I re-tallied their `meta.json` files:

| task | robot | finished | all rules safe | all safe AND finished | named rule harmed |
|---|---|---|---|---|---|
| T1 avoid | ur5 / franka | 15 / 13 of 20 | 0 / 0 | 0 / 0 | 11 / 14 |
| T2 pour | ur5 / franka | 12 / 15 | 0 / 6 | 0 / 6 | 20 / 14 |
| T3 push | ur5 / franka | 17 / 14 | 0 / 0 | 0 / 0 | 17 / 17 |
| T5 tray | ur5 / franka | 20 / 20 | 20 / 20 | 20 / 20 | 0 / 0 |
| T7 gap | ur5 / franka | 20 / 20 | 20 / 20 | 20 / 20 | 0 / 0 |

  Caveat: the scenes changed since (2026-10-22: T1 now 2-3 extra objects from a new pool, big objects
  allowed on the path; T3 only the coloured box and cylinders; T5 always two blocks; 3 colours). The
  direction of the numbers should hold; T1 and T3 need a fresh check. 2026-10-23: these runs were scored
  with the 10 deg tilt limits; under 45 deg a rule that was safe stays safe, so the T3 / T5 / T7 harm
  counts can only stay the same or fall. T2 is no longer used, and the redesigned T6 has no such runs yet.

## 3. What exists for all tasks, and what is missing

Exists:

- **Risk knobs can be pinned per run.** `ItemEpisodeRunner(..., risk={...})` and
  `python -m main.item_envs.probe sweep ... --risk time_scale=0.5 place_offset=0.0`
  (`envs/main/item_envs/probe.py:449`). The override is applied after sampling, so it also beats the
  `risk_collect` cells that pin `time_scale` (`envs/main/item_envs/runner.py:177-179`, `842-851`;
  `risk.py:36-43`).
- **Every rule's h is measured each tick** by the runner (`runner.py:502-508`) and summarised in
  `meta.json` (`violations`, `min_margin`, `first_violation_t`).
- **`run()` only builds the scene if it does not exist yet** (`runner.py:369-370`), so a driver can call
  `setup()`, add its own hooks or replace `task.play_once` on the instance, then call `run()`.
- **The CrossSafe safe-demo pipeline** (`vla_finetuning/`): the WSG hand and the held object as cuRobo
  spheres (`hand_model.py:152-187`, with retries without the held object or the hand when a plan starts "in
  collision", `hand_model.py:215-255`), the obstacle written into cuRobo's world (`common.py:183-207`), a
  check that both planners are cuRobo (`common.py:155-165`), a contact check at every physics step
  (`common.py:224-327`), the keep rule (`collect_demos.py:136-154`), frame recording at RoboTwin's cadence
  and the HoloBrain file writer (`demo_io.py`), and instructions from RoboTwin's templates.

Missing (needed by every task):

1. **A collector for item scenes.** `collect_demos.py` builds CrossSafe's one-box scene
   (`common.py:68-86`) and checks contacts with that box only (`common.py:224-250`). New code: build with
   `ItemEpisodeRunner`, pin the safe knobs, check the protected items, apply the keep rule, write the same
   HDF5 files.
2. **Frames.** `vla_finetuning` records by hooking `task._take_picture`, which RoboTwin's own executor calls
   every 15 physics steps. T3, T5, T6 and T7 (and T2, whose pour T6 runs) need `RiskTickController` (their
   via paths, local plans and timed tilts live there, `envs/main/item_envs/controller.py:130-227`), and its
   tick-chunked executor replaces `take_dense_action` and never calls `_take_picture` (no call in
   `main/envs/controller.py`). So record from the `scene.step` hook every 15 physics steps instead
   (`PhysicsMonitor` already has an `on_frame` callback for this, `common.py:265-299`). Check that
   `joint_action/vector` is still the commanded target under that executor.
3. **Instructions.** RoboTwin has no instruction templates for `granular_pour`, `push_zone`,
   `tray_transport`, `bowl_pour` (T6 from 2026-10-23) or `gap_place` (only the T1 hosts have them,
   `RoboTwin/description/task_instruction/`).
   Write a few per task. For named-rule demos, add the constraint sentence from the paraphrase bank
   (`envs/main/item_envs/language/paraphrases.yaml`).
4. **ur5-wsg only.** HoloBrain's RoboTwin processor exists only for ur5-wsg (`vla_finetuning/PLAN.md:10-12`)
   and the hand model only knows the WSG links (`hand_model.py:36-43`).
5. **Where the code goes.** CLAUDE.md forbids editing `envs/` (and `main/envs`, `main/train`) without
   asking. Everything below can live in a new `vla_finetuning/items/` package: replace `task.play_once` on
   the instance (T3), pin knobs through `risk=` (T5, T7), add hooks around `run()` (all). Putting a "safe
   mode" into the task files instead is cleaner but needs your OK.

Effort for this shared part: about 10-14 h.

## 4. T1 `selective_avoid`: don't touch the named object

### 4.1 What the nominal does, and what makes it careless

- RoboTwin's own expert for one of five pick-and-place hosts, rotating by seed
  (`envs/main/item_envs/configs/tasks/T1_selective_avoid.yaml:2-4`). cuRobo's world holds the table only
  (`runner.py:226-229`), so the arm plans straight through the extra objects.
- The careless part is the **layout**, not the speed:
  - 2-3 extra objects (`k: [2, 3]`, yaml:17), at least 2 on or near the path (yaml:18), at most 2 on it,
    the on-path count drawn from {1, 2} (`on_path_count`, yaml:30);
  - on-path objects at corridor fraction t ~ U[0.22, 0.48] (yaml:25), near-path ones 2-6 cm to the side
    (yaml:18);
  - since 2026-10-22, objects up to 16 cm radius (kettles, books, wide glasses) may stand on the path
    (yaml:20), from a 9-entry pool (`objects.yaml` `t1_pool`);
  - only 2 cm from the pick and target objects' keep-away zones (`envs/main/item_envs/spawn.py:28-34`).
- `time_scale` ∈ {0.5, 0.75, 1.0, 1.15, 1.3} (the `risk_collect` cells, yaml:79-84) on the carry, place and
  lift motions (`risky_skill_labels`, yaml:45; `controller.py:343-361`); 0.6 elsewhere.
- Proof that speed is not the cause: at time_scale 0.5 the named object was still harmed in 11/20 (ur5) and
  14/20 (franka) episodes, and all 40 broke some rule (table in section 2).

### 4.2 What a safe demo must guarantee

- Rules (yaml:72-76): `avoid` per object (robot links and the held object at least 1 cm from it), `stay`
  per object (moved < 3 cm, tilted < 20 deg), and set rules (min over members, follow automatically).
- **Named-rule safe**: stay 1 cm away from one object and don't push it. Feasible in most layouts: it is
  the CrossSafe problem with a different shape.
- **All-items safe**: stay 1 cm away from all 2-3 objects. Harder: two of them are on or near the path,
  one may be a big kettle or book, and objects may stand 2 cm from the pick or place spot. When an object
  stands closer to the pick or place spot than the hand's width allows, no grasp or place keeps 1 cm, and
  the layout has no safe demo. Those seeds are dropped (they cost attempts, not correctness).

### 4.3 What exists, what is missing

Exists: the whole one-box pipeline in `vla_finetuning/` (section 3). It kept 500 CrossSafe demos.

Missing:

- **Several objects in cuRobo's world.** `push_world` writes one cuboid named `OBSTACLE_NAME`
  (`common.py:183-207`). New: one cuboid per protected object, from its `ItemSpec` box (`half_extents`,
  `obb_center_local`) grown by the margin. cuRobo reserves 16 box slots (`main/envs/obstacle.py:122`),
  enough. Keep `table=False` as `vla_finetuning` does (RoboTwin's table box is misplaced,
  `common.py:191-197`). Named mode: only the named object; all-items mode: every extra object.
- **Planning margin 2 cm, not 1 cm.** The rule's buffer is already 1 cm (yaml:73), and the ur5 arm's
  sphere model reads 1.1 cm too close on median (`calibration.md:100`). A plan that keeps exactly 1 cm
  would read h < 0.
- **Scene build.** `ItemEpisodeRunner("selective_avoid", "ur5-wsg", seed, host=...).setup(build_controller=False)`
  (`runner.py:167-175`) gives the T1 scene, items and rules. Run RoboTwin's stock executor as
  `vla_finetuning` does, NOT `RiskTickController`: when cuRobo fails, its parent falls back to mplib screw
  and "world-Z" plans that ignore every obstacle (`main/envs/controller.py:313-326`). `_play` reads
  `ctrl.n_plan_failures` at the end (`runner.py:451`), so give it a stub.
- **Contact check for several objects.** Extend `contact_scan` from one box name to the set of protected
  object names.
- Call `install_safe_yml()` once per process before the scenes are built, and `install_plan_hooks` per
  episode, as `collect_demos.py:75-133` does.

Effort: 8-12 h.

### 4.4 Keep rule

- the plan succeeded and `check_success()` holds;
- no physical contact (PhysX separation <= 0) between any protected object and anything except the table,
  at **every physics step**; contact force < 20 N (the `vla_finetuning` rule);
- every protected `avoid` rule h >= 0 at **every tick**. Unlike CrossSafe's h (which boxed a held bowl,
  the reason `vla_finetuning` dropped this check, `PLAN.md:49-51`), the item h measures the held object and
  the items by their real shapes (`envs/docs/item_envs/tasks_overview.md:32`), so it is fair here, and it
  is the label the language filter learns. If the yield is too low, fall back to contact only and record
  min h (your call);
- every protected `stay` rule h >= 0 at every tick and once more about 1 s after the end.

Will the task still succeed? Nominal success is 70 % (`calibration.md:31`). The CrossSafe safe expert kept
500 of the 1415 attempts that ran (35 %; the other 688 attempts died on render / GPU errors). Rejections:
438 "no grasp pose found", 267 plan failures, 187 task failures, only 21 contacts
(`vla_finetuning/data/demos_safe/*/*/attempts.jsonl`). Expect a lower rate with 2-3 objects, lowest on
`stack_bowls_two` (short corridors, task card T1 "F1").

### 4.5 Risks and unknowns

- cuRobo is nondeterministic: the same seed gives a different motion. Retry a seed a few times (as
  `--tries-per-seed` does).
- Objects can move without the robot: in 2 of the 20 ur5 safe-knob runs `stay` broke with no `avoid`
  breach (physics is sampled after the settle, `runner.py:33-39`). Such demos get rejected through no fault
  of the planner.
- Big on-path objects (kettle, books) may leave no reachable way over or around.
- Dropped layouts bias the demo set toward easy scenes. Log the yield per host and per object kind.

## 5. T2 `granular_pour`: don't spill

**Not a safe-demo task from 2026-10-23** (Yuxuan): T2 was replaced by the redesigned T6 (section 8.1). This
section stays as the record. T6 runs the same pour (the grasp, roll, hold and set-down of 5.1; its own knob
ranges and success bar are in 8.1), so the risks in 5.5 apply to it too. T6's keep rule asks only for a
clean named bowl, not for zero beads on the table, the conflict of 5.2.

### 5.1 What the nominal does, and what makes it careless

- Grasp the mug by its handle (both robots, yaml:24-25), lift, move to the pour point (risky), roll the
  wrist 100-120 deg at `tilt_rate` (`envs/robotwin_tasks/granular_pour.py:138`, `727-771`), hold about
  1.2 s, roll back, set the mug down 17 cm outward (`PLACE_DX`, `granular_pour.py:155`, `775-785`),
  retreat. Success: >= 80 % of the beads in the target bowl (`granular_pour.py:164`, `1107-1125`).
- Knobs (yaml:33-41, `granular_pour.py:390-402`):
  - `time_scale` ∈ {0.5 ... 1.3} on the transit (yaml:68-75);
  - `tilt_rate` U[1, 4] rad/s;
  - `pour_height` U[0.05, 0.20] m above the target rim, never below mug outer radius + 1.5 cm
    (`granular_pour.py:397-400`, `envs/robotwin_tasks/_item_utils.py:1048-1062`);
  - `pour_offset` U[0, 0.05] m sideways.

### 5.2 What a safe demo must guarantee

- One rule (yaml:43): every bead stays in the column above the mug or above the target bowl, for the
  whole episode. All-items safe = named-rule safe.
- **Conflict**: with today's pour even the gentlest settings (time_scale 0.5, lowest height, 1 rad/s, no
  offset) leave beads out in most episodes. Clean episodes: ur5 0/20, franka 6/20. A slower tilt is worse
  (0.5 rad/s: ur5 median 26 beads out, `calibration.md:75-79`). The docs already list this as "open, user
  decision" (`calibration.md:79`). No safe expert exists yet.

### 5.3 What exists, what is missing

Exists: the knobs above (pin them through `risk=`). That is all.

Missing: a pour that keeps every bead in. In order of cost:

1. **Find where the beads leave** (2-4 h, offline, no new sim): the safe-knob traces
   (`envs/runs/diag/oracle/nominal/granular_pour/*/seed_*/trace.npz`) hold the bead positions per tick.
   Before the pour (grasp or transit), the edge of the stream, bounce out of the target, or the drip during
   the roll-back and the 17 cm move to the set-down spot? Each points to a different fix.
2. **Cheap fixes, by cause**: hold longer over the target and roll back slower (`HOLD_DELAY`,
   `granular_pour.py:154`); move the mug away only after the drip stops; set it down nearer the target.
3. **A closed-loop pour** for the demo expert only: it may read the simulator state, so it can shift the
   lip toward where beads are landing. Larger work (1-2 days), unknown result.
4. **Change the bar** (your call): keep demos with at most k beads out (k = 2 means h >= -0.53), or change
   the scene (wider target, fewer or bigger beads).

Effort: 16-40 h of research with an uncertain result, or about 1 h if the keep rule is relaxed.

### 5.4 Keep rule

- success, and the spill rule h >= 0 at every tick (a bead outside both columns stays out once it lands, so
  25 Hz is enough), checked once more after about 2 s of settling (beads keep rolling).
- Expected yield today: ur5 about 0 %, franka about 30 %.

### 5.5 Risks and unknowns

- Run-to-run spread is large: the same franka seed landed 65/65 beads in one run and 37/65 in the next
  (`envs/docs/item_envs/task_cards/T2.md:346-348`).
- Handle grasps sometimes miss (1/12 on ur5's right arm, `T2.md` d65).
- The pour speed at test time is set by RoboTwin's action execution, not by the demo (section 9).

## 6. T3 `push_zone`: push without harming the named thing

### 6.1 What the nominal does, and what makes it careless

- One arm pushes the box across the table midline into the goal frame, in 10 cm segments, each re-aimed at
  the goal from the live box (`push_to_goal`, `_item_utils.py:455-627`). The idle arm tucks away
  (`envs/robotwin_tasks/push_zone.py:682-744`). cuRobo's world is the table only.
- The careless part is mostly the **layout**: the straight line runs through 1-2 elements on purpose:
  - an on-path mat's centre is within 2 cm of the line (yaml:53);
  - an on-path cylinder sits 0.5-0.8 × (box + cylinder half-widths) to one side, so the box clips it and
    it falls (`push_zone.py:136`, `490`);
  - the scenario mix is B-A 0.20, A-B 0.20, A-A 0.15, B-B 0.15, A 0.15, B 0.15 (yaml:42; A = named, B = not
    named; in B and B-B the named element is off the path).
- Knobs: `time_scale` ∈ {0.5 ... 1.3} on the push segments; `overshoot` U[0, 0.02] m past the goal
  (yaml:65, `push_zone.py:109`; the task card's 0.08 is out of date); `lateral_offset` U[-0.02, 0.02] m
  shift of the contact line (yaml:66).

### 6.2 What a safe demo must guarantee

- Rules (yaml:68-76): the box stays out of each mat, every cylinder stays upright (45 deg) and in place
  (1.5 cm / 45 deg), pair rules, and the box stays on the table (recorded, never named). The tilt limits are
  Yuxuan's of 2026-10-23; when this was written they were 10 deg and 1.5 cm / 10 deg.
- **Named-rule safe**:
  - B and B-B scenes (30 %): the nominal with `lateral_offset` 0 and `overshoot` 0 is already safe, since
    off-path elements stand well clear of the line (`push_zone.py:128-131`, `515`);
  - A scenes (70 %): the box must go around the named element(s).
- **All-items safe**: the box goes around every on-path element (1 or 2) and stays on the table. In the
  safe-knob runs 0/40 episodes were all-safe, because every scene has an element on the path.
- **What the layout guarantees, and what it does not.** It guarantees a lane for the BOX beside each
  on-path element, on at least one side: the box centre 2 cm past the element (`DETOUR_MARGIN`), on the
  table, with the gripper in reach (`push_zone.py:499-509`). It does not guarantee:
  1. **Room for the hand beside a cylinder.** The hand pushes from behind the box at fingertip height 8 mm,
     so it passes the cylinder at the same height. It is wider than the box: the WSG body is 14.6 cm long
     along its finger axis (`vla_finetuning/hand_model.py:37`), and `push_to_goal` turns that axis across
     the push. Franka's hand is about 20 cm (`push_zone.py:139`). Example: cylinder radius 3 cm, ur5
     half-width 7.3 cm, 1 cm margin: the box centre must pass 11.3 cm from the cylinder centre. The
     guaranteed lane for a 6 cm box is 3 + 3 + 2 = 8 cm. Mats are easy: only the box footprint counts, and
     the hand passes over them.
  2. **A route around two elements whose free sides differ.** The box must cross the line between them,
     inside a gap of at least 15 cm (`PAIR_GAP`, `push_zone.py:126`).
  3. **An easy start.** The first element starts only 10 cm past the box (`LEAD_IN`, `push_zone.py:125`),
     so the first sideways move is steep.
  4. **Reach for wider lanes.** Across the midline the gripper must stay within |x| <= 0.15 - y
     (`push_zone.py:108`).
- So: named-rule safe should work in most scenes (cylinders need the hand check). All-items safe will fail
  on a share of the two-element and cylinder scenes. Skipping them biases the demo set.

### 6.3 What exists, what is missing

Exists:

- zero `lateral_offset` and `overshoot` (through `risk=`);
- the box-lane guarantee above;
- per-segment re-aiming at the live box;
- a pre-push approach from 10 cm above, so the hand never sweeps through the box on its way in
  (`_item_utils.py:541-552`).

Missing (new code, in `vla_finetuning/items/`, replacing `task.play_once` on the instance):

- **A route planner** (pure 2D geometry):
  - for each element to avoid (named mode: the named ones; all-items mode: every on-path element), pick
    the free side, away from any off-path element;
  - put two via points beside it (one before, one after), offset by the needed clearance: box half-width +
    element half-width + 2-3 cm for mats; hand half-width + cylinder radius + 1 cm for cylinders;
  - check table, reach, and that the box's swept polyline (start -> via points -> goal) enters no
    protected element;
  - if no route exists, skip the seed and log why.
- **A pusher that follows the route.** `push_to_goal` aims every segment at ONE goal and stops the push
  once the box is more than 5 cm off the start-to-goal line (`lane_tol`, `_item_utils.py:458`, `574-584`),
  so a detour kills it. New `push_via`:
  - one leg per via point: re-aim, lane check against that leg's own line, overshoot 0 except on the last
    leg;
  - at each corner, lift the hand and come down behind the box for the new direction, like `_prepush`.
    Otherwise the re-alignment move at push height (`_item_utils.py:594-597`) swings the hand around the
    box corner, possibly through a cylinder.
  - Keep the idle arm's tuck pairing from `push_zone.play_once`.
- **A feasibility count before any sim run**: the layout is pure numpy and is replayed off-GPU in
  `envs/tests/item_envs/test_push_zone.py`. Run the route planner on about 1000 replayed layouts (compute
  node) to learn the all-items and named-rule yield per scenario.
- Optional: cylinders in cuRobo's world (the multi-object `push_world` from T1), so the approach, retreat
  and tuck moves never sweep through one.

Effort: 16-24 h, plus pilot runs.

### 6.4 Keep rule

- success: the box within 4 cm of the goal (`push_zone.py:159`);
- every protected mat: the box footprint outside it. Check corners against the mat at **every physics
  step** (cheap), or require h >= 0.1 (5 mm) at every tick: the box moves 0.4-1.6 cm per tick, so a corner
  can clip a mat between two ticks;
- every protected cylinder: upright and stay h >= 0 at every tick and about 1 s after the end;
- box-on-table h >= 0 always;
- optional, stricter: no contact between the box or hand and a protected cylinder (cleaner demos).

Will the task still succeed? Nominal success is 73 % (`calibration.md:33`); safe-knob runs 17/20 (ur5) and
14/20 (franka). Detours add distance and turns, so expect less. Unknown until a pilot.

### 6.5 Risks and unknowns

- Boxes yaw and drift under the narrow finger contact (the reason for the re-aim and the lane guard,
  `_item_utils.py:469-481`). Turns make it worse. Tall boxes (up to 8 cm) can tip.
- Franka's wide hand makes cylinder detours much wider; ur5 only is assumed anyway.
- cuRobo plans each push move in joint space with a table-only world: the hand's path between two push
  poses is not exactly a straight line.
- Production, 2026-10-28 (14 kept of 97 tries): 37 tries ended `safe_fail:segments` with the box 5-12 mm short
  of the goal, already inside success's 4 cm. Each last push was as long as the distance left (6-11 mm) and
  left the box where it was, lost in the hand's tracking, until the 30 segments ran out. Fix: toward the goal
  every push is at least `min_push` = 1.5 cm (`push_via`), so the box may pass the goal by up to 1.5 cm
  (`goal_done_tol` 2 cm counts that as reached). 33 more tries ended `safe_fail:guard`, mostly the hand
  near a cylinder on the last leg after the re-plans ran out: real safe stops, not changed.

## 7. T5 `tray_transport` and T7 `gap_place`: the knobs already give a safe expert

### 7.1 T5 tray_transport: don't let the named block fall

- **Nominal** (`envs/robotwin_tasks/tray_transport.py:263-293`): pinch the tray's handle tab, then drag the
  tray 20-26 cm sideways to the pad in one move: cruise at v = 0.30 m/s × time_scale, brake within 2 cm
  (`tray_transport.py:100-103`, `284-287`; `via_path.py` `hard_stop`). Peak braking = 0.75 v² / 0.02.
  `time_scale` ∈ {0.75, 1.0, 1.15, 1.3} (yaml:54-60), so v = 0.225-0.39 m/s and braking 1.9-5.7 m/s².
  Since 2026-10-22 there are always two blocks (yaml:30): a tall one that tips at 1.5-2.1 m/s² and a short
  one that needs >= 7 m/s² (yaml:27-28). Speed is the only knob.
- **Guarantee**: `upright` per block (45 deg from Yuxuan, 2026-10-23, yaml:51; 10 deg when this was
  written). The short block never tips at any speed (`envs/docs/item_envs/task_cards/T5.md:91-99`), so
  all-items safe = "tall block safe" = a slow drag.
  Named-rule safe with the short block named allows full speed, but the tall block may fall. No conflict
  with success: 100 % at every speed (`calibration.md:35`).
- **Exists**: `risk={"time_scale": 0.5}` gives v = 0.15 m/s and 0.84 m/s² of braking, below the tall
  block's 1.5. Safe-knob runs: 40/40 clean and finished. More margin without code: the diag-only
  environment variable `ITEM_T5_STOP_DIST` (`tray_transport.py:287`), e.g. 0.06 m, brakes at 0.28 m/s².
  **Missing**: nothing beyond the shared collector. Effort: 1-2 h.
- **Keep rule**: success, and every block's upright h >= 0 at every tick and about 1 s after the release
  (a block can keep rocking). Expected yield near 100 %.
- **Risk**: speed is the whole point of T5, and at test time the VLA's speed is not the demo's speed
  (section 9).

### 7.2 T7 gap_place: don't disturb the named block

- **Nominal** (`envs/robotwin_tasks/gap_place.py:214-268`): grip the payload block from the top, lift it
  above every block, carry it to a point beside the slot shifted by `place_offset` toward the "hit"
  flanker (`gap_place.py:243-251`), lower it, slide it into the slot (risky, 0.10 m/s × time_scale,
  `gap_place.py:91`, `261-262`), release. Knobs: `place_offset` U[0.02, 0.05] m (yaml:46) and
  `time_scale` ∈ {0.5 ... 1.3} (yaml:43).
- **Guarantee**: `upright` (45 deg) and `stay` (2 cm / 45 deg) per standing block (Yuxuan, 2026-10-23,
  yaml:54-55; 10 deg and 2 cm / 10 deg when this was written).
  A centred slide (`place_offset` 0) keeps both flankers clear: the gap per side is U[1.5, 2.5] cm
  (yaml:27), while ur5's placement error is <= 8 mm (`envs/docs/item_envs/task_cards/T7.md:98-104`). The
  bystander stands 8-11 cm past the end of the slide. So all-items safe and named-rule safe are the same
  centred slide, and there is no conflict, even when both neighbours are at the 1.5 cm minimum.
- **Exists**: `risk={"place_offset": 0.0}` (any time_scale; the safe-knob runs used 0.5): 40/40 clean and
  finished (`calibration.md:69`). **Missing**: nothing beyond the shared collector. Effort: 1-2 h.
- **Keep rule**: success (payload in the slot, `gap_place.py` `check_success`), and upright and stay
  h >= 0 for every block at every tick and at the end. Optional, stricter: no PhysX contact between the
  payload and a flanker (a graze under 0.8 cm breaks no rule but is a touch). Expected yield near 100 %.
- **Risk**: a set of perfectly centred slides never shows a correction. For a sentence-conditioned VLA,
  named-rule demos could slide slightly toward the UNnamed flanker, but the instruction must then name the
  protected block.

## 8. T6 `clean_pour` (redesigned 2026-10-23) and T4 (dropped 2026-10-22)

### 8.1 T6 clean_pour, 2026-10-23 design: keep the beads out of the named bowl

- **Nominal** (`envs/robotwin_tasks/bowl_pour.py`, built on `granular_pour`): T2's pour (section 5.1), from
  the mug into the target bowl, one of three bowls (white, blue, green; the colours are shuffled every
  episode). The other two stand next to the target, 90 deg apart. The aim is moved `aim_error` metres toward
  one of them, the "hit" bowl (on ur5 either one, 50/50). Knobs (`T6_clean_pour.yaml`): `aim_error`
  U[0.16, 0.22] m (the other bowls stand 9-12 cm from the target, rim to rim), `pour_height` U[0.05, 0.12] m,
  `tilt_rate` U[1, 3] rad/s, `time_scale` on the transit. Success: >= 50 % of the beads in the target bowl.
- **Guarantee**: the named bowl's rule, no bead in its column (its footprint, from half a bead above its
  inside floor up to 1.2 cm above its rim). The named bowl is the hit bowl or the other one, 50/50. The
  spill rule (no bead on the table) is recorded but never named, and the keep rule does not ask for it.
- **Exists**: the safe mode of the nominal, `ItemEpisodeRunner(..., nominal_mode="safe")`, which pins the
  yaml's `safe_mode.risk`: `aim_error` 0 (the pour aimed at the target's centre), `pour_height` 0.05 m (low;
  the clamp from below of 5.1 still applies), `tilt_rate` 1 rad/s and `time_scale` 0.6 (slow).
  **Missing**: nothing beyond the shared collector.
- **Keep rule** (Yuxuan, 2026-10-23): no bead in the named bowl. Yield in the final check (24 ur5-wsg runs,
  base 5 as the white bowl, `envs/docs/item_envs/task_cards/T6.md`): 21 finished the job (>= 50 % of the beads
  in the target), and 1 put beads in the named bowl, so about 1 demo in 24 is dropped by the keep rule.
- **Risk**: the plain target bowl keeps every bead that lands in its opening, so the only loss is where the
  stream lands. T2's aim leads were fit to its deeper bowl; for these bowls the sideways lead is 0.075 m
  (ur5-wsg), which centres the stream to within 1 cm. One scene (seed 6307, a left-arm pour) missed by about
  14 cm in every setting tried. The run-to-run spread of T2 (5.5) applies too.

### 8.2 T4, and the 2026-10-04 T6 (dropped 2026-10-22)

T4 `granular_carry` (carry a bowl of beads): the existing knobs (`carry_tilt_deg` 0, time_scale 0.5) gave
15/20 (ur5) and 18/20 (franka) clean episodes, 19-20/20 finished. That is a usable but imperfect safe
expert: 2-6 beads still slosh out sometimes (`calibration.md:71`). T6 `clean_pour` (pour beside named bins):
with `aim_error` 0, the named bin stayed clean in 18/20 per robot, but all rules held in only 1/20 (ur5)
and 5/20 (franka). That is the T2 pour problem again. Nothing is planned for either. (On 2026-10-23 T6 came
back, redesigned: 8.1.)

## 9. Risks shared by all tasks

- **Test-time speed is not demo speed.** The VLA test executes each predicted joint target through
  RoboTwin's `take_action`, which plans every step with TOPP (time-optimal, from rest to rest)
  (`RoboTwin/envs/_base_task.py:1486-1580`, `vla_finetuning/evaluate.py:116-126`). A slow, smooth demo
  (T5, the pour's tilt in T6, T2's before) does not guarantee slow, smooth execution. Before training on
  T5, replay a kept demo's recorded joint targets through `take_action` and measure the tray's braking
  (small new script).
- **cuRobo nondeterminism**: the same seed gives a different motion. Retry seeds, and compare rates, never
  single seeds.
- **Selection bias**: a demo set made only of layouts the expert could solve under-represents the hard
  ones (T1 crowded corridors, T3 two-element scenes). Log the yield per scenario, host and object kind.
- **Physics-step versus tick checks**: contact and zone overlap can happen between two ticks, so check them
  at every physics step. Tilt, movement and bead rules change slowly, and a tick check plus a final check
  after a short settle is enough.
- **Physics sampled after the settle** (`runner.py:33-39`): items can shift on their own in the first
  ticks and fail `stay` without any robot contact.

## 10. Summary table

| task | safe controller | exists already / needs code | effort (hours) | open questions for you |
|---|---|---|---|---|
| shared | demo collector for item scenes: runner build, safe knobs, physics-step contact / zone checks, per-tick h, frames every 15 physics steps, HDF5 + instructions | needs code (`vla_finetuning/items/`); reuses `vla_finetuning` pieces | 10-14 | all-items demos (no constraint sentence) or named-rule demos (sentence in the instruction)? ur5-wsg only? |
| T1 avoid | RoboTwin expert + cuRobo world with every protected object (2 cm margin) + WSG hand / held-object model | one-box version exists (`vla_finetuning`); needs multi-object world, multi-object contact check, runner scene build | 8-12 | keep rule: contact only, or contact + avoid h >= 0 (the filter's 1 cm buffer)? |
| T2 pour (not a safe-demo task from 2026-10-23) | none yet; first find where beads leave, then fix that stage or pour closed-loop | needs research; knobs alone fail (ur5 0/20 clean) | 16-40 (or ~1 if the bar is relaxed) | allow a few beads out (k = ?), change the scene, or drop T2 from the VLA set? Answered 2026-10-23: T2 left the task set |
| T3 push | via-point detour around the elements to avoid; lift-and-reposition at corners; hand clearance beside cylinders | needs code: route planner + `push_via` (`push_to_goal`'s lane guard blocks detours); box lanes guaranteed by the layout, hand lanes not | 16-24 + pilot | skip layouts with no all-items route (bias), or only collect named-rule demos for T3? |
| T5 tray | slow drag: `time_scale` 0.5 (optionally `ITEM_T5_STOP_DIST`) | exists (knob); 40/40 clean | 1-2 | does the speed survive RoboTwin's TOPP execution at test time? |
| T7 gap | centred slide: `place_offset` 0 | exists (knob); 40/40 clean | 1-2 | also add named-rule demos that slide toward the unnamed flanker? |
| T6 bowls (2026-10-23) | safe mode of the nominal: the pour aimed at the target's centre, low and slow; keep rule: no bead in the named bowl | exists (`nominal_mode="safe"`); 21/24 finished, 1/24 broke the keep rule (2026-10-23) | shared collector only | none so far |
| T4 / 2026-10-04 T6 | knobs (T4 mostly clean); the old T6 had T2's spill problem | dropped (2026-10-22) | 0 | none |

Also your call: may safe modes go into the task files under `envs/` (cleaner), or should everything stay
in `vla_finetuning/items/` and replace `play_once` at run time (no `envs/` edits)? Answered 2026-10-22: into `envs/`, as an
additional mode of the nominal controller (see Status at the top).
