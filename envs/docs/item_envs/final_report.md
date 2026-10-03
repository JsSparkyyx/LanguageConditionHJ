# Item-safety environments — final report (PLAN v3 §10)

Status date: 2026-10-02. Cluster: NCSA Delta, A40 (`gpuA40x4-interactive`),
apptainer `robot-sim_cu128.sif`, sapien 3.0.0b1. All numbers below were
measured on this hardware; cuRobo planning is nondeterministic run-to-run
(regression/README.md), so absolute cm-level numbers do not transfer between
machines.

<!-- CALIBRATION: the per-task calibration tables are filled from
     docs/item_envs/calibration.md once the sweeps finish. -->

## 1. Build status

| task | ur5-wsg | franka-panda | notes |
|---|---|---|---|
| T1 `selective_avoid` | built | built | 5 CrossSafe hosts, round-robin by seed |
| T2 `granular_pour` | built | built | 100 beads, lip-pivot pour, direction by dry run |
| T3 `push_zone` | built | built | one cross-midline push per episode (user decision) |

Phase commits: 0 recon (3de7a96) · 1 numpy core (212b532) · 2 sim infra
(bef4161) · 3 feasibility (2c2820a … f18faef) · 4 tasks (14c6180 → 32ae062)
· 5 CEHJ integration (2c0f272, a4802a6, 58f83c1).

## 2. Probe gates (feasibility.md)

| probe | verdict |
|---|---|
| P1 multi-object path crossing (T1) | PASS after Fix 1 (near2 63.8 %, touch 62.8 %, baseline 94.3 %) |
| P2 bowl containment + beads (T2) | rest containment PASS, zero tunneling PASS, ≤ 20 ms/tick PASS; translate-spill clause not achievable (F2); tilt saturates at 110° (F3) |
| P3 push coast (T3) | PASS |
| P3-robot push primitive | PASS (ur5 1.00, franka 0.95 at time_scale 0.5) |
| P4 visibility | T2 PASS (bowl_src 0.95, borderline); T1 ACCEPTED below gate by the human (on 0.948, near 0.947, off 0.995) |
| P5 token alignment | PASS (overlap 0.5, borderline) |

Post-gate finding F6: every primitive cylinder (T1 candidates, T3 bystanders,
the visibility-band probe) had been spawned lying down — SAPIEN cylinders run
along local x. Fixed 2026-10-02; the F5 band was measured with the lying
probe (see feasibility.md F6).

## 3. Deviations from the plan (with reasons)

1. **T3: one push per episode, not two simultaneous pushes** (user decision
   2026-10-02). Two simultaneous cross-table pushes collide mid-table (diag
   job 22636048: both EEs shoved sideways) and exceed franka's reach. The
   object spawns on a random side; the idle arm closes and tucks.
2. **T3 cross-table push** (user request): goals in the opposite x band,
   midway zone near the corridor midpoint; lane kept near the robot by the
   shared reach band |x| ≤ 0.15 − y (workspace_*.npz).
3. **T3 closed-loop push** (`push_to_goal`): 10 cm segments re-aimed at the
   live box; the plan's single open-loop displacement walked the box 3–5 cm
   off a 0.2–0.5 m push.
4. **T2 N = 100 beads** (user request; P2 validated N = 60).
5. **T2 pour**: the tilt pivots about the pouring lip, pinned on the pour
   point; the pour direction (far lip, −y, +y) is the first whose full tilted
   pose passes a screw-planning dry run (ur5-wsg wrist 1 is limited to
   ±1.75 rad, which blocks the far-lip tilt); tilt steps are planned as local
   screws (cuRobo goal IK flipped branches mid-tilt). The pour point is raised
   until the swept bowl clears the target rim by 3 cm, so the effective pour
   height can exceed the `pour_height` knob (which acts as a lower bound).
6. **T2 static target is a region, not an item** (physics sampling cannot set
   mass on static bodies).
7. **T1 keepaway gap 2 cm** (CEHJ parity) instead of 3 cm; on-path placement
   is otherwise infeasible on the short corridors (F1).
8. **`stay` margin units**: `min(d_tol − displacement [m], θ_tol − tilt
   [rad])` — the binding term sets the unit (kept as the plan defines it,
   user decision 2026-10-02); downstream κ = 20/m scales both terms.
9. **Held detection**: an item is held only when pinched (≥ 2 gripper links
   in contact); "first contact wins" latched bystanders brushing a finger.
10. **Regression contract**: byte-identical h traces are unachievable
    (cuRobo nondeterminism even on one node); the Phase-5 regression asserts
    a statistical contract instead (regression/README.md).

## 4. Calibration (PLAN §8)

See `calibration.md` (per task × embodiment: violation rate per rule,
non-specified violation rate, nominal success, wall time per episode,
physics ms/tick) and the `risk_collect` blocks in the task yamls.

## 5. Known issues

- T1 `stack_bowls_two` remains the hardest host (22–28 % layout feasibility).
- T1 `place_bread_basket` / franka nominal success with candidates is ~50 %
  (dynamic candidates on the path get pushed into the host items).
- P4/P5 pass at their thresholds exactly (borderline).
- Video episodes take 3–10 min each with ray tracing; calibration runs
  no-render.
