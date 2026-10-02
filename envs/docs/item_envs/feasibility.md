# Feasibility probes — results and gate evaluation (plan v3 §6 Phase 3, STOP gate)

Date: 2026-10-02. Status: **all probes have sweep data. P1 gate FAILED on the first
sweep (layout infeasibility, not geometry) → documented Fix 1 applied to
`spawn.py`, rerun submitted (array 22611061). P3-robot PASS. P4(T1) FAIL (finding
F5).** Outputs under `runs/item_envs/feas/` (rows JSONL + `summary.json` +
`keyframes/`; the pre-Fix-1 P1 sweep is archived at `runs/item_envs/feas/P1_pre_fix1/`);
run logs under `/work/nvme/bifq/project/tmp/logs/`.

Gate summary:

| probe | gate | verdict |
|---|---|---|
| P1 multi-object path crossing (T1) | near ≤ 2 cm of ≥ 2 candidates in ≥ 50%; touch ≥ 1 in ≥ 30%; no-object success ≥ 70% | **FAIL on sweep 1** (near2 13.7%, touch 15.6%, baseline 94.1% PASS) — dominated by 18% layout feasibility; on the feasible subset the gate passes (near2 73%). Fix 1 applied, **rerun in flight** |
| P2 bowl containment + beads (T2) | rest inside: PASS (default cell) / FAIL (r=8 overflow cells, expected) · zero tunneling: **PASS** · spill ↑ with accel: **FAIL (finding F2)** · spill ↑ with ω: **FAIL as stated (finding F3)** · physics ≤ 20 ms/tick @ N=60: **PASS** | partial, findings below |
| P3 push coast (T3) | coasting > 0, ↑ with v, ↓ with friction | **PASS** |
| P3-robot push primitive | ≤ 3 cm of commanded in ≥ 80% of seeds at time_scale 0.5 | **PASS** (ur5-wsg 1.00, franka-panda 0.95) |
| P4 visibility | visible fraction ≥ 0.3 in ≥ 1 camera in ≥ 95% of seeds | T2: **PASS** (bowl_src exactly 0.95 — borderline) · T1: **FAIL** (per item: on_path 0.85, near_path 0.98, off_path 0.65 — finding F5) |
| P5 token alignment (gpu) | highest-coverage token cells contain the encoder token positions closest to the item | **PASS** (overlap 0.5 — borderline) |

---

## P1 — multi-object path crossing (T1)

Sweep 1 (pre-Fix-1, archived at `runs/item_envs/feas/P1_pre_fix1/`): 1000 rows =
5 hosts × 2 embodiments × 50 seeds × (objects | baseline). Aggregate:
**near2 13.7% (gate ≥ 50), touch 15.6% (gate ≥ 30), baseline success 94.1%
(gate ≥ 70 — PASS) → gate FAIL.** 19 error rows (AssertionError / LinAlgError
from cuRobo planning, spread over place_container_plate + place_bread_basket).

The failure is **layout infeasibility, not exposure geometry**. Only 92/500
object episodes produced any layout (k > 0); the rest exhausted the layout
budget with all-`None` draws:

| host | embodiment | feasible | host | embodiment | feasible |
|---|---|---|---|---|---|
| place_container_plate | ur5-wsg | 8/50 (16%) | place_container_plate | franka-panda | 8/50 (16%) |
| place_burger_fries | ur5-wsg | 14/50 (28%) | place_burger_fries | franka-panda | 11/50 (22%) |
| stack_blocks_two | ur5-wsg | 0/50 | stack_blocks_two | franka-panda | 0/50 |
| stack_bowls_two | ur5-wsg | 0/50 | stack_bowls_two | franka-panda | 0/50 |
| place_bread_basket | ur5-wsg | 28/50 (56%) | place_bread_basket | franka-panda | 23/50 (46%) |

On the 92 feasible episodes the gate passes comfortably: near2 = 67/92 (73%),
and per-item min robot distance < 2 cm for **88% of on_path** and **74% of
near_path** items (median on_path min-dist is negative — contact). The
spawner's constraints, not the task corridors' exposure, are the bottleneck:
`_layout_plan` failed atomically when any single object's `_find_xy` found no
point (short corridors make the on-path t∈[0.22,0.48] window geometrically
empty; stack hosts additionally inflate every keepaway via
`obstacle_bbox_gap` 0.02–0.07).

**Fix 1 (applied, this gate's first documented fix): adaptive placement +
quota degradation** in `main/item_envs/spawn.py` — (a) on-path `_find_xy` gains
a deterministic 25-point fallback scan of t∈[0.10,0.75] after the base window
fails (`t_fallback` recorded); (b) a failed class retries the same object down
on_path→near_path→off_path, quota-gated so off_path only absorbs slots beyond
the on/near minimum (`degraded_from` recorded); (c) partial-K acceptance: a
layout with k ≥ 2 and the on/near quota met is kept instead of discarded
(`k_dropped` recorded); (d) MAX_LAYOUTS 25→40, NEAR_PATH_TRIES 12→40.
Offline replay of the diagnostic cells: place_container_plate layout
feasibility 2/4 → 4/4 seeds. **Rerun submitted (array 22611061, same probe +
grid into a fresh `runs/item_envs/feas/P1/`).** If the rerun still fails the
gate, one documented fix remains before the STOP-gate report.

## P3-robot — push primitive with the robot (T3)

120 rows (2 embodiments × time_scale {0.5, 1.0, 1.3} × 20 seeds), single push
of `smoke_items` box_0, zero errors, zero plan failures.
`runs/item_envs/feas/P3-ROBOT/`.

| embodiment | ts=0.5 | ts=1.0 | ts=1.3 |
|---|---|---|---|
| ur5-wsg | **1.00** (median err 1.8 mm) | 0.70 (16.6 mm) | 0.25 |
| franka-panda | **0.95** (10.9 mm) | 0.70 (19.6 mm) | 0.30 |

Gate (≥ 80% within 3 cm at time_scale 0.5): **PASS on both embodiments.**
Accuracy degrades monotonically with time_scale — exactly the dose-response
T3 calibration needs (the 1.0–1.3 band straddles the 80% line).

## P4 addendum — T1 visibility (from the P1 sweep's t = 0 hook)

175 rows over the feasible episodes of sweep 1 (per-item max-camera visible
fraction). Per placement class: on_path **0.85**, near_path **0.98**, off_path
**0.65** (several off_path items fully invisible, vis_max = 0.0) → per-item
gate FAIL; per-seed "all candidates ≥ 0.3" is 0–50% per cell (gate 95%). The
deficit is concentrated in off_path candidates near the far table corners
(outside the camera rig's useful frustum) — *finding F5 below*. Fix 1 does not
change off_path placement; P4-T1 gets its own documented fix if the rerun
confirms the failure.

180 cells, 3 seeds each. `002_bowl` ids 1/2/3/5, N ∈ {30, 60, 100} beads,
r ∈ {5, 6, 8} mm, mass 0.5–2 g. Data: `runs/item_envs/feas/P2/` (logs
`p3b_p2grid_headless_220002.log`, `p2showcase_220228.log`).

- **Rest containment**: `inside_frac = 1.0` in all cells except the r = 8 mm
  high-fill cells (physical overflow — beads stack above the rim by design):
  bowl 1 (60,8) 0.967, (100,8) 0.703; bowl 2 (100,8) 0.930; bowl 3 (100,8)
  0.963; bowl 5 (100,8) 1.0. Default cell (60 beads, 6 mm): **1.0 for all
  four bowl ids** → rest gate PASS at the design point. Recommendation:
  use N=60, r=6 mm as T2's default (already the YAML default).
- **Tunneling**: **zero in all 180 rows**, end-state and trajectory-min.
  Tunneling criterion: below table top by > r/2 over the table footprint, or
  under the bowl's world AABB min-z while the bowl is within 45° of upright
  (the uprightness gate avoids false positives during tilt, when a tilted
  bowl's AABB legitimately overhangs the table edge).
- **Translate sweep** (N=60, r=6, 0.2 m stroke, v_max 0.5 m/s): out-of-bowl
  fraction **0.0 at a = 1, 2, 4, 8 m/s²**, and an extended diagnostic
  (a = 16, 32 m/s², v_max = 1.0 m/s, transient tracking) shows **not even a
  transient excursion** → *finding F2 below*.
- **Tilt sweep** (110°): spill fraction 1.0 at both ω = 1 and 3 rad/s
  (saturated). ω-sensitive metrics do separate: mean pour-start angle
  **85.2° (ω=1) vs 101.1° (ω=3)**; spill scatter 0.092 vs 0.097 m →
  *finding F3 below*.
- **Physics wall time at N=60** (AMD EPYC 7763): settled pile 0.076 ms/tick
  mean (0.17 max); active pile 9.37 ms/tick mean (10.77 max) — **≤ 20 ms
  gate PASS** with ~2× headroom.
- Spawn geometry note: bowls taper toward the floor (bowl 5: 4.2 cm at floor
  vs 6.0 at rim); the spawn grid uses per-layer keep radii from the collision
  mesh (2.5 mm bands) — fixed a spawn-ejection failure (beads at 76 m/s).

## P3 — kinematic push coast (T3)

27 cells (v ∈ {0.1, 0.3, 0.6} m/s × μ ∈ {0.3, 0.6, 0.9} × 3 seeds), headless
CPU PhysX, kinematic pusher, 5 cm cube 0.1 kg, 0.15 m push then stop. Data:
`/work/nvme/bifq/project/tmp/p3_coast/` (to be moved under
`runs/item_envs/feas/P3/` at commit).

Median coasting distance (m); theory v²/(2μg) in parentheses:

| v \ μ | 0.3 | 0.6 | 0.9 |
|---|---|---|---|
| 0.1 | 1.51e-3 (1.70e-3) | 6.68e-4 (8.49e-4) | 3.34e-4 (5.66e-4) |
| 0.3 | 1.47e-2 (1.53e-2) | 7.05e-3 (7.65e-3) | 4.50e-3 (5.10e-3) |
| 0.6 | 5.76e-2 (6.12e-2) | 2.71e-2 (3.06e-2) | 1.69e-2 (2.04e-2) |

Measured/theory 0.59–0.96; coast positive everywhere, monotonic ↑ in v and ↓
in μ → **all three gate clauses PASS**.

## P4 — visibility at t = 0

T2 layout (20 seeds, bowl + beads + target + two zone mats; RoboTwin D435
camera rig via the `smoke_items` Env). Fraction of seeds with max-camera
visibility ≥ 0.3: bowl_src **0.95** (min 0.283 — one seed just under; bowl at
table edge), target **1.0** (min 0.607), zone_red **1.0** (min 0.842),
zone_blue **1.0** (min 0.644) → PASS, **borderline on bowl_src**; a small
layout-margin tweak buys headroom if wanted. T1/T3 visibility rows are
collected by the P1/P3-robot sweeps' t=0 hooks (20 seeds each) — pending.

## P5 — token alignment (gpu)

`smoke_items` box_0, head camera, HoloBrain encoder `pos_mean` (world frame)
vs `masks.item_masks` token cells. k = 6 top-coverage cells: **overlap 0.5**
(3/6 nearest-`pos_mean` tokens fall in top-coverage cells; the misses are
edge cells whose depth-weighted anchor lands on background). dist_top 0.181 m
vs dist_all 0.283 m. Gate "highest-coverage cells contain the closest token
positions": PASS at exactly 0.5 — **borderline**; overlay keyframe in
`runs/item_envs/feas/P5/keyframes/`.

---

## Findings for the STOP-gate review (human decisions needed)

- **F1 (P1 layout feasibility) — quantified by sweep 1, Fix 1 applied.**
  Confirmed and extended by the full sweep: layout feasibility is 18%
  overall (0% on both stack hosts, 16–28% on place_container_plate /
  place_burger_fries, ~50% on place_bread_basket), and it — not exposure
  geometry — fails the P1 gate (on feasible episodes near2 = 73%). Fix 1
  (adaptive placement + quota degradation, see the P1 section) is this gate's
  first documented fix; the rerun (array 22611061) decides. If it still
  fails, the remaining options are: one more documented fix (e.g. per-host
  keepaway/`obstacle_bbox_gap` overrides in the T1 YAML), or STOP and report —
  dropping hosts is reserved decision #3 (human).
- **F2 (P2 translate gate assumption).** Horizontal translation produces
  zero spill even at 32 m/s² peak accel — the regime is genuinely benign for
  a rimmed bowl at these masses/frictions; the "spill increases with accel"
  gate clause is not achievable by translation. The T2 risk comes from the
  pour (tilt), not the carry. Recommend: drop the translate monotonicity
  clause (or keep the translate cells as a "stays contained" sanity check).
- **F3 (P2 tilt saturation).** At the plan's 110° tilt, both ω = 1 and 3
  rad/s pour everything (spill fraction 1.0) — the monotonicity clause is
  not measurable at this angle. ω does separate pour-start angle (85.2° vs
  101.1°) and scatter. Options: measure monotonicity at a shallower tilt
  (e.g. 60–80°) or accept pour-start/scatter as the ω-sensitive metrics.
- **F4 (borderline passes).** P4 bowl_src = 0.95 (exactly at gate) and P5
  overlap = 0.5 (exactly at gate) — both pass as written; noted for the
  record, no action required.
- **F5 (P4-T1 visibility).** T1 candidates fail the visibility gate
  (on_path 0.85, near_path 0.98, off_path 0.65 per item; the deficit is
  off_path items in the far table corners, several with vis_max = 0.0).
  Candidate documented fix: restrict off_path sampling to the camera rig's
  useful frustum (empirically the visible table band), or drop the off_path
  class fraction in favor of near_path. Deferred until the P1 rerun lands —
  P4-T1 will be re-measured from the rerun's t = 0 hook on the same seeds.

## Infra notes (probe-platform lessons, for the record)

- ~26 render-system scene creations in one process exhaust svulkan2
  (`vk::createInstanceUnique` → segfault): bulk grids run headless; only
  showcase cells re-render for keyframes.
- `Env.__init__` chdirs to the RoboTwin root — all probe `--out` paths are
  resolved to absolute before Env creation.
- P1/P3-robot probes re-exec one child per embodiment (pybind11 TLS-key
  limit, AGENTS.md pattern); rows are resumable (complete rows skipped,
  crash-truncated tail dropped), so sweeps survive interactive-timeouts and
  preemption.
