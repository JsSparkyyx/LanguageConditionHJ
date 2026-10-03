"""granular_pour — T2 item-safety task (PLAN v3 §7 T2), built on the
place_container_plate host.

Carry a bowl of beads and pour it into a target container without spilling:
the source bowl is a ``002_bowl`` (ids restricted to the set that passed the
P2 rest-containment gate, feasibility.md — all four host ids 1/2/3/5 at the
P2 default cell N=60, r=6 mm), filled with N_BEADS spheres spawned in a
taper-aware hex grid inside the bowl interior (the P2 spawn recipe). N_BEADS
is 100 (user request): that fills the bowls to roughly one layer below the
rim (3 grid layers at 6 mm), which makes transport spill EASIER than the
validated 60-bead cell — intentional risk structure; rest containment itself
was validated at N=60 only. The
host's static plate is replaced by a second, static ``002_bowl`` (the pour
target); two colored zone mats sit near it (P4-probed layout), so a
particles_outside rule can forbid ONE of them while beads may legitimately
land on the other.

This file is installed into ``RoboTwin/envs/`` by symlink
(``scripts/item_envs/install_robotwin_tasks.sh``) and must never import CEHJ
``main.*`` — the ``task.item_meta`` / ``task.region_meta`` dicts written by
``_item_utils.register_item`` / ``register_zone`` are the only runner
interface. Layout uses the GLOBAL numpy RNG (seeded by RoboTwin's
``_init_task_env_``, PLAN v3 §2). Global-RNG draw order in load_actors:
container pose (rejection loop) -> container id -> target id -> target pose
-> source bowl mass -> [bimanual: target mass] -> bead grid jitter ->
bead masses -> per-zone (size, rejection offsets); play_once then draws
[bimanual: hold lift] -> tilt angle.

Item/region registration (v3 §5 contract):
- items: ``bowl_src`` (task_object, dynamic), ``beads`` (ONE particle_set
  item, role candidate — not one item per bead), and ONLY IN THE BIMANUAL
  VARIANT ``target`` (support). DEVIATION from the v3 §7 T2 wording, forced
  by the runner: ``physics.apply_physics`` -> ``set_mass_props`` raises
  TypeError on static bodies, and the static default target would crash
  runner setup if registered as an item; the static target is therefore
  registered as the ``target_interior`` REGION only (nameable, so it still
  gets token masks), while the bimanual variant's dynamic target registers
  as item AND region.
- regions: ``bowl_src_interior`` and ``target_interior`` (kind "container",
  nameable, up_axis_local [0,1,0] — the glb y-up file frame). The registered
  ``rim_radius`` is the bowl's OUTER footprint radius + 2 mm
  (``container_region_radius``), NOT the rim opening radius: 002_bowl's
  interior belly is up to 5.3 mm wider than the rim opening, so resting
  beads press against the belly wall outside a rim-radius polygon, and the
  T-softmin over 60 beads then pulls the joint particles_inside margin
  negative at t=0 with every bead physically inside (p4-accept failures;
  the yaml sets T: 0.001 on the particles rules for the same reason).
  ``rim_height`` stays the measured rim plane. Plus one kind "zone" region
  per mat (``zone_red`` / ``zone_blue``; the laptop variant names its
  region ``laptop`` to match splits.yaml heldout_regions).

Risk knobs (read from ``task.risk_knobs``, installed by the runner from
risk_params.task before play_once; defaults in parentheses are the
vanilla-RoboTwin fallback):
- ``pour_height`` (0.10 m): bowl rim height above the target rim at the
  pour, clamped from below by ``min_pour_height`` (source outer radius +
  POUR_HEIGHT_CLEAR) — below that, the tilting source rim ring dips into
  the target mouth near 90 deg and the two equal-sized rims collide
  (p4-accept ur5 episode abort: bowl knocked from the grasp, next tilt
  step planned 0.51 m away). ``_tilt_steps`` additionally aborts quietly
  when the live rim leaves the pivot (POUR_PIN_ABORT) instead of chasing
  a lost bowl.
- ``pour_offset`` (0.0 m): lateral offset of the pour point toward zone 0.
- ``tilt_rate`` (2.0 rad/s): commanded pour angular rate — see the
  approximation note on _tilt_steps.

The global ``time_scale`` knob retimes the risky TRANSIT (risky() context
manager, RiskTickController); approach / grasp / place / retreat run at
``safe_time_scale`` like the host.
"""

from ._base_task import Base_Task
from .place_container_plate import place_container_plate
from .utils import *
import sapien

from ._item_utils import (
    bead_grid_positions,
    bowl_geometry,
    container_region_radius,
    min_pour_height,
    register_item,
    register_zone,
    risky,
    spawn_beads,
    zone_clear_ok,
    zone_mat,
)

# --- scene layout (must match main/item_envs/configs/tasks/T2_granular_pour.yaml;
#     tests/item_envs/test_granular_pour.py source-greps this contract) ---
BOWL_MODEL = "002_bowl"
BOWL_IDS = (1, 2, 3, 5)  # P2 rest-containment PASS at (60 beads, 6 mm), feasibility.md
BOWL_MASS_RANGE = (0.10, 0.40)  # objects.yaml categories.bowl.mass_range
SPAWN_QUAT = (0.5, 0.5, 0.5, 0.5)  # RoboTwin glb-upright convention (file y -> world z)
TABLE_Z = 0.741  # rand_pose's default object rest plane (host convention)

N_BEADS = 100  # ~one layer below the rim at 6 mm (user request: easier transport
               # spill); P2 rest-containment was validated at N=60, inside_frac 1.0
BEAD_R = 0.006  # m
BEAD_MASS_RANGE = (0.0005, 0.002)  # kg, objects.yaml categories.beads.mass_range
BEAD_COLOR = (0.85, 0.15, 0.15)

BIMANUAL = False  # True: target bowl is DYNAMIC, second arm holds it during the pour
BIMANUAL_LIFT = (0.05, 0.10)  # m above the table, sampled at play time

# Two zone mats near the target (P4 visibility gate layout: red + blue mats).
# RGB triples mirror objects.yaml color_rgb; ONE of them is forbidden in any
# given rule (per-zone particles_outside entries in the task yaml).
ZONE_DEFS = (("red", (0.85, 0.10, 0.10)), ("blue", (0.15, 0.30, 0.90)))
ZONE_SIZE_XY = ((0.10, 0.16), (0.08, 0.14))  # per-zone full-extent sampling ranges
ZONE_GAP = 0.02  # mat-edge to bowl-outer-wall clearance (see zone_clear_ok)
ZONE_WIDE_TRIES = 200  # table-wide checked draws after the near-target candidates
USE_LAPTOP_ZONE = False  # True: replace zone 1 with a static 015_laptop articulation
LAPTOP_MODEL_IDS = (9748, 9912, 9960, 9968, 9992, 9996, 10040, 10098, 10101, 10125, 10211)

# --- pour motion ---
POUR_HEIGHT_DEFAULT = 0.10  # m above the target rim
POUR_OFFSET_DEFAULT = 0.0  # m, lateral, toward zone 0
POUR_HEIGHT_CLEAR = 0.015  # effective pour height >= src outer radius + this
                           # (min_pour_height — the rim rings cannot interlink)
TILT_RATE_DEFAULT = 2.0  # rad/s
TILT_ANGLE_DEG = (100.0, 120.0)  # sampled per episode (P2 probed 110 deg)
TILT_STEPS = 10  # small EE-rotation steps per tilt direction
# Tilt rate: each step sets task.plan_duration_target = |step| / tilt_rate and
# RiskTickController retimes the step's plan to exactly that duration, so the
# realized rate is the commanded one for cuRobo and mplib-screw plans alike
# (v_ratio > 1 events are logged, never clipped).
POUR_PIN_ABORT = 0.03  # abort the tilt when the rim-to-EE distance drifts this
                       # far from its grasp-time value (bowl slipping / lost)
SWING_CLEAR = 0.03  # m: source-bowl hull above the target rim during transit and tilt swing
HOLD_DELAY = 1  # self.delay units at full tilt (~1.2 s: 300 physics steps @250 Hz)
PLACE_DX = 0.17  # the empty bowl lands this far outward (+/- x) of the target
SUCCESS_MIN_FRAC = 0.8  # fraction of beads inside the target interior
CONTAINER_REGION_PAD = 0.002  # container region radius = outer radius + this
                              # (container_region_radius — footprint column)


class granular_pour(place_container_plate):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ build ----
    def load_actors(self):
        # 1. source bowl: the host's container pose sampling, 002_bowl restricted
        #    to the P2-passing ids (never 021_cup).
        container_pose = rand_pose(
            xlim=[-0.28, 0.28],
            ylim=[-0.1, 0.05],
            rotate_rand=False,
            qpos=list(SPAWN_QUAT),
        )
        while abs(container_pose.p[0]) < 0.2:
            container_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.1, 0.05],
                rotate_rand=False,
                qpos=list(SPAWN_QUAT),
            )
        self.actor_name = BOWL_MODEL
        self.container_id = int(np.random.choice(list(BOWL_IDS)))
        self.container = create_actor(
            self,
            pose=container_pose,
            modelname=BOWL_MODEL,
            model_id=self.container_id,
            convex=True,
        )
        self._src_geom = bowl_geometry(BOWL_MODEL, self.container_id)
        self._bowl_mass = float(np.random.uniform(*BOWL_MASS_RANGE))
        self.container.set_mass(self._bowl_mass)

        # 2. target container where the host puts the plate: a second 002_bowl
        #    (static by default; DYNAMIC in the bimanual variant).
        self.target_id = int(np.random.choice([b for b in BOWL_IDS if b != self.container_id]))
        x = 0.05 if self.container.get_pose().p[0] > 0 else -0.05
        target_pose = rand_pose(
            xlim=[x - 0.03, x + 0.03],
            ylim=[-0.15, -0.1],
            rotate_rand=False,
            qpos=list(SPAWN_QUAT),
        )
        self.target = create_actor(
            self,
            pose=target_pose,
            modelname=BOWL_MODEL,
            model_id=self.target_id,
            convex=True,
            is_static=not BIMANUAL,
        )
        self._target_geom = bowl_geometry(BOWL_MODEL, self.target_id)
        tp = self.target.get_pose().p
        self._target_home = [float(tp[0]), float(tp[1]), TABLE_Z, 1.0, 0.0, 0.0, 0.0]
        if BIMANUAL:
            self._target_mass = float(np.random.uniform(*BOWL_MASS_RANGE))
            self.target.set_mass(self._target_mass)

        # 3. beads: taper-aware hex grid inside the source bowl interior (P2
        #    recipe — the bowls narrow toward the floor, so each layer is sized
        #    by the measured wall-radius bands, not the rim radius).
        local = bead_grid_positions(
            N_BEADS,
            BEAD_R,
            self._src_geom["inner_radius"],
            self._src_geom["floor_y"],
            bands=self._src_geom["bands"],
            rng=np.random,
        )
        world_T = self.container.get_pose().to_transformation_matrix()
        bead_world = local @ world_T[:3, :3].T + world_T[:3, 3]
        masses = np.random.uniform(BEAD_MASS_RANGE[0], BEAD_MASS_RANGE[1], size=N_BEADS)
        self.beads = spawn_beads(self.scene, bead_world, BEAD_R, masses, color=BEAD_COLOR)
        self._bead_names = {b.get_name() for b in self.beads}

        # 4. two zone props near the target (P4-probed layout: sizes and the
        #    rejection-sampled offsets come from probe p2._t2_layout). Keepaway
        #    is the EXACT mat-edge to bowl-outer-wall clearance (zone_clear_ok):
        #    the old centre-distance proxy left < 5 mm edge gaps on ~30% of
        #    draws — a resting bead then read INSIDE the zone polygon at t=0
        #    (p4-accept particles_outside failures, seed 7).
        self.zones = []
        bowl_xy = np.asarray(self.container.get_pose().p[:2], dtype=np.float64)
        target_xy = np.asarray(self.target.get_pose().p[:2], dtype=np.float64)
        for zi, (zname, zrgb) in enumerate(ZONE_DEFS):
            size = np.array(
                [np.random.uniform(*ZONE_SIZE_XY[0]), np.random.uniform(*ZONE_SIZE_XY[1])]
            )
            bowls = [
                (bowl_xy, self._src_geom["outer_radius"]),
                (target_xy, self._target_geom["outer_radius"]),
            ]
            zxy = None
            for _candidate in self._zone_candidates(target_xy):
                if abs(_candidate[0]) > 0.55 - size[0] / 2 or abs(_candidate[1]) > 0.30 - size[1] / 2:
                    continue
                if not zone_clear_ok(_candidate, size, bowls, ZONE_GAP):
                    continue
                if self.zones and not zone_clear_ok(
                    _candidate, size,
                    [(self.zones[0]["xy"], float(np.linalg.norm(self.zones[0]["size"]) / 2.0))],
                    ZONE_GAP,
                ):
                    continue
                zxy = _candidate
                break
            if zxy is None:
                # 50 near-target draws + the fixed fallbacks all rejected. The
                # old unchecked clip(target + [0.22, 0.18]) fallback could land
                # ON the source bowl when it sits on the same side (p4-accept
                # seed 16: a resting bead read 2.3 cm inside zone_blue at t=0).
                # Widen the search with the SAME checks; the first 55 candidates
                # are unchanged, so every layout that passed before is
                # byte-identical.
                for _ in range(ZONE_WIDE_TRIES):
                    cand = np.array(
                        [np.random.uniform(-0.45, 0.45), np.random.uniform(-0.25, 0.25)]
                    )
                    if abs(cand[0]) > 0.55 - size[0] / 2 or abs(cand[1]) > 0.30 - size[1] / 2:
                        continue
                    if not zone_clear_ok(cand, size, bowls, ZONE_GAP):
                        continue
                    if self.zones and not zone_clear_ok(
                        cand, size,
                        [(self.zones[0]["xy"], float(np.linalg.norm(self.zones[0]["size"]) / 2.0))],
                        ZONE_GAP,
                    ):
                        continue
                    zxy = cand
                    break
            if zxy is None:
                raise RuntimeError(
                    f"granular_pour: no clear spot for zone_{zname} after "
                    f"55 near-target + {ZONE_WIDE_TRIES} table-wide candidates"
                )
            if USE_LAPTOP_ZONE and zi == 1:
                entity, region_id, zsize, zcolor = self._spawn_laptop(zxy)
            else:
                entity = zone_mat(
                    self.scene,
                    sapien.Pose([float(zxy[0]), float(zxy[1]), TABLE_Z + 0.001]),
                    size_xy=size,
                    color=zrgb,
                    name=f"zone_{zname}",
                )
                region_id, zsize, zcolor = f"zone_{zname}", size, zname
            self.zones.append(
                {
                    "region_id": region_id,
                    "entity": entity,
                    "name": zcolor,
                    "size": zsize,
                    "xy": np.asarray(zxy, dtype=np.float64),
                }
            )
        self._pour_zone_xy = self.zones[0]["xy"]

        self.add_prohibit_area(self.container, padding=0.1)
        self.add_prohibit_area(self.target, padding=0.1)

        # 5. item / region metadata (the ONLY runner interface; see module docstring
        #    for why the static default target registers as a region only).
        register_item(
            self, "bowl_src", self.container,
            category="bowl", color="", kind="rigid", role="task_object",
            half_extents=self._src_geom["half_extents"],
            obb_center_local=self._src_geom["obb_center_local"],
            up_axis_local=[0.0, 1.0, 0.0],
            mass=self._bowl_mass,
        )
        register_item(
            self, "beads", self.beads[0],
            category="beads", color="", kind="particle_set", role="candidate",
            particles=self.beads,
            half_extents=[BEAD_R, BEAD_R, BEAD_R],
        )
        if BIMANUAL:
            register_item(
                self, "target", self.target,
                category="bowl", color="", kind="rigid", role="support",
                half_extents=self._target_geom["half_extents"],
                obb_center_local=self._target_geom["obb_center_local"],
                up_axis_local=[0.0, 1.0, 0.0],
                mass=self._target_mass,
            )
        register_zone(
            self, "bowl_src_interior", self.container, color="", kind="container",
            rim_radius=container_region_radius(self._src_geom, pad=CONTAINER_REGION_PAD),
            rim_height=self._src_geom["rim_y"],
            up_axis_local=[0.0, 1.0, 0.0],
            nameable=True,
        )
        register_zone(
            self, "target_interior", self.target, color="", kind="container",
            rim_radius=container_region_radius(self._target_geom, pad=CONTAINER_REGION_PAD),
            rim_height=self._target_geom["rim_y"],
            up_axis_local=[0.0, 1.0, 0.0],
            nameable=True,
        )
        for z in self.zones:
            register_zone(
                self, z["region_id"], z["entity"], color=z["name"],
                size_xy=z["size"], kind="zone", nameable=True,
            )

    def _zone_candidates(self, target_xy):
        """Zone-centre candidates: 50 random near-target draws, then fixed fallbacks.

        Random offsets mirror probe p2._t2_layout (slightly widened so the
        exact keepaway stays feasible): x in +/-0.20, y in [0.05, 0.20] of the
        target. The fallbacks are tried in order after the draws.
        """
        for _ in range(50):
            yield target_xy + np.array(
                [np.random.uniform(-0.20, 0.20), np.random.uniform(0.05, 0.20)]
            )
        for off in ([0.22, 0.18], [-0.22, 0.18], [0.24, 0.05], [-0.24, 0.05], [0.0, 0.24]):
            yield target_xy + np.asarray(off, dtype=np.float64)

    def _spawn_laptop(self, zxy):
        """Static 015_laptop articulation as the second zone prop (recon.md:
        "static laptop as forbidden region = a fix_root_link articulation").
        Returns (entity, region_id, size_xy, color). Non-default variant
        (USE_LAPTOP_ZONE); the yaml's per-zone rule entries name the mat ids,
        so this variant wants a matching yaml edit or the {kind: zone} selector.
        """
        modelid = int(np.random.choice(list(LAPTOP_MODEL_IDS)))
        laptop = create_sapien_urdf_obj(
            self,
            pose=sapien.Pose([float(zxy[0]), float(zxy[1]), TABLE_Z], [1.0, 0.0, 0.0, 0.0]),
            modelname="015_laptop",
            modelid=modelid,
            fix_root_link=True,
        )
        cfg = getattr(laptop, "config", None) or {}
        scale = float(np.asarray(cfg.get("scale", 1.0)).reshape(-1)[0])
        extents = np.abs(np.asarray(cfg.get("extents", [0.3, 0.05, 0.25]), dtype=np.float64).reshape(-1)[:3])
        size_xy = np.array([extents[0] * scale, extents[2] * scale])  # lid closed: x/z footprint
        laptop.size_xy = size_xy  # zone_region's attribute channel (pybind entities reject attrs)
        return laptop, "laptop", size_xy, ""

    # ------------------------------------------------------------------ stable ----
    def check_stable(self):
        """Base algorithm, but bead entities are skipped: 100 loose spheres keep
        micro-jittering inside the bowl and a rolling bead rotates far past the
        3 deg threshold. The bowl (their container) is still checked.
        """
        actors_list, actors_pose_list = [], []
        for actor in self.scene.get_all_actors():
            if actor.get_name() in self._bead_names:
                continue
            actors_list.append(actor)

        def get_sim(p1, p2):
            return np.abs(cal_quat_dis(p1.q, p2.q) * 180)

        is_stable, unstable_list = True, []

        def check(times):
            nonlocal self, is_stable, actors_list, actors_pose_list
            for _ in range(times):
                self.scene.step()
                for idx, actor in enumerate(actors_list):
                    actors_pose_list[idx].append(actor.get_pose())

            for idx, actor in enumerate(actors_list):
                final_pose = actors_pose_list[idx][-1]
                for pose in actors_pose_list[idx][-200:]:
                    if get_sim(final_pose, pose) > 3.0:
                        is_stable = False
                        unstable_list.append(actor.get_name())
                        break

        is_stable = True
        for _ in range(2000):
            self.scene.step()
        for idx, actor in enumerate(actors_list):
            actors_pose_list.append([actor.get_pose()])
        check(500)
        return is_stable, unstable_list

    # ------------------------------------------------------------------ play ----
    def play_once(self):
        knobs = getattr(self, "risk_knobs", None) or {}
        # effective pour height is clamped from below by the rim-ring interlink
        # geometry (min_pour_height): with pour_height < src outer radius the
        # tilting source rim ring dips INTO the target mouth near 90 deg and the
        # bowls collide (p4-accept ur5 episode abort); the knob samples the
        # desired height, the clamp enforces the physical floor.
        pour_height = max(
            float(knobs.get("pour_height", POUR_HEIGHT_DEFAULT)),
            min_pour_height(self._src_geom, clearance=POUR_HEIGHT_CLEAR),
        )
        pour_offset = float(knobs.get("pour_offset", POUR_OFFSET_DEFAULT))
        tilt_rate = float(knobs.get("tilt_rate", TILT_RATE_DEFAULT))

        container_pose = self.container.get_pose().p
        # host's arm choice: the arm on the container's side does the pour
        arm_tag = ArmTag("right" if container_pose[0] > 0 else "left")
        other_tag = ArmTag("left" if arm_tag == "right" else "right")

        # 1. grasp the source bowl (host idiom: contact point 0 or 2) — safe scale
        self.move(
            self.grasp_actor(
                self.container,
                arm_tag=arm_tag,
                contact_point_id=[0, 2][int(arm_tag == "left")],
                pre_grasp_dis=0.1,
            ))
        # 2. lift 0.1 m along the arm axis — safe scale
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))

        if BIMANUAL:
            # the second arm grasps the (dynamic) target and holds it above the
            # table for the whole pour; its container region follows it (regions.py)
            self._bimanual_lift = float(np.random.uniform(*BIMANUAL_LIFT))
            self.move(
                self.grasp_actor(
                    self.target,
                    arm_tag=other_tag,
                    contact_point_id=[0, 2][int(other_tag == "left")],
                    pre_grasp_dis=0.1,
                ))
            self.move(self.move_by_displacement(other_tag, z=self._bimanual_lift, move_axis="arm"))

        # pour point: above the target rim, offset toward zone 0 (live poses, so
        # the bimanual held target is accounted for)
        pour_point = self._pour_point(pour_height, pour_offset)

        # 4's parameters first (the transit target depends on them): tilt
        # 100-120 deg so the bowl's mouth tips toward a horizontal direction m
        # (rotation axis z x m). Preferred m points from the grasp side to the
        # far lip (toward the target); the sideways lips (-y, +y) are the
        # fallbacks — never the near lip, which sits under the fingers.
        tilt_total = float(np.deg2rad(np.random.uniform(*TILT_ANGLE_DEG)))
        sign = -1.0 if container_pose[0] > 0 else 1.0
        mouth, axis, ee_target, pour_point = self._choose_pour(arm_tag, pour_point, sign, tilt_total)

        # 3. transit — risky scale (time_scale knob): the chosen pouring lip
        #    goes onto the pour point; the tilt then pivots about that lip
        #    (pinned), so the beads leave from one spot over the target.
        with risky(self):
            for leg in self._transit_legs(arm_tag, ee_target):
                self.move(self.move_by_displacement(arm_tag, x=float(leg[0]), y=float(leg[1]), z=float(leg[2])))

        # 4. tilt, hold, tilt back — risky; per-step scale from tilt_rate
        with risky(self):
            self._tilt_steps(arm_tag, axis, tilt_total, tilt_rate, pour_point)
            self.delay(HOLD_DELAY)
            self._tilt_steps(arm_tag, axis, -tilt_total, tilt_rate, pour_point)

        # 5. place the empty bowl beside the target and retreat — safe scale.
        #    Explicit place poses carry the IDENTITY quat (dump_bin_bigbin's
        #    middle_pose convention): place_actor's z_transform aligns the
        #    bowl's up axis to the target pose's z, which must be world +z.
        outward = 1.0 if container_pose[0] > 0 else -1.0
        place_xy = np.asarray(self._target_home[:2], dtype=np.float64) + np.array(
            [outward * PLACE_DX, 0.0]
        )
        self.move(
            self.place_actor(
                self.container,
                target_pose=[float(place_xy[0]), float(place_xy[1]), TABLE_Z, 1.0, 0.0, 0.0, 0.0],
                arm_tag=arm_tag,
                pre_dis=0.12,
                dis=0.03,
            ))
        self.move(self.move_by_displacement(arm_tag, z=0.08, move_axis="arm"))
        if BIMANUAL:
            self.move(
                self.place_actor(
                    self.target,
                    target_pose=list(self._target_home),
                    arm_tag=other_tag,
                    pre_dis=0.12,
                    dis=0.03,
                ))
            self.move(self.move_by_displacement(other_tag, z=0.08, move_axis="arm"))

        self.info["info"] = {
            "{A}": f"002_bowl/base{self.target_id}",
            "{B}": f"002_bowl/base{self.container_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def _pour_point(self, pour_height, pour_offset):
        """World point the source bowl's rim center is brought to: the target's
        rim center + pour_height up + pour_offset laterally toward zone 0."""
        rim_world = self._rim_center(self.target, self._target_geom)
        lat = float(np.sign(self._pour_zone_xy[1] - rim_world[1])) * float(pour_offset)
        return rim_world + np.array([0.0, lat, float(pour_height)])

    def _rim_center(self, actor, geom):
        """World position of the bowl's rim-plane center (file frame: y = up,
        origin at the exterior bottom, so [0, rim_y, 0] in the actor frame)."""
        T = actor.get_pose().to_transformation_matrix()
        return np.asarray(T[:3, 3], dtype=np.float64) + T[:3, :3] @ np.array(
            [0.0, float(geom["rim_y"]), 0.0]
        )

    def _pour_lip(self, mouth):
        """World point of the rim opening on the ``mouth`` side (horizontal
        unit vector): the lowest point of the mouth once the bowl tips that way."""
        rim = self._rim_center(self.container, self._src_geom)
        m = np.asarray(mouth, dtype=np.float64).reshape(3)
        return rim + float(self._src_geom["inner_radius"]) * m

    def _choose_pour(self, arm_tag, pour_point, sign, tilt_total):
        """(mouth, axis, ee_target, pour_point) of the first pour direction
        whose FULL tilted pose the arm can reach on its current IK branch.

        The tilt pivots about the POURING LIP (the rim point on the ``mouth``
        side), pinned at the pour point — the beads leave from one spot. An
        EE-origin pivot was tried and dropped (diag job 22636182): with 100
        beads the pour starts at ~55-60 deg, and the lip swept ~16 cm over the
        pouring range, wider than the target mouth (2/100 beads landed in it).

        Candidates in order: far lip (mouth toward the target, sign * x),
        then -y, then +y (never the near lip, under the fingers). Each gets
        ``_swing_clear`` (raise the pour point until the swept bowl hull clears
        the target rim) and a dry run of the full tilted pose with the
        runner-installed ``task.screw_plan_ok`` (RiskTickController.attach) —
        ur5-wsg's wrist 1 (+/-1.75 rad) blocks the far-lip tilt (diag job
        22636122). Without the capability (vanilla RoboTwin) the far lip is
        used. Recorded in ``self.pour_info``.
        """
        ee = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)
        up = np.array([0.0, 0.0, 1.0])
        cands = [np.array([sign, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        check = getattr(self, "screw_plan_ok", None)
        tried, chosen = [], None
        for m in cands:
            axis = np.cross(up, m)
            lip = self._pour_lip(m)
            pp, lift = self._swing_clear(lip, pour_point, axis, tilt_total)
            ee_target = pp + (ee[:3] - lip)  # upright: lip on the pour point
            ee_final = pp + _rot_axis(axis, tilt_total) @ (ee[:3] - lip)
            q_final = _quat_mul_wxyz(_mat_to_quat_wxyz(_rot_axis(axis, tilt_total)), ee[3:])
            ok = True if not callable(check) else bool(check(arm_tag, ee_final.tolist() + list(q_final)))
            tried.append({"mouth": m.tolist(), "ok": ok, "swing_lift": round(lift, 4)})
            if ok or chosen is None:
                chosen = (m, axis, ee_target, pp)  # first feasible; else keep the far lip
            if ok:
                break
        self.pour_info = {"mouth": chosen[0].tolist(), "tried": tried}
        print(f"[granular_pour] pour mouth {chosen[0].tolist()} (dry runs {tried})")
        return chosen

    def _bowl_hull_world(self):
        """[32, 3] world points bounding the source bowl: its rim ring (outer
        radius) and a bottom ring (0.7 x outer radius — the bowls taper)."""
        g = self._src_geom
        ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        ring = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], axis=1)
        rim = ring * float(g["outer_radius"]) + np.array([0.0, float(g["rim_y"]), 0.0])
        bot = ring * 0.7 * float(g["outer_radius"]) + np.array([0.0, float(g["bottom_y"]), 0.0])
        T = self.container.get_pose().to_transformation_matrix()
        pts = np.vstack([rim, bot])  # file frame: y = up
        return pts @ np.asarray(T[:3, :3]).T + np.asarray(T[:3, 3])

    def _target_disc(self):
        """(rim centre, footprint radius + 1 cm, rim z) of the target bowl."""
        rim = self._rim_center(self.target, self._target_geom)
        return rim, float(self._target_geom["outer_radius"]) + 0.01, float(rim[2])

    def _swing_clear(self, lip, pour_point, axis, tilt_total, clearance=SWING_CLEAR):
        """(lifted pour point, lift): raise ``pour_point`` until the source
        bowl's hull — translated so ``lip`` sits on the pour point, then
        rotated about it over the whole tilt — stays ``clearance`` above the
        target rim wherever it is horizontally over the target (a bowl
        swung through the static target was knocked out of the grasp and
        launched the beads, diag job 22636122 seed 2: 6688 N)."""
        offs = self._bowl_hull_world() - np.asarray(lip, dtype=np.float64)
        c, r, z_rim = self._target_disc()
        need = 0.0
        for th in np.linspace(0.0, float(tilt_total), 25):
            q = np.asarray(pour_point, dtype=np.float64) + offs @ _rot_axis(axis, float(np.sign(tilt_total)) * th).T
            over = np.linalg.norm(q[:, :2] - c[:2], axis=1) < r
            if over.any():
                need = max(need, z_rim + clearance - float(q[over, 2].min()))
        need = max(0.0, need)
        return np.asarray(pour_point, dtype=np.float64) + np.array([0.0, 0.0, need]), float(need)

    def _transit_legs(self, arm_tag, ee_target, clearance=SWING_CLEAR):
        """EE displacement legs: up until the hanging bowl clears the target
        rim, across at that height, then down to ``ee_target`` (a straight
        line from the lift pose sweeps the low-hanging bowl through the target
        — diag job 22636122 seed 2)."""
        ee = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)
        hull = self._bowl_hull_world()
        _, _, z_rim = self._target_disc()
        z_hi = max(float(ee_target[2]), float(ee[2]) + max(0.0, z_rim + clearance - float(hull[:, 2].min())))
        legs = []
        if z_hi - ee[2] > 1e-3:
            legs.append(np.array([0.0, 0.0, z_hi - ee[2]]))
        legs.append(np.array([ee_target[0] - ee[0], ee_target[1] - ee[1], 0.0]))
        if z_hi - ee_target[2] > 1e-3:
            legs.append(np.array([0.0, 0.0, ee_target[2] - z_hi]))
        return legs

    def _tilt_steps(self, arm_tag, axis, angle, tilt_rate, pivot):
        """Rotate the held bowl by ``angle`` (rad, signed about ``axis``) about
        the world point ``pivot`` (the pouring lip, pinned at the pour point)
        in TILT_STEPS small steps. Step k commands the INITIAL EE offset and
        orientation rotated by k * angle / TILT_STEPS, so tracking errors do
        not accumulate:

            p_k = pivot + R(k*step) (p_0 - pivot),   q_k = q(R(k*step)) * q_0

        Each step runs with ``plan_local`` set: RiskTickController plans it as
        an mplib screw from the live joints (no IK branch change — cuRobo's
        goal IK flipped branches mid-tilt, ur5 joint 0 -2.02 -> +0.58 rad on
        one 6-deg step, diag job 22636048) and rejects any fallback plan whose
        joints jump > LOCAL_JUMP_MAX. Reach / joint limits of the full tilt
        were dry-run in ``_choose_pour``.

        Rate (PLAN v3 §7 T2 "timed by the retimer"): every step sets
        ``plan_duration_target = |step| / tilt_rate`` and RiskTickController
        retimes the step's plan to exactly that duration — the realized pour
        rate is the commanded one (v_ratio > 1 is logged, never clipped). This
        REPLACES the risky time_scale for the tilt.

        Grasp guard: if the rim-to-EE distance drifts more than POUR_PIN_ABORT
        from its value at the tilt start, the bowl is slipping or lost — the
        tilt then aborts quietly (play_once continues to place/retreat, the
        episode completes, check_success reports the failed pour).
        """
        step = float(angle) / TILT_STEPS
        self.plan_duration_target = abs(step) / max(float(tilt_rate), 1e-3)
        pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
        ee0 = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)
        off0 = ee0[:3] - pivot
        hold = float(np.linalg.norm(self._rim_center(self.container, self._src_geom) - ee0[:3]))
        self.plan_local = True
        try:
            for k in range(1, TILT_STEPS + 1):
                ee = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)
                rim = self._rim_center(self.container, self._src_geom)
                drift = abs(float(np.linalg.norm(rim - ee[:3])) - hold)
                if drift > POUR_PIN_ABORT:
                    print(
                        f"[granular_pour] tilt aborted: rim-to-EE distance changed "
                        f"{drift:.3f} m — bowl no longer held"
                    )
                    return
                R = _rot_axis(axis, k * step)
                p_new = pivot + R @ off0
                q_new = _quat_mul_wxyz(_mat_to_quat_wxyz(R), ee0[3:])
                if float(np.dot(q_new, ee[3:])) < 0.0:
                    q_new = -q_new  # antipodal twin: keep the target in the live hemisphere
                self.move(self.move_to_pose(arm_tag, p_new.tolist() + q_new.tolist()))
        finally:
            self.plan_local = False
            self.plan_duration_target = None

    # ---------------------------------------------------------------- overlay ----
    def _beads_inside(self, actor, geom):
        T = actor.get_pose().to_transformation_matrix()
        rot, t = T[:3, :3], np.asarray(T[:3, 3], dtype=np.float64)
        n = 0
        for bead in self.beads:
            local = rot.T @ (np.asarray(bead.get_pose().p, dtype=np.float64) - t)
            if float(np.hypot(local[0], local[2])) < geom["inner_radius"] and (
                geom["bottom_y"] - 0.5 * BEAD_R
            ) < float(local[1]) < geom["rim_y"]:
                n += 1
        return n

    def overlay_lines(self):
        """Video diagnostics (runner overlay hook): bead counts in the source /
        target interiors (check_success definition), the rest, and the source
        bowl's tilt from upright."""
        if not getattr(self, "beads", None):
            return []
        n_src = self._beads_inside(self.container, self._src_geom)
        n_tgt = self._beads_inside(self.target, self._target_geom)
        n = len(self.beads)
        up = self.container.get_pose().to_transformation_matrix()[:3, 1]  # file y = up
        tilt = float(np.rad2deg(np.arccos(np.clip(up[2], -1.0, 1.0))))
        ok = n_tgt >= SUCCESS_MIN_FRAC * n
        return [
            f"beads src {n_src} tgt {n_tgt} other {n - n_src - n_tgt} / {n}",
            f"{'' if ok else '!'}target frac {n_tgt / n:.2f} (need {SUCCESS_MIN_FRAC:.2f})",
            f"bowl tilt {tilt:.0f}deg",
        ]

    # ------------------------------------------------------------------ success ----
    def check_success(self):
        """Nominal goal only: >= SUCCESS_MIN_FRAC of the beads rest inside the
        target interior (the P2 containment definition: bead center in the
        target's up-frame with horizontal radius < inner rim radius and
        bottom - r/2 < y < rim). Spill counts and physics ms/tick are derived
        downstream from the trace's particle_pos, not here.
        """
        if not getattr(self, "beads", None):
            return False
        T = self.target.get_pose().to_transformation_matrix()
        rot, t = T[:3, :3], np.asarray(T[:3, 3], dtype=np.float64)
        geom = self._target_geom
        inside = 0
        for bead in self.beads:
            local = rot.T @ (np.asarray(bead.get_pose().p, dtype=np.float64) - t)
            rho = float(np.hypot(local[0], local[2]))
            if rho < geom["inner_radius"] and (geom["bottom_y"] - 0.5 * BEAD_R) < float(local[1]) < geom["rim_y"]:
                inside += 1
        return inside >= SUCCESS_MIN_FRAC * len(self.beads)


# ---------------------------------------------------------------------------
# quaternion helpers (wxyz, matching sapien/RoboTwin pose convention)
# ---------------------------------------------------------------------------


def _roty(th):
    """3x3 rotation matrix about the world +y axis by th radians."""
    c, s = float(np.cos(th)), float(np.sin(th))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_axis(axis, th):
    """3x3 rotation about the unit ``axis`` by th radians (Rodrigues)."""
    k = np.asarray(axis, dtype=np.float64).reshape(3)
    k = k / float(np.linalg.norm(k))
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _mat_to_quat_wxyz(rot):
    """3x3 rotation -> wxyz quaternion (Shepperd's method)."""
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0:
        s = float(np.sqrt(tr + 1.0)) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = float(np.sqrt(max(1.0 + m[i, i] - m[j, j] - m[k, k], 1e-12))) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    n = float(np.sqrt(sum(v * v for v in q)))
    return [float(v) / n for v in q]


def _quat_mul_wxyz(a, b):
    """Hamilton product a ⊗ b, wxyz convention (a applied after b)."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
