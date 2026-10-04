"""granular_pour — T2 item-safety task (PLAN v3 §7 T2), built on the
place_container_plate host.

Carry a bowl of beads and pour it into a target container without spilling.
The SOURCE is a ``002_bowl`` (ids 1/2/3/5, P2 rest-containment set) with a
RAISED FLOOR (``create_bowl_raised_floor``: interior SRC_INTERIOR_DEPTH =
2.5 cm deep, outer wall unchanged; user review 5 — a low container spills at
small tilts and in transit, so the filter must control tilt AND velocity),
filled with a random pack of SRC_FILL_FRAC x its rim capacity (capped at
N_BEADS). The host's static plate is replaced by a second, plain static
``002_bowl`` (the pour target). The only safety constraint is "spill no
bead" (user decision 2026-10-03: the forbidden-mat rules were dropped).

Pour styles (``embodiment_params.pour_style``): default — top-down rim
pinch, lip-pivot tilt by dry-run-chosen mouth (franka); ``wrist_roll`` —
tangent rim pinch pitched ROLL_PITCH_DEG down, pour = roll about the
approach axis (ur5-wsg: its wrist 1 blocks the top-down tilts).

This file is installed into ``RoboTwin/envs/`` by symlink
(``scripts/item_envs/install_robotwin_tasks.sh``) and must never import CEHJ
``main.*`` — the ``task.item_meta`` / ``task.region_meta`` dicts written by
``_item_utils.register_item`` / ``register_zone`` are the only runner
interface. Layout uses the GLOBAL numpy RNG (seeded by RoboTwin's
``_init_task_env_``, PLAN v3 §2). Global-RNG draw order in load_actors:
container pose (rejection loop) -> container id -> target id -> target pose
-> source bowl mass -> [bimanual: target mass] -> bead grid jitter ->
bead masses -> pour-offset side; play_once then draws
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
  ``rim_height`` stays the measured rim plane.

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
- ``pour_offset`` (0.0 m): lateral offset of the pour point toward a side drawn per episode.
- ``tilt_rate`` (2.0 rad/s): commanded pour angular rate — see the
  approximation note on _tilt_steps.

The global ``time_scale`` knob retimes the risky TRANSIT (risky() context
manager, RiskTickController); approach / grasp / place / retreat run at
``safe_time_scale`` like the host.
"""

import os

from ._base_task import Base_Task
from .place_container_plate import place_container_plate
from .utils import *
import sapien

from ._item_utils import (
    APPROACH_TO_TCP,
    bead_grid_positions,
    bead_random_positions,
    bowl_bead_capacity,
    bowl_geometry,
    container_region_radius,
    min_pour_height,
    register_item,
    register_zone,
    risky,
    spawn_beads,
)

# --- scene layout (must match main/item_envs/configs/tasks/T2_granular_pour.yaml;
#     tests/item_envs/test_granular_pour.py source-greps this contract) ---
BOWL_MODEL = "002_bowl"  # the TARGET bowl (deep: catches the pour)
SRC_BOWL_MODEL = "002_bowl"  # the SOURCE bowl ...
SRC_INTERIOR_DEPTH = 0.025  # ... with a raised floor: rim 2.5 cm above it (create_bowl_raised_floor; user review 5: a low container spills at small tilts and in transit)
SRC_FILL_FRAC = 0.65  # low source: beads = this x its random-pack rim capacity (capped at N_BEADS); 0.8 put the surface where the pinch's inner finger enters (T4 d46)
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


# --- pour motion ---
POUR_HEIGHT_DEFAULT = 0.10  # m above the target rim
POUR_OFFSET_DEFAULT = 0.0  # m, lateral, toward a side drawn per episode
POUR_HEIGHT_CLEAR = 0.015  # effective pour height >= src outer radius + this
                           # (min_pour_height — the rim rings cannot interlink)
TILT_RATE_DEFAULT = 2.0  # rad/s
GRASP_CONTACT_IDS = (0, 2)  # [right, left] 002_bowl contact points (host idiom); per-embodiment override in the yaml
TILT_ANGLE_DEG = (100.0, 120.0)  # sampled per episode (P2 probed 110 deg)
TILT_STEPS = 10  # small EE-rotation steps per tilt direction
# Tilt rate: each step sets task.plan_duration_target = |step| / tilt_rate and
# RiskTickController retimes the step's plan to exactly that duration, so the
# realized rate is the commanded one for cuRobo and mplib-screw plans alike
# (v_ratio > 1 events are logged, never clipped).
POUR_PIN_ABORT = 0.03  # abort the tilt when the rim-to-EE distance drifts this
                       # far from its grasp-time value (bowl slipping / lost)
SWING_CLEAR = 0.03  # m: source-bowl hull above the target rim during transit and tilt swing
POUR_REF_DEG = 85.0  # mid-pour angle (100 beads start leaving at ~55-60 deg) at which the lip sits on the pour point
PIVOT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # pivot = lip + alpha * (EE - lip); smallest executable wins
ROLL_AIM_TILT_DEG = 35.0  # wrist-roll pour: the exit lip is placed over the target at THIS bowl tilt (pour onset of a low bowl)
ROLL_PITCH_DEG = 30.0  # wrist-roll pour: approach pitched this far below horizontal (max bowl tilt 180 - 2 x this)
HOLD_DELAY = 1  # self.delay units at full tilt (~1.2 s: 300 physics steps @250 Hz)
PLACE_DX = 0.17  # the empty bowl lands this far outward (+/- x) of the target
GRASP_DIS = 0.0  # m: grasp target short of the contact point along the approach (embodiment_overrides)
GRASP_RADIAL_OUT = 0.0  # m: grasp target shifted radially outward (embodiment_overrides)
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
        src_model = os.environ.get("ITEM_SRC_MODEL", SRC_BOWL_MODEL)  # diag-only override
        self.src_model = src_model
        self.actor_name = src_model
        self.container_id = int(np.random.choice(list(BOWL_IDS)))
        depth = float(os.environ.get("ITEM_SRC_DEPTH", SRC_INTERIOR_DEPTH))  # diag-only override (0 = plain bowl)
        self.container, self._src_geom = create_bowl_raised_floor(
            self, container_pose, src_model, self.container_id, depth)
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
        n_beads = int(os.environ.get("ITEM_T2_N", N_BEADS))  # diag-only override
        if self._src_geom.get("raised_floor"):
            # low (raised-floor) source: a random pack (a hex lattice settles into a
            # crystal that holds far past the angle of repose, T4 diag d21-d32)
            # filled to SRC_FILL_FRAC of the rim capacity
            n_beads = min(n_beads, int(round(SRC_FILL_FRAC * bowl_bead_capacity(self._src_geom, BEAD_R))))
            local = bead_random_positions(n_beads, BEAD_R, self._src_geom, np.random)
        else:
            local = bead_grid_positions(
                n_beads,
                BEAD_R,
                self._src_geom["inner_radius"],
                self._src_geom["floor_y"],
                bands=self._src_geom["bands"],
                rng=np.random,
            )
        world_T = self.container.get_pose().to_transformation_matrix()
        bead_world = local @ world_T[:3, :3].T + world_T[:3, 3]
        masses = np.random.uniform(BEAD_MASS_RANGE[0], BEAD_MASS_RANGE[1], size=n_beads)
        self.beads = spawn_beads(self.scene, bead_world, BEAD_R, masses, color=BEAD_COLOR)
        self._bead_names = {b.get_name() for b in self.beads}

        # 4. (user decision 2026-10-03: no forbidden mats — the only safety
        #    constraint is "spill no bead"). The pour_offset knob shifts the
        #    pour point sideways toward a random side.
        self.zones = []
        self._pour_lat_sign = 1.0 if np.random.uniform() < 0.5 else -1.0

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

        eparams_ = getattr(self, "embodiment_params", None) or {}
        if knobs.get("pour_style", eparams_.get("pour_style")) == "wrist_roll":  # knob: diag override
            return self._play_wrist_roll(arm_tag, knobs, eparams_, pour_height, pour_offset, tilt_rate,
                                         container_pose)

        # 1. grasp the source bowl — safe scale. Contact ids per embodiment
        #    (embodiment_params.grasp_contact_ids = [right, left]; host idiom
        #    [0, 2]). 002_bowl's ids 1/3 grasp the same rim point with the
        #    gripper turned 180 deg about vertical, which mirrors how the
        #    far-lip tilt maps onto the wrist joints (ur5-wsg wrist 1 is
        #    limited to +/-1.75 rad and sits at -1.54 with ids 0/2).
        eparams = getattr(self, "embodiment_params", None) or {}
        cp_ids = list(eparams.get("grasp_contact_ids", GRASP_CONTACT_IDS))
        # optional firmer pinch (embodiment_params.gripper_stiffness): the
        # rim pinch is a gripper PD drive (RoboTwin default stiffness 1000);
        # ur5-wsg can only pour over a side lip (wrist 1 blocks the far lip,
        # t2_pour_feasibility probe), tilting about the pinch line, where the
        # loaded bowl pivots out of a soft grasp (diag jobs 22636542/22636635)
        k_grip = eparams.get("gripper_stiffness")
        if k_grip is not None:
            for joints, damping in ((self.robot.left_gripper, self.robot.left_gripper_damping),
                                    (self.robot.right_gripper, self.robot.right_gripper_damping)):
                for joint in joints:
                    joint[0].set_drive_property(stiffness=float(k_grip), damping=float(damping))
        # shallow, wall-centred rim pinch (embodiment_params grasp_dis /
        # grasp_radial_out, as T4): the default pinch closed its inner finger
        # into the full bead pile and ejected most beads at the grasp (ur5,
        # diag d34: 73 / 100 beads on the table before the pour)
        g_dis = float(knobs.get("grasp_dis", eparams.get("grasp_dis", GRASP_DIS)))
        g_out = float(knobs.get("grasp_radial_out", eparams.get("grasp_radial_out", GRASP_RADIAL_OUT)))
        self.grip_info = {"grasp_contact_ids": cp_ids, "gripper_stiffness": k_grip,
                          "grasp_dis": g_dis, "grasp_radial_out": g_out}
        g_tag, g_acts = self.grasp_actor(
            self.container,
            arm_tag=arm_tag,
            contact_point_id=int(cp_ids[int(arm_tag == "left")]),
            pre_grasp_dis=0.1,
            grasp_dis=g_dis,
        )
        if g_out and g_acts:
            c = np.asarray(self.container.get_pose().p, dtype=np.float64)
            for act in g_acts:
                if act.action == "move" and any(act.target_pose):
                    u = np.asarray(act.target_pose[:2], dtype=np.float64) - c[:2]
                    u /= max(float(np.linalg.norm(u)), 1e-9)
                    act.target_pose[0] += g_out * float(u[0])
                    act.target_pose[1] += g_out * float(u[1])
        self.move((g_tag, g_acts))
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

        # pour point: above the target rim, offset sideways (live poses, so
        # the bimanual held target is accounted for)
        pour_point = self._pour_point(pour_height, pour_offset)

        # 4's parameters first (the transit target depends on them): tilt
        # 100-120 deg so the bowl's mouth tips toward a horizontal direction m
        # (rotation axis z x m). Preferred m points from the grasp side to the
        # far lip (toward the target); the sideways lips (-y, +y) are the
        # fallbacks — never the near lip, which sits under the fingers.
        tilt_total = float(np.deg2rad(np.random.uniform(*TILT_ANGLE_DEG)))
        sign = -1.0 if container_pose[0] > 0 else 1.0
        mouth, axis, ee_target, pivot = self._choose_pour(arm_tag, pour_point, sign, tilt_total)

        # 3. transit — risky scale (time_scale knob): the chosen pouring lip
        #    goes onto the pour point; the tilt then pivots about that lip
        #    (pinned), so the beads leave from one spot over the target.
        with risky(self):
            for leg in self._transit_legs(arm_tag, ee_target):
                self.move(self.move_by_displacement(arm_tag, x=float(leg[0]), y=float(leg[1]), z=float(leg[2])))

        # 4. tilt, hold, tilt back — risky; per-step scale from tilt_rate
        with risky(self):
            self._tilt_steps(arm_tag, axis, tilt_total, tilt_rate, pivot)
            self.delay(HOLD_DELAY)
            self._tilt_steps(arm_tag, axis, -tilt_total, tilt_rate, pivot)

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

    def _play_wrist_roll(self, arm_tag, knobs, eparams, pour_height, pour_offset, tilt_rate, container_pose):
        """Pour by a roll about the gripper's approach axis
        (embodiment_params.pour_style = "wrist_roll"; ur5-wsg, user review 5).

        ur5-wsg's wrist 1 (+/-1.75 rad) blocks every top-down-grasp tilt but a
        side-lip one, whose plans failed or flung the bowl (diag d37 / d39).
        Here the rim is pinched with the approach axis a along the rim
        TANGENT, pitched ROLL_PITCH_DEG down (a horizontal approach put the
        wrist at table height — unreachable, d43); the fingers straddle the
        wall and close radially. The pour is a rotation about a — through the
        EE origin AND the TCP, i.e. essentially a wrist-3 roll, which has no
        practical limit. With a pitched axis the bowl tips at most
        180 - 2 x pitch deg (a half-turn); the roll angle for the sampled tilt
        is solved numerically, and the pinch is placed so the FAR lip (where
        the beads leave) ends over the target after the roll."""
        g = self._src_geom
        side = 1.0 if str(arm_tag) == "right" else -1.0
        beta = np.deg2rad(float(eparams.get("roll_pitch_deg", ROLL_PITCH_DEG)))
        u = np.array([side, 0.0, 0.0])  # pinch on the bowl's outer (arm-side) rim, closing radially
        a = np.array([0.0, np.cos(beta), -np.sin(beta)])  # approach: from the robot side, along the tangent, pitched down
        R0 = np.column_stack([a, u, np.cross(a, u)])  # EE x = approach, EE y = finger closing axis
        q0 = list(_mat_to_quat_wxyz(R0))
        r_pinch = 0.5 * (float(g["inner_radius"]) + float(g["outer_radius"]))

        k_grip = eparams.get("gripper_stiffness")
        if k_grip is not None:
            for joints, damping in ((self.robot.left_gripper, self.robot.left_gripper_damping),
                                    (self.robot.right_gripper, self.robot.right_gripper_damping)):
                for joint in joints:
                    joint[0].set_drive_property(stiffness=float(k_grip), damping=float(damping))

        rim = self._rim_center(self.container, g)
        pinch = rim + u * r_pinch - np.array([0.0, 0.0, float(eparams.get("roll_pinch_depth", 0.008))])
        ee_grasp = pinch - a * APPROACH_TO_TCP
        self.grip_info = {"pour_style": "wrist_roll", "pitch_deg": float(np.rad2deg(beta)),
                          "pinch": pinch.tolist(), "gripper_stiffness": k_grip}
        # 1. tangent rim grasp — safe: pre-grasp 10 cm back along a, slide in, close
        self.move(self.open_gripper(arm_tag))
        self.move(self.move_to_pose(arm_tag, (ee_grasp - a * 0.10).tolist() + q0))
        self.move(self.move_to_pose(arm_tag, ee_grasp.tolist() + q0))
        self.move(self.close_gripper(arm_tag))
        # 2. lift — safe (world z)
        self.move(self.move_by_displacement(arm_tag, z=0.10))

        # the roll: sign sends the far lip DOWN; angle for the sampled tilt
        tilt_total = float(np.deg2rad(np.random.uniform(*TILT_ANGLE_DEG)))
        far0 = -2.0 * r_pinch * u  # far lip relative to the pinch (bowl upright)
        sgn = 1.0 if (_rot_axis(a, np.deg2rad(10.0)) @ far0)[2] < 0.0 else -1.0
        phis = np.deg2rad(np.arange(0.0, 180.5, 0.5))
        tilts = np.array([np.arccos(np.clip((_rot_axis(a, sgn * ph) @ np.array([0.0, 0.0, 1.0]))[2], -1.0, 1.0))
                          for ph in phis])
        phi_total = float(phis[int(np.argmin(np.abs(tilts - tilt_total)))])
        # aim: the exit lip = the LOWEST rim point (a pitched roll swings the
        # mouth sideways — the diametrically far lip is not the lowest) at the
        # pour ONSET tilt goes over the target: a low (raised-floor) bowl
        # empties from ~30-40 deg on, long before the final tilt (diag d45:
        # aiming at the final tilt put the beads 20-60 cm off target)
        aim_tilt = min(np.deg2rad(float(eparams.get("roll_aim_tilt_deg", ROLL_AIM_TILT_DEG))), tilt_total)
        phi_aim = float(phis[int(np.argmin(np.abs(tilts - aim_tilt)))])
        rim_c = rim - pinch  # rim centre relative to the pinch
        ths = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
        ring = np.stack([rim_c + float(g["inner_radius"]) * np.array([np.cos(t), np.sin(t), 0.0]) for t in ths])
        ring_aim = ring @ _rot_axis(a, sgn * phi_aim).T
        far_end = ring_aim[int(np.argmin(ring_aim[:, 2]))]  # exit lip relative to the pinch at the aim tilt

        # 3. transit — risky: put the pinch where the far lip ends over the
        #    target centre (+ lateral pour_offset), pour_height above its rim
        t_rim = self._rim_center(self.target, self._target_geom)
        lat = float(getattr(self, "_pour_lat_sign", 1.0)) * float(pour_offset)
        lip_tgt = t_rim + np.array([0.0, lat, float(pour_height)])
        pinch_tgt = lip_tgt - far_end
        ee = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)
        d = pinch_tgt - (ee[:3] + a * APPROACH_TO_TCP)
        with risky(self):
            self.move(self.move_by_displacement(arm_tag, z=float(max(d[2], 0.0))))
            self.move(self.move_by_displacement(arm_tag, x=float(d[0]), y=float(d[1])))
            if d[2] < 0.0:
                self.move(self.move_by_displacement(arm_tag, z=float(d[2])))

        # 4. roll about a — risky, duration-targeted at tilt_rate (rad of
        #    BOWL tilt per second: the roll step is scaled by dphi / dtilt)
        ee0 = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)
        step = np.deg2rad(5.0)
        n = max(1, int(np.ceil(phi_total / step)))
        dt_step = max((tilt_total / n) / max(float(tilt_rate), 1e-3), 0.04)  # >= one control tick
        self.pour_info = {"style": "wrist_roll", "phi_deg": float(np.rad2deg(phi_total)),
                          "tilt_deg": float(np.rad2deg(tilt_total)), "lip_target": lip_tgt.tolist()}

        def _roll(angles):
            self.plan_local = True
            try:
                for ang in angles:
                    q = _quat_mul_wxyz(_mat_to_quat_wxyz(_rot_axis(a, sgn * float(ang))), ee0[3:])
                    self.plan_duration_target = dt_step
                    self.move(self.move_to_pose(arm_tag, ee0[:3].tolist() + [float(v) for v in q]))
            finally:
                self.plan_local = False
                self.plan_duration_target = None

        with risky(self):
            _roll([phi_total * (k + 1) / n for k in range(n)])
            self.delay(HOLD_DELAY)
            _roll([phi_total * (n - 1 - k) / n for k in range(n)])

        # 5. place the empty bowl beside the target, release, back out — safe
        outward = 1.0 if container_pose[0] > 0 else -1.0
        place_xy = np.asarray(self._target_home[:2], dtype=np.float64) + np.array([outward * PLACE_DX, 0.0])
        bowl = np.asarray(self.container.get_pose().p, dtype=np.float64)
        self.move(self.move_by_displacement(arm_tag, x=float(place_xy[0] - bowl[0]), y=float(place_xy[1] - bowl[1])))
        bowl = np.asarray(self.container.get_pose().p, dtype=np.float64)
        self.move(self.move_by_displacement(arm_tag, z=-float(max(bowl[2] - (TABLE_Z + 0.005), 0.0))))
        self.move(self.open_gripper(arm_tag))
        self.move(self.move_by_displacement(arm_tag, x=float(-a[0] * 0.08), y=float(-a[1] * 0.08), z=float(-a[2] * 0.08)))
        self.move(self.move_by_displacement(arm_tag, z=0.08))

        self.info["info"] = {
            "{A}": f"002_bowl/base{self.target_id}",
            "{B}": f"{self.src_model}/base{self.container_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def _pour_point(self, pour_height, pour_offset):
        """World point the source bowl's rim center is brought to: the target's
        rim center + pour_height up + pour_offset sideways."""
        rim_world = self._rim_center(self.target, self._target_geom)
        lat = float(getattr(self, "_pour_lat_sign", 1.0)) * float(pour_offset)
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
        """(mouth, axis, ee_target, pivot) of the first executable pour.

        The tilt rotates the bowl about a PIVOT on the segment from the
        pouring lip (rim point on the ``mouth`` side) to the EE origin:
        ``pivot0 = lip + alpha * (ee - lip)`` (bowl upright, just lifted).

          alpha = 0  lip pinned: beads leave from one spot, but the EE swings
                     ~0.3 m — beyond ur5-wsg's wrist (dry runs: every
                     direction infeasible, diag job 22639777);
          alpha = 1  EE-origin pivot: a pure wrist rotation, but the lip
                     sweeps ~16 cm over the pour (2/100 beads landed, diag
                     job 22636182).

        The pivot is placed so the lip sits on ``pour_point`` at the
        reference angle POUR_REF_DEG (mid-pour); the lip arc shrinks with
        alpha. Search order: far lip (mouth toward the target — the only
        tilt whose axis is perpendicular to the rim pinch; side lips rotate
        about the pinch line and the bowl pivots out of the grasp, diag job
        22636635) for alpha ascending, then -y, +y. Each candidate gets
        ``_swing_clear`` and a dry run of the SEQUENCE play_once executes
        (transit legs + every tilt step, chained screws) with the
        runner-installed ``task.screw_path_ok``. Without the capability
        (vanilla RoboTwin) the far lip with alpha = 0 is used. Recorded in
        ``self.pour_info``.
        """
        ee = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)
        up = np.array([0.0, 0.0, 1.0])
        cands = [np.array([sign, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        check_path = getattr(self, "screw_path_ok", None)
        r_ref = float(np.deg2rad(POUR_REF_DEG)) * float(np.sign(tilt_total))
        tried, chosen = [], None
        for m in cands:
            axis = np.cross(up, m)
            lip0 = self._pour_lip(m)
            for alpha in PIVOT_ALPHAS:
                pivot0 = lip0 + float(alpha) * (ee[:3] - lip0)
                pivot = np.asarray(pour_point, dtype=np.float64) - _rot_axis(axis, r_ref) @ (lip0 - pivot0)
                pivot, lift = self._swing_clear(pivot0, pivot, axis, tilt_total)
                ee_target = pivot + (ee[:3] - pivot0)
                ok = True
                if callable(check_path):
                    chain, p = [], ee[:3].copy()
                    for leg in self._transit_legs(arm_tag, ee_target):
                        p = p + leg
                        chain.append(p.tolist() + list(ee[3:]))
                    off0 = ee_target - pivot
                    for k in range(1, TILT_STEPS + 1):
                        R = _rot_axis(axis, k * tilt_total / TILT_STEPS)
                        chain.append((pivot + R @ off0).tolist() + list(_quat_mul_wxyz(_mat_to_quat_wxyz(R), ee[3:])))
                    ok = bool(check_path(arm_tag, chain))
                tried.append({"mouth": m.tolist(), "alpha": float(alpha), "ok": ok, "swing_lift": round(lift, 4)})
                if chosen is None or ok:
                    chosen = (m, axis, ee_target, pivot, float(alpha))  # first feasible; else far lip, alpha 0
                if ok:
                    break
            if tried[-1]["ok"]:
                break
        self.pour_info = {"mouth": chosen[0].tolist(), "alpha": chosen[4], "tried": tried}
        print(f"[granular_pour] pour mouth {chosen[0].tolist()} alpha {chosen[4]} (dry runs {tried})")
        return chosen[:4]

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

    def _swing_clear(self, pivot0, pivot, axis, tilt_total, clearance=SWING_CLEAR):
        """(lifted pivot, lift): raise ``pivot`` until the source bowl's hull
        — translated so ``pivot0`` (the pivot's upright position) sits on
        ``pivot``, then rotated about it over the whole tilt — stays
        ``clearance`` above the target rim wherever it is horizontally over
        the target (a bowl swung through the static target was knocked out of
        the grasp and launched the beads, diag job 22636122 seed 2: 6688 N)."""
        pour_point = pivot
        offs = self._bowl_hull_world() - np.asarray(pivot0, dtype=np.float64)
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



def create_bowl_raised_floor(task, pose, modelname, model_id, depth, color=(0.62, 0.55, 0.45)):
    """002_bowl with a solid insert raising its interior floor so the rim
    sits only ``depth`` above it (user review 5: a LOW container spills at
    small tilts and needs velocity control). The OUTSIDE stays the full bowl
    — the rim pinch keeps its 5-7 cm of wall (a height-squashed bowl was too
    low to grasp: the fingers reach 3-4 cm below the TCP, diag d44). The
    insert is one cylinder in the bowl's file frame (y = up, origin at the
    exterior bottom), radius = the narrowest wall radius over its height
    minus 1 mm, rendered in a neutral colour. Returns (Actor, geometry dict
    with floor_y raised and the wall bands trimmed). depth <= 0 or >= the
    bowl's own depth: the plain bowl (RoboTwin's create_actor)."""
    import json
    from pathlib import Path

    geom = dict(bowl_geometry(modelname, model_id))
    floor0, rim = float(geom["floor_y"]), float(geom["rim_y"])
    raise_h = (rim - floor0) - float(depth)
    if depth <= 0.0 or raise_h <= 0.002:
        return create_actor(task, pose=pose, modelname=modelname, model_id=model_id, convex=True), geom
    from .utils.create_actor import get_glb_or_obj_file, preprocess
    from .utils.actor_utils import Actor

    scene, pose = preprocess(task, pose)
    modeldir = Path("assets/objects") / modelname
    model_data = json.loads((modeldir / f"model_data{int(model_id)}.json").read_text())
    scale = model_data["scale"]
    collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
    bands = [b for b in (geom.get("bands") or []) if b[0] < floor0 + raise_h]
    r_ins = (min(float(b[1]) for b in bands) if bands else float(geom["inner_radius"])) - 0.001
    # SAPIEN cylinders run along local x: rotate +90 deg about z -> along y (up)
    ins_pose = sapien.Pose([0.0, floor0 + 0.5 * raise_h, 0.0], [0.7071067811865476, 0.0, 0.0, 0.7071067811865476])
    builder.add_cylinder_collision(pose=ins_pose, radius=r_ins, half_length=0.5 * raise_h)
    builder.add_cylinder_visual(pose=ins_pose, radius=r_ins, half_length=0.5 * raise_h, material=tuple(color))
    entity = builder.build(name=modelname)
    entity.set_name(modelname)
    entity.set_pose(pose)
    geom["floor_y"] = floor0 + raise_h
    geom["bands"] = [b for b in (geom.get("bands") or []) if b[0] >= floor0 + raise_h]
    geom["raised_floor"] = raise_h
    return Actor(entity, model_data), geom

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
