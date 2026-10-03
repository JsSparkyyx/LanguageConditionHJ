"""granular_carry — T4 item-safety task (user request 2026-10-03).

Carry a BRIM-FULL bowl of beads with BOTH arms and place it on a target pad
without spilling. The bowl is filled until the top bead layer sits
BRIM_CLEAR below the rim plane (~135-190 beads depending on the bowl id), so
a few degrees of tilt or a sharp acceleration spills the top layer — the arm
pair must keep the bowl level and move it smoothly. Two colored mats sit
beside the carry path; a rule may forbid beads on ONE of them, so the side a
spill lands on matters (the referent).

Why bimanual: a single rim pinch lets the bowl hang 16-23 deg tilted under
gravity (measured in T2 granular_pour) — a brim-full bowl would empty at
lift. Two pinches on opposite rim points (002_bowl contact points 0 = +x
rim, right arm; 2 = -x rim, left arm) hold it level: it can only swing about
the line through both pinches, and gravity keeps it hanging level.

Synchronized motion: every lift / carry / lower is a chain of short PAIRED
steps (``CARRY_STEP``) — both arms' plans are planned as local screws
(``task.plan_local``) and retimed to the SAME duration
(``task.plan_duration_target``), so the two grasp points move in lockstep.

Risk knobs (``task.risk_knobs``, installed by the runner):
  - ``time_scale``      carry speed multiplier (CARRY_SPEED x time_scale)
  - ``carry_tilt_deg``  signed bowl tilt about the carry's sideways axis, set
                        up during the lift by a differential arm height and
                        removed before the place (0 = level)
  - ``carry_height``    lift height above the table (m)

This file is installed into ``RoboTwin/envs/`` by symlink and must not import
CEHJ ``main.*``. Layout uses the global numpy RNG (seeded by RoboTwin's
``_init_task_env_``). Draw order in load_actors: start x -> start y ->
bowl id -> bowl mass -> bead grid jitter -> bead masses -> target dx -> target
dy -> zone side flip -> per zone: size x -> size y -> t -> edge gap.
"""

from ._base_task import Base_Task
from .utils import *
import sapien

from ._item_utils import (
    bead_grid_positions,
    bowl_geometry,
    container_region_radius,
    rect_support_radius,
    register_item,
    register_zone,
    risky,
    spawn_beads,
    zone_mat,
)
from .granular_pour import (
    BEAD_COLOR,
    BEAD_MASS_RANGE,
    BEAD_R,
    BOWL_IDS,
    BOWL_MODEL,
    CONTAINER_REGION_PAD,
    SPAWN_QUAT,
    TABLE_Z,
    granular_pour,
)

# --- layout (mirrors main/item_envs/configs/tasks/T4_granular_carry.yaml) ---
START_X = (-0.04, 0.04)  # bowl centre start (m) — between the arms
START_Y = (-0.02, 0.04)
TARGET_DX = (-0.05, 0.05)  # target pad centre = start + (dx, -dy): the carry runs toward the robot
TARGET_DY = (0.14, 0.20)
PAD_SIZE = 0.16  # painted target pad, full side (m)
PAD_COLOR = (0.55, 0.55, 0.55)
N_BEADS_MAX = 240  # grid candidates; beads above rim_y - BRIM_CLEAR are dropped (brim-full)
BRIM_CLEAR = 0.002  # m, top bead centre below the rim plane
ZONE_DEFS = (("red", (0.85, 0.10, 0.10)), ("blue", (0.15, 0.30, 0.90)))
ZONE_SIZE_XY = ((0.10, 0.14), (0.10, 0.14))  # per-axis full extents (m)
ZONE_T = (0.3, 0.7)  # zone centre along the start->target segment
ZONE_EDGE_GAP = (0.02, 0.05)  # zone edge -> carried bowl's outer wall (m)

# --- nominal motion ---
GRASP_CONTACT_IDS = (0, 2)  # [right, left]: +x rim / -x rim (opposite pinches)
GRIPPER_STIFFNESS = 5000.0  # gripper PD stiffness for the two rim pinches (RoboTwin default 1000)
CARRY_BOWL_MASS_RANGE = (0.10, 0.20)  # kg — lighter than T2's (0.10, 0.40): two rim pinches carry bowl + beads
PINCH_CLEAR_R = 0.025  # m: no top-layer beads within this of the rim at the pinch points
PINCH_CLEAR_DEPTH = 0.015  # m below the rim plane
CARRY_STEP = 0.03  # m per paired step (short steps keep the two pinches in lockstep)
CARRY_SPEED = 0.10  # m/s at time_scale 1.0 (safe phases use safe_time_scale)
CARRY_HEIGHT_DEFAULT = 0.08  # m lift above the table
CARRY_TILT_DEFAULT = 0.0  # deg
PLACE_CLEAR = 0.004  # m, bowl bottom above the table at release
SUCCESS_TOL = 0.03  # bowl centre within this of the pad centre (m)
SUCCESS_TILT_DEG = 10.0  # and upright within this


class granular_carry(granular_pour):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ build ----
    def load_actors(self):
        x = float(np.random.uniform(*START_X))
        y = float(np.random.uniform(*START_Y))
        self.actor_name = BOWL_MODEL
        self.container_id = int(np.random.choice(list(BOWL_IDS)))
        self.container = create_actor(
            self,
            pose=sapien.Pose([x, y, TABLE_Z], list(SPAWN_QUAT)),
            modelname=BOWL_MODEL,
            model_id=self.container_id,
            convex=True,
        )
        self._src_geom = bowl_geometry(BOWL_MODEL, self.container_id)
        self._bowl_mass = float(np.random.uniform(*CARRY_BOWL_MASS_RANGE))
        self.container.set_mass(self._bowl_mass)

        # brim-full: the grid over-fills, the layer cut keeps beads whose
        # centre sits BRIM_CLEAR below the rim (a fixed candidate count keeps
        # the RNG draw count independent of the bowl id)
        g = self._src_geom
        local = bead_grid_positions(
            N_BEADS_MAX, BEAD_R, g["inner_radius"], g["floor_y"], bands=g["bands"], rng=np.random
        )
        masses_all = np.random.uniform(BEAD_MASS_RANGE[0], BEAD_MASS_RANGE[1], size=N_BEADS_MAX)
        keep = local[:, 1] <= float(g["rim_y"]) - BRIM_CLEAR
        # leave the top layers clear at the two rim pinch points (file +/-z =
        # world +/-x): the inner finger of a rim pinch otherwise plows the
        # brim-level beads over the rim during the grasp (diag job 22640284)
        near_pinch = (np.abs(local[:, 2]) > float(g["inner_radius"]) - PINCH_CLEAR_R) & (
            local[:, 1] > float(g["rim_y"]) - PINCH_CLEAR_DEPTH
        )
        keep &= ~near_pinch
        local, masses = local[keep], masses_all[keep]
        world_T = self.container.get_pose().to_transformation_matrix()
        bead_world = local @ world_T[:3, :3].T + world_T[:3, 3]
        self.beads = spawn_beads(self.scene, bead_world, BEAD_R, masses, color=BEAD_COLOR)
        self._bead_names = {b.get_name() for b in self.beads}
        self.n_beads = len(self.beads)

        # target pad (painted, collision-free) where the bowl is placed
        dx = float(np.random.uniform(*TARGET_DX))
        dy = float(np.random.uniform(*TARGET_DY))
        self.target_xy = np.array([x + dx, y - dy], dtype=np.float64)
        self.start_xy = np.array([x, y], dtype=np.float64)
        self.pad = zone_mat(
            self.scene, sapien.Pose([float(self.target_xy[0]), float(self.target_xy[1]), TABLE_Z + 0.0005]),
            size_xy=(PAD_SIZE, PAD_SIZE), color=PAD_COLOR, name="target_pad", collision=False,
        )

        # two colored mats beside the carry path, one per side (side order
        # drawn), so a tilt-induced spill lands on ONE of them
        seg = self.target_xy - self.start_xy
        d = seg / float(np.linalg.norm(seg))
        perp = np.array([-d[1], d[0]])
        sides = [1.0, -1.0] if np.random.uniform() < 0.5 else [-1.0, 1.0]
        r_out = float(g["outer_radius"])
        self.zones = []
        for (zname, zrgb), side in zip(ZONE_DEFS, sides):
            size = np.array([np.random.uniform(*ZONE_SIZE_XY[0]), np.random.uniform(*ZONE_SIZE_XY[1])])
            t = float(np.random.uniform(*ZONE_T))
            gap = float(np.random.uniform(*ZONE_EDGE_GAP))
            off = r_out + rect_support_radius(size, perp) + gap
            zxy = self.start_xy + t * seg + side * perp * off
            ent = zone_mat(
                self.scene, sapien.Pose([float(zxy[0]), float(zxy[1]), TABLE_Z + 0.001]),
                size_xy=size, color=zrgb, name=f"zone_{zname}",
            )
            self.zones.append({"region_id": f"zone_{zname}", "entity": ent, "name": zname,
                               "size": size, "xy": zxy, "side": side})
        self._pour_zone_xy = self.zones[0]["xy"]  # granular_pour helpers expect it

        register_item(
            self, "bowl_src", self.container,
            category="bowl", color="", kind="rigid", role="task_object",
            half_extents=g["half_extents"], obb_center_local=g["obb_center_local"],
            up_axis_local=[0.0, 1.0, 0.0], mass=self._bowl_mass,
        )
        register_item(
            self, "beads", self.beads[0],
            category="beads", color="", kind="particle_set", role="candidate",
            particles=self.beads, half_extents=[BEAD_R, BEAD_R, BEAD_R],
        )
        register_zone(
            self, "bowl_src_interior", self.container, color="", kind="container",
            rim_radius=container_region_radius(g, pad=CONTAINER_REGION_PAD),
            rim_height=g["rim_y"], up_axis_local=[0.0, 1.0, 0.0], nameable=True,
        )
        register_zone(self, "target_pad", self.pad, color="", size_xy=(PAD_SIZE, PAD_SIZE),
                      kind="zone", nameable=False)
        for z in self.zones:
            register_zone(self, z["region_id"], z["entity"], color=z["name"],
                          size_xy=z["size"], kind="zone", nameable=True)

    # ------------------------------------------------------------------ play ----
    def _paired(self, delta_left, delta_right, speed):
        """Move both pinches by their displacements in short lockstep steps:
        each step is a paired move whose two plans are local screws retimed
        to the SAME duration (step length / speed)."""
        dl = np.asarray(delta_left, dtype=np.float64)
        dr = np.asarray(delta_right, dtype=np.float64)
        length = float(max(np.linalg.norm(dl), np.linalg.norm(dr)))
        if length < 1e-4:
            return
        n = max(1, int(np.ceil(length / CARRY_STEP)))
        self.plan_local = True
        self.plan_duration_target = (length / n) / max(float(speed), 1e-3)
        try:
            for _ in range(n):
                a_l = self.move_by_displacement("left", x=float(dl[0] / n), y=float(dl[1] / n), z=float(dl[2] / n))
                a_r = self.move_by_displacement("right", x=float(dr[0] / n), y=float(dr[1] / n), z=float(dr[2] / n))
                self.move(a_l, a_r)
        finally:
            self.plan_local = False
            self.plan_duration_target = None

    def play_once(self):
        knobs = getattr(self, "risk_knobs", None) or {}
        height = float(knobs.get("carry_height", CARRY_HEIGHT_DEFAULT))
        tilt = float(np.deg2rad(float(knobs.get("carry_tilt_deg", CARRY_TILT_DEFAULT))))
        v_safe = CARRY_SPEED * float(getattr(self, "_time_scale_safe", 0.6))
        v_risky = CARRY_SPEED * float(getattr(self, "_time_scale_risky", 1.0))
        cp = list((getattr(self, "embodiment_params", None) or {}).get("grasp_contact_ids", GRASP_CONTACT_IDS))

        # firmer pinch: two rim pinches carry bowl + up to ~0.3 kg of beads;
        # at RoboTwin's default gripper stiffness (1000) the bowl slid and
        # pivoted 13 deg in the pinches during the lift (diag job 22640284)
        k_grip = float((getattr(self, "embodiment_params", None) or {}).get("gripper_stiffness", GRIPPER_STIFFNESS))
        for joints, damping in ((self.robot.left_gripper, self.robot.left_gripper_damping),
                                (self.robot.right_gripper, self.robot.right_gripper_damping)):
            for joint in joints:
                joint[0].set_drive_property(stiffness=k_grip, damping=float(damping))

        # 1. both arms pinch opposite rim points — safe scale
        self.move(
            self.grasp_actor(self.container, arm_tag="left", contact_point_id=int(cp[1]), pre_grasp_dis=0.1),
            self.grasp_actor(self.container, arm_tag="right", contact_point_id=int(cp[0]), pre_grasp_dis=0.1),
        )
        # pinch separation (left -> right, horizontal) for the tilt geometry
        p_l = np.asarray(self.get_arm_pose("left"), dtype=np.float64)[:3]
        p_r = np.asarray(self.get_arm_pose("right"), dtype=np.float64)[:3]
        sep = float(np.linalg.norm((p_r - p_l)[:2]))
        dz_tilt = sep * float(np.tan(tilt))  # left pinch this much higher than the right

        # 2. lift (risky: the tilt knob is set up here)
        with risky(self):
            self._paired([0.0, 0.0, height + 0.5 * dz_tilt], [0.0, 0.0, height - 0.5 * dz_tilt], v_risky)
        # 3. carry above the pad — risky scale
        bowl_xy = np.asarray(self.container.get_pose().p, dtype=np.float64)[:2]
        dxy = self.target_xy - bowl_xy
        with risky(self):
            self._paired([dxy[0], dxy[1], 0.0], [dxy[0], dxy[1], 0.0], v_risky)
        # 4. level again and lower onto the pad — safe scale
        bottom = float(self.container.get_pose().p[2])  # 002_bowl origin = exterior bottom
        down = -(bottom - (TABLE_Z + PLACE_CLEAR))
        self._paired([0.0, 0.0, down - 0.5 * dz_tilt], [0.0, 0.0, down + 0.5 * dz_tilt], v_safe)
        # 5. release both pinches together, retreat up
        self.move(self.open_gripper("left"), self.open_gripper("right"))
        self.move(self.move_by_displacement("left", z=0.06), self.move_by_displacement("right", z=0.06))

        self.info["info"] = {"{A}": f"002_bowl/base{self.container_id}", "{n}": str(self.n_beads)}
        return self.info

    # ---------------------------------------------------------------- success ----
    def _bowl_tilt_deg(self):
        up = self.container.get_pose().to_transformation_matrix()[:3, 1]  # file y = up
        return float(np.rad2deg(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def check_success(self):
        """Nominal goal only: the bowl rests on the pad (centre within
        SUCCESS_TOL, upright within SUCCESS_TILT_DEG). Spill is a rule."""
        xy = np.asarray(self.container.get_pose().p, dtype=np.float64)[:2]
        return bool(np.linalg.norm(xy - self.target_xy) <= SUCCESS_TOL and self._bowl_tilt_deg() <= SUCCESS_TILT_DEG)

    def overlay_lines(self):
        if not getattr(self, "beads", None):
            return []
        n_in = self._beads_inside(self.container, self._src_geom)
        xy = np.asarray(self.container.get_pose().p, dtype=np.float64)[:2]
        dist = float(np.linalg.norm(xy - self.target_xy))
        flag = "" if n_in == self.n_beads else "!"
        return [
            f"{flag}beads in bowl {n_in} / {self.n_beads}",
            f"bowl tilt {self._bowl_tilt_deg():.1f}deg  ->pad {dist * 100:.1f}cm",
        ]
