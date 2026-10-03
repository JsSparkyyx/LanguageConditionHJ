"""granular_carry — T4 item-safety task (user request 2026-10-03, revised).

Pick up a bowl FULL of beads with ONE arm (the arm on the bowl's side) and
place it on a target pad without spilling a bead. The bowl is filled until
the top bead centre sits BRIM_CLEAR below the rim, so a tilt of roughly
atan(BRIM_CLEAR / inner radius) ~ 12 deg or a sharp start/stop spills the top
layer; the nominal must keep the bowl level. The only safety rule is
"spill no bead" (particles_inside of the carried bowl).

Leveling: a single rim pinch lets the loaded bowl sag ~15-20 deg about the
pinch once it leaves the table (measured in T2). Right after lift-off the
nominal measures the bowl's up axis and rotates the EE about the fingertip
pinch (TCP) to bring it back to vertical (LEVEL_ITERS passes), then carries
with that orientation. All moves are short steps with absolute targets
interpolated from the step-chain start (a loaded arm lags each short step;
live-relative steps compound the lag), planned as local screws
(``task.plan_local``) retimed to step length / speed
(``task.plan_duration_target``).

Risk knobs (``task.risk_knobs``):
  - ``time_scale``      carry speed multiplier (CARRY_SPEED x time_scale)
  - ``carry_tilt_deg``  signed tilt applied after leveling (about the
                        horizontal axis across the carry), removed before
                        the lowering
  - ``carry_height``    lift height above the table (m)

This file is installed into ``RoboTwin/envs/`` by symlink and must not import
CEHJ ``main.*``. Global-RNG draw order in load_actors: side -> start |x| ->
start y -> bowl id -> bowl mass -> bead grid jitter -> bead masses -> target
dx -> target dy.
"""

from ._base_task import Base_Task
from .utils import *
import sapien

from ._item_utils import (
    APPROACH_TO_TCP,
    bead_grid_positions,
    bowl_geometry,
    container_region_radius,
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
    _mat_to_quat_wxyz,
    _quat_mul_wxyz,
    _rot_axis,
    granular_pour,
)

# --- layout (mirrors main/item_envs/configs/tasks/T4_granular_carry.yaml) ---
START_ABS_X = (0.15, 0.25)  # bowl centre |x| on its (randomly drawn) side
START_Y = (-0.06, 0.04)
TARGET_DX = (-0.06, 0.06)  # target pad centre = start + (dx, -dy), kept on the bowl's side
TARGET_DY = (0.12, 0.18)  # the carry runs toward the robot
TARGET_MIN_ABS_X = 0.10
PAD_SIZE = 0.16  # painted target pad, full side (m)
PAD_COLOR = (0.55, 0.55, 0.55)
N_BEADS_MAX = 240  # grid candidates; beads above rim_y - BRIM_CLEAR are dropped
BRIM_CLEAR = 0.012  # m, top bead centre below the rim plane: spills beyond ~12 deg of tilt
CARRY_BOWL_MASS_RANGE = (0.10, 0.20)  # kg
PINCH_CLEAR_R = 0.035  # m: no top-layer beads within this of the rim at the pinch point
PINCH_CLEAR_DEPTH = 0.020  # m below the rim plane

# --- nominal motion ---
GRASP_CONTACT_IDS = (0, 2)  # [right, left]: outer rim of the bowl (host idiom)
GRIPPER_STIFFNESS = 5000.0  # gripper PD stiffness (RoboTwin default 1000) — firmer rim pinch
LIFTOFF = 0.015  # m: first lift, then level
LEVEL_ITERS = 2  # leveling passes after lift-off
LEVEL_TOL_DEG = 1.0  # skip a leveling pass below this tilt
CARRY_STEP = 0.03  # m per step
CARRY_SPEED = 0.10  # m/s at time_scale 1.0 (safe phases use safe_time_scale)
CARRY_HEIGHT_DEFAULT = 0.08  # m lift above the table
CARRY_TILT_DEFAULT = 0.0  # deg
PLACE_CLEAR = 0.004  # m, bowl bottom above the table at release
PLACE_TOL = 0.008  # m: residual gap that triggers a correction step
SUCCESS_TOL = 0.03  # bowl centre within this of the pad centre (m)
SUCCESS_TILT_DEG = 10.0  # and upright within this


class granular_carry(granular_pour):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ build ----
    def load_actors(self):
        side = 1.0 if np.random.uniform() < 0.5 else -1.0
        x = side * float(np.random.uniform(*START_ABS_X))
        y = float(np.random.uniform(*START_Y))
        self.carry_arm = "right" if side > 0 else "left"
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

        g = self._src_geom
        local = bead_grid_positions(
            N_BEADS_MAX, BEAD_R, g["inner_radius"], g["floor_y"], bands=g["bands"], rng=np.random
        )
        masses_all = np.random.uniform(BEAD_MASS_RANGE[0], BEAD_MASS_RANGE[1], size=N_BEADS_MAX)
        keep = local[:, 1] <= float(g["rim_y"]) - BRIM_CLEAR
        # top layers clear at the pinch point (outer rim: file z sign = world
        # x sign of the bowl's side); the inner finger otherwise plows beads
        pinch_side = side  # cp 0 / 2 grasp the OUTER rim (+x for the right arm)
        near_pinch = (pinch_side * local[:, 2] > float(g["inner_radius"]) - PINCH_CLEAR_R) & (
            local[:, 1] > float(g["rim_y"]) - PINCH_CLEAR_DEPTH
        )
        keep &= ~near_pinch
        local, masses = local[keep], masses_all[keep]
        world_T = self.container.get_pose().to_transformation_matrix()
        bead_world = local @ world_T[:3, :3].T + world_T[:3, 3]
        self.beads = spawn_beads(self.scene, bead_world, BEAD_R, masses, color=BEAD_COLOR)
        self._bead_names = {b.get_name() for b in self.beads}
        self.n_beads = len(self.beads)

        dx = float(np.random.uniform(*TARGET_DX))
        dy = float(np.random.uniform(*TARGET_DY))
        tx = x + dx
        if abs(tx) < TARGET_MIN_ABS_X:
            tx = side * TARGET_MIN_ABS_X
        self.target_xy = np.array([tx, y - dy], dtype=np.float64)
        self.start_xy = np.array([x, y], dtype=np.float64)
        self.pad = zone_mat(
            self.scene, sapien.Pose([float(self.target_xy[0]), float(self.target_xy[1]), TABLE_Z + 0.0005]),
            size_xy=(PAD_SIZE, PAD_SIZE), color=PAD_COLOR, name="target_pad", collision=False,
        )
        self.zones = []

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

    # ------------------------------------------------------------------ play ----
    def _steps_to(self, arm, target_pose, speed):
        """Absolute-target step chain from the live pose to ``target_pose``
        (position AND orientation interpolated; slerp-free: orientation steps
        are small), local screws retimed to step length / speed."""
        p0 = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        p1 = np.asarray(target_pose, dtype=np.float64).reshape(7)
        if float(np.dot(p0[3:], p1[3:])) < 0.0:
            p1[3:] = -p1[3:]
        length = float(np.linalg.norm(p1[:3] - p0[:3]))
        ang = 2.0 * float(np.arccos(np.clip(abs(float(np.dot(p0[3:], p1[3:]))), -1.0, 1.0)))
        n = max(1, int(np.ceil(max(length / CARRY_STEP, ang / np.deg2rad(5.0)))))
        self.plan_local = True
        self.plan_duration_target = max(length / n / max(float(speed), 1e-3), 0.25)
        try:
            for k in range(1, n + 1):
                t = k / n
                q = (1.0 - t) * p0[3:] + t * p1[3:]
                q = q / float(np.linalg.norm(q))
                pos = p0[:3] + t * (p1[:3] - p0[:3])
                self.move(self.move_to_pose(arm, pos.tolist() + q.tolist()))
        finally:
            self.plan_local = False
            self.plan_duration_target = None

    def _tcp(self, ee):
        """Fingertip pinch point: APPROACH_TO_TCP along the EE x (approach) axis."""
        R = np.asarray(sapien.Pose(ee[:3].tolist(), ee[3:].tolist()).to_transformation_matrix())[:3, :3]
        return ee[:3] + APPROACH_TO_TCP * R[:, 0]

    def _rotated_about_tcp(self, ee, R):
        tcp = self._tcp(ee)
        p = tcp + R @ (ee[:3] - tcp)
        q = _quat_mul_wxyz(_mat_to_quat_wxyz(R), ee[3:])
        return np.concatenate([p, q])

    def _level(self, arm, speed):
        """Rotate the EE about the pinch so the bowl's up axis is vertical."""
        for _ in range(LEVEL_ITERS):
            up = np.asarray(self.container.get_pose().to_transformation_matrix())[:3, 1]  # file y = up
            ang = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
            if np.degrees(ang) < LEVEL_TOL_DEG:
                return
            axis = np.cross(up, [0.0, 0.0, 1.0])
            if float(np.linalg.norm(axis)) < 1e-9:
                return
            ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
            self._steps_to(arm, self._rotated_about_tcp(ee, _rot_axis(axis, ang)), speed)

    def play_once(self):
        knobs = getattr(self, "risk_knobs", None) or {}
        height = float(knobs.get("carry_height", CARRY_HEIGHT_DEFAULT))
        tilt = float(np.deg2rad(float(knobs.get("carry_tilt_deg", CARRY_TILT_DEFAULT))))
        v_safe = CARRY_SPEED * float(getattr(self, "_time_scale_safe", 0.6))
        v_risky = CARRY_SPEED * float(getattr(self, "_time_scale_risky", 1.0))
        eparams = getattr(self, "embodiment_params", None) or {}
        cp = list(eparams.get("grasp_contact_ids", GRASP_CONTACT_IDS))
        arm = self.carry_arm

        # firmer pinch for the loaded rim grasp
        k_grip = float(eparams.get("gripper_stiffness", GRIPPER_STIFFNESS))
        joints, damping = ((self.robot.left_gripper, self.robot.left_gripper_damping) if arm == "left"
                           else (self.robot.right_gripper, self.robot.right_gripper_damping))
        for joint in joints:
            joint[0].set_drive_property(stiffness=k_grip, damping=float(damping))

        # 1. grasp the outer rim with the arm on the bowl's side — safe scale
        self.move(self.grasp_actor(self.container, arm_tag=arm,
                                   contact_point_id=int(cp[int(arm == "left")]), pre_grasp_dis=0.1))
        # 2. lift off, then level the sagging bowl about the pinch
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        self._steps_to(arm, np.concatenate([ee[:3] + [0.0, 0.0, LIFTOFF], ee[3:]]), v_safe)
        self._level(arm, v_safe)
        # 3. rest of the lift + the knob tilt (about the axis across the carry)
        seg = self.target_xy - self.start_xy
        d = seg / float(np.linalg.norm(seg))
        tilt_axis = np.array([-d[1], d[0], 0.0])
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        up_pose = np.concatenate([ee[:3] + [0.0, 0.0, max(0.0, height - LIFTOFF)], ee[3:]])
        with risky(self):
            self._steps_to(arm, self._rotated_about_tcp(up_pose, _rot_axis(tilt_axis, tilt))
                           if abs(tilt) > 1e-6 else up_pose, v_risky)
        # 4. carry over the pad — risky scale
        bowl_xy = np.asarray(self.container.get_pose().p, dtype=np.float64)[:2]
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        dxy = self.target_xy - bowl_xy
        with risky(self):
            self._steps_to(arm, np.concatenate([ee[:3] + [dxy[0], dxy[1], 0.0], ee[3:]]), v_risky)
        # 5. undo the knob tilt (re-level) and lower onto the pad — safe scale
        self._level(arm, v_safe)
        for _ in range(3):
            gap = float(self.container.get_pose().p[2]) - (TABLE_Z + PLACE_CLEAR)  # origin = exterior bottom
            if gap <= PLACE_TOL:
                break
            ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
            self._steps_to(arm, np.concatenate([ee[:3] - [0.0, 0.0, gap], ee[3:]]), v_safe)
        # 6. release and retreat
        self.move(self.open_gripper(arm))
        self.move(self.move_by_displacement(arm, z=0.06))

        self.info["info"] = {"{A}": f"002_bowl/base{self.container_id}", "{a}": arm, "{n}": str(self.n_beads)}
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
