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

import os

from ._base_task import Base_Task
from .utils import *
import sapien

from ._item_utils import (
    APPROACH_TO_TCP,
    bead_grid_positions,
    bead_random_positions,
    bowl_bead_capacity,
    bowl_geometry,
    container_region_radius,
    register_item,
    register_zone,
    risky,
    spawn_beads,
    goal_frame,
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
START_ABS_X = (0.15, 0.28)  # bowl centre |x| on its (randomly drawn) side
START_Y = (-0.02, 0.08)  # farther back than the pad (user review 4: not next to the goal)
TARGET_ABS_X = (0.08, 0.30)  # pad centre |x|, on the bowl's side (the carrying arm's half)
TARGET_Y = (-0.24, -0.12)  # toward the robot; the bowl (outer r <= 7 cm) stays on the table
MIN_CARRY_DIST = 0.20  # m: start -> pad, resampled until at least this
LAYOUT_TRIES = 200
PAD_SIZE = 0.16  # target marker: hollow dark frame, outer side (m) — visual only
PAD_FRAME_W = 0.012  # frame strip width (m)
N_BEADS_MAX = 240  # grid candidates at BEAD_R (scaled by (BEAD_R / r)^3 for smaller beads); beads above the fill line are dropped
CARRY_BEAD_R = BEAD_R  # m: T4 bead radius (diag override ITEM_T4_BEAD_R); smaller than BEAD_R -> category "small_beads" (lighter per bead)
BRIM_CLEAR = 0.010  # m, min top-bead-centre gap below the rim plane (layer pitch 12.6 mm -> 11-19 mm by bowl id)
CARRY_MODES = ("table", "held")  # table: grasp the bowl from the table at t >= 0 (fill PACK_FILL); held: BRIM-full (HELD_PACK_FILL), grasped + levelled + held still BEFORE t = 0 and every bead spilled by then removed (user review 4, option 2)
CARRY_MODE_P = (0.3, 0.7)  # held is the cleanly avoidable regime (spills during the carry, not at the grasp)
HELD_PACK_FILL = 1.15  # held mode over-fills to this x the bowl's rim capacity (bowl_bead_capacity, random pack phi 0.58); the pre-roll spills the excess and leaves a brim-full bowl (1.4 x the lattice count left bowls 2 / 5 at 65-74 % of capacity, diag d38)
HELD_SETTLE_STEPS = 250  # physics steps (1 s) the levelled bowl is held still before t = 0
BEAD_PACK = "random"  # "lattice" (hex grid, settles into a crystal) | "random" (random sequential addition, settles into a random pack)
PACK_FILL = 1.1  # random pack: bead count = this x the lattice count at BRIM_CLEAR / DOME_H
DOME_H = 0.0  # m: extra fill height at the bowl centre (parabolic in radius, 0 at the wall) — a heaped bowl
CARRY_BOWL_MASS_RANGE = (0.10, 0.20)  # kg
PINCH_CLEAR_R = 0.030  # m: no top-layer beads within this of the rim at the pinch point (0.05 emptied the whole pinch-side half)
PINCH_CLEAR_DEPTH = 0.025  # m below the rim plane

# --- nominal motion ---
GRASP_CONTACT_IDS = (0, 2)  # [right, left]: outer rim of the bowl (host idiom)
GRASP_DIS = 0.02  # m: grasp target stops this short of the contact point along the approach (shallower rim pinch)
GRASP_RADIAL_OUT = 0.006  # m: grasp target shifted radially outward (centres the fingers on the wall)
GRIPPER_STIFFNESS = 5000.0  # gripper PD stiffness (RoboTwin default 1000) — firmer rim pinch
LIFTOFF = 0.015  # m: first lift, then level
LEVEL_ITERS = 2  # leveling passes after lift-off
LEVEL_TOL_DEG = 1.0  # skip a leveling pass below this tilt
CARRY_STEP = 0.03  # m per step
CARRY_SPEED = 0.25  # m/s at time_scale 1.0 (risky lift + carry: 0.125-0.325 m/s over time_scale 0.5-1.3; safe phases use safe_time_scale) — at 0.10 m/s a table-mode carry never spilled (diag d33)
CARRY_HEIGHT_DEFAULT = 0.08  # m lift above the table
CARRY_TILT_DEFAULT = 0.0  # deg
STEP_MIN_T = 0.25  # s: step-duration floor for the SAFE steps (lift-off, leveling, lowering)
SAFE_SPEED = 0.10  # m/s at time_scale 1.0 for the safe phases (x safe_time_scale 0.6 = 0.06 m/s, as validated d15-d33)
RISKY_STEP_MIN_T = 0.08  # s: floor for the risky lift / carry steps (0.25 capped the carry at 0.12 m/s) (diag knob step_min_t)
LEVEL_STEP_T = 0.5  # s per <= 5 deg leveling step (10 deg/s): a faster wrist turn slipped the loaded rim pinch (diag d26)
PLACE_CLEAR = 0.004  # m, bowl bottom above the table at release
PLACE_XY_TOL = 0.005  # m: re-centre over the pad after re-leveling when farther than this
PLACE_TOL = 0.008  # m: residual gap that triggers a correction step
SUCCESS_TOL = 0.03  # bowl centre within this of the pad centre (m)
SUCCESS_TILT_DEG = 10.0  # and upright within this


class granular_carry(granular_pour):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ build ----
    def load_actors(self):
        side = 1.0 if np.random.uniform() < 0.5 else -1.0
        self.carry_mode = CARRY_MODES[int(np.random.choice(len(CARRY_MODES), p=np.asarray(CARRY_MODE_P)))]
        self.carry_mode = os.environ.get("ITEM_T4_MODE", self.carry_mode)  # diag-only override
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
        bead_r = float(os.environ.get("ITEM_T4_BEAD_R", CARRY_BEAD_R))
        self._bead_r = bead_r
        n_cand = int(np.ceil(N_BEADS_MAX * (BEAD_R / bead_r) ** 3))
        local = bead_grid_positions(
            n_cand, bead_r, g["inner_radius"], g["floor_y"], bands=g["bands"], rng=np.random
        )
        masses_all = np.random.uniform(BEAD_MASS_RANGE[0], BEAD_MASS_RANGE[1], size=n_cand) * (bead_r / BEAD_R) ** 3
        # fill line: BRIM_CLEAR under the rim at the wall, heaped by DOME_H at the
        # centre (ITEM_T4_DOME_H: diag-only override for fill sweeps)
        dome_h = float(os.environ.get("ITEM_T4_DOME_H", DOME_H))
        rho = np.linalg.norm(local[:, [0, 2]], axis=1)
        y_cap = float(g["rim_y"]) - BRIM_CLEAR + dome_h * np.clip(1.0 - (rho / float(g["inner_radius"])) ** 2, 0.0, 1.0)
        keep = local[:, 1] <= y_cap
        # top layers clear at the pinch point (outer rim: file z sign = world
        # x sign of the bowl's side); the inner finger otherwise plows beads
        pinch_side = side  # cp 0 / 2 grasp the OUTER rim (+x for the right arm)
        pinch_r = float(os.environ.get("ITEM_T4_PINCH_R", PINCH_CLEAR_R))  # diag-only override
        near_pinch = (pinch_side * local[:, 2] > float(g["inner_radius"]) - pinch_r) & (
            local[:, 1] > float(g["rim_y"]) - PINCH_CLEAR_DEPTH
        )
        keep &= ~near_pinch
        local, masses = local[keep], masses_all[keep]
        pack = os.environ.get("ITEM_T4_PACK", BEAD_PACK)  # diag-only override
        if pack == "random":
            if self.carry_mode == "held":
                n_fill = int(round(bowl_bead_capacity(g, bead_r) * HELD_PACK_FILL))
            else:
                n_fill = int(round(len(local) * float(os.environ.get("ITEM_T4_PACK_N", PACK_FILL))))
            local = bead_random_positions(n_fill, bead_r, g, np.random)
            extra = max(0, n_fill - len(masses_all))
            masses_all = np.concatenate([masses_all, np.random.uniform(BEAD_MASS_RANGE[0], BEAD_MASS_RANGE[1], size=extra)])
            masses = masses_all[:n_fill]
        world_T = self.container.get_pose().to_transformation_matrix()
        bead_world = local @ world_T[:3, :3].T + world_T[:3, 3]
        sleep = os.environ.get("ITEM_T4_BEAD_SLEEP")  # diag-only override of the PhysX sleep threshold
        self.beads = spawn_beads(self.scene, bead_world, bead_r, masses, color=BEAD_COLOR,
                                 sleep_threshold=None if sleep is None else float(sleep))
        self._bead_names = {b.get_name() for b in self.beads}
        self.n_beads = len(self.beads)

        # pad on the bowl's side, toward the robot, >= MIN_CARRY_DIST away
        for _ in range(LAYOUT_TRIES):
            t = np.array([side * float(np.random.uniform(*TARGET_ABS_X)), float(np.random.uniform(*TARGET_Y))])
            if float(np.linalg.norm(t - np.array([x, y]))) >= MIN_CARRY_DIST:
                break
        self.target_xy = t.astype(np.float64)
        self.start_xy = np.array([x, y], dtype=np.float64)
        self.pad = goal_frame(self.scene, self.target_xy, TABLE_Z, PAD_SIZE, PAD_FRAME_W, name="target_pad")
        self.zones = []

        register_item(
            self, "bowl_src", self.container,
            category="carry_bowl", color="", kind="rigid", role="task_object",
            half_extents=g["half_extents"], obb_center_local=g["obb_center_local"],
            up_axis_local=[0.0, 1.0, 0.0], mass=self._bowl_mass,
        )
        register_item(
            self, "beads", self.beads[0],
            category="carry_beads" if self._bead_r >= BEAD_R - 1e-9 else "small_beads", color="",
            kind="particle_set", role="candidate",
            particles=self.beads, half_extents=[self._bead_r] * 3,
        )
        register_zone(
            self, "bowl_src_interior", self.container, color="", kind="container",
            rim_radius=container_region_radius(g, pad=CONTAINER_REGION_PAD),
            rim_height=g["rim_y"], up_axis_local=[0.0, 1.0, 0.0], nameable=True,
        )

    # ------------------------------------------------------------------ play ----
    def _steps_to(self, arm, target_pose, speed, min_t=STEP_MIN_T):
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
        self.plan_duration_target = max(length / n / max(float(speed), 1e-3), float(min_t))
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
            self._steps_to(arm, self._rotated_about_tcp(ee, _rot_axis(axis, ang)), speed, min_t=LEVEL_STEP_T)

    def _grasp_and_level(self, arm, v_safe, knobs, eparams):
        """Stiffen the pinch, grasp the outer rim, lift off, level."""
        cp = list(eparams.get("grasp_contact_ids", GRASP_CONTACT_IDS))
        # firmer pinch for the loaded rim grasp
        k_grip = float(knobs.get("gripper_stiffness", eparams.get("gripper_stiffness", GRIPPER_STIFFNESS)))
        joints, damping = ((self.robot.left_gripper, self.robot.left_gripper_damping) if arm == "left"
                           else (self.robot.right_gripper, self.robot.right_gripper_damping))
        for joint in joints:
            joint[0].set_drive_property(stiffness=k_grip, damping=float(damping))

        # 1. grasp the outer rim with the arm on the bowl's side — safe scale
        # (grasp_dis / grasp_radial_out: risk-knob override for sweeps, else the
        # embodiment override, else the module default)
        g_dis = float(knobs.get("grasp_dis", eparams.get("grasp_dis", GRASP_DIS)))
        g_out = float(knobs.get("grasp_radial_out", eparams.get("grasp_radial_out", GRASP_RADIAL_OUT)))
        arm_tag, acts = self.grasp_actor(self.container, arm_tag=arm,
                                         contact_point_id=int(cp[int(arm == "left")]),
                                         pre_grasp_dis=0.1, grasp_dis=g_dis)
        if g_out and acts:
            c = np.asarray(self.container.get_pose().p, dtype=np.float64)
            for act in acts:
                if act.action == "move" and any(act.target_pose):
                    u = np.asarray(act.target_pose[:2], dtype=np.float64) - c[:2]
                    u /= max(float(np.linalg.norm(u)), 1e-9)
                    act.target_pose[0] += g_out * float(u[0])
                    act.target_pose[1] += g_out * float(u[1])
        self.move((arm_tag, acts))
        # 2. lift off, then level the sagging bowl about the pinch
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        self._steps_to(arm, np.concatenate([ee[:3] + [0.0, 0.0, LIFTOFF], ee[3:]]), v_safe)
        self._level(arm, v_safe)

    def _remove_spilled_beads(self):
        """Delete every bead not inside the bowl's INNER wall (spilled, or
        perched on the rim lip); the bead list is shared with the registered
        particle set, so it is edited in place."""
        T = np.asarray(self.container.get_pose().to_transformation_matrix(), dtype=np.float64)
        g = self._src_geom
        keep, gone = [], []
        for b in self.beads:
            p_loc = T[:3, :3].T @ (np.asarray(b.get_pose().p, dtype=np.float64) - T[:3, 3])
            rho = float(np.hypot(p_loc[0], p_loc[2]))  # file y = up
            inside = rho <= float(g["inner_radius"]) - 0.5 * self._bead_r and p_loc[1] > float(g["floor_y"]) - 0.005
            (keep if inside else gone).append(b)
        for b in gone:
            self.scene.remove_entity(b)
        self.beads[:] = keep
        # two steps flush PhysX's contact buffers of the removed bodies (reading
        # stale contacts segfaults the gripper-contact query, diag d33)
        for _ in range(2):
            self.scene.step()
        self._bead_names = {b.get_name() for b in self.beads}
        self.n_beads = len(self.beads)
        self.n_beads_removed = len(gone)

    def prepare_episode(self):
        """Runner hook, BEFORE t = 0 (nothing recorded). held mode: grasp,
        lift off and level the over-filled bowl, hold it still, then delete
        every bead that left it — the episode starts with a brim-full,
        levelled bowl in hand, and every spill is caused by the carry."""
        if getattr(self, "carry_mode", "table") != "held":
            return
        knobs = getattr(self, "risk_knobs", None) or {}
        eparams = getattr(self, "embodiment_params", None) or {}
        v_safe = SAFE_SPEED * float(getattr(self, "_time_scale_safe", 0.6))
        self._grasp_and_level(self.carry_arm, v_safe, knobs, eparams)
        for _ in range(HELD_SETTLE_STEPS):
            self.scene.step()
        self._remove_spilled_beads()
        self._prerolled = True

    def play_once(self):
        knobs = getattr(self, "risk_knobs", None) or {}
        height = float(knobs.get("carry_height", CARRY_HEIGHT_DEFAULT))
        tilt = float(np.deg2rad(float(knobs.get("carry_tilt_deg", CARRY_TILT_DEFAULT))))
        v_safe = SAFE_SPEED * float(getattr(self, "_time_scale_safe", 0.6))
        # carry_speed / step_min_t: diag knobs (faster nominal carry)
        v_risky = float(knobs.get("carry_speed", CARRY_SPEED)) * float(getattr(self, "_time_scale_risky", 1.0))
        t_min = float(knobs.get("step_min_t", RISKY_STEP_MIN_T))
        eparams = getattr(self, "embodiment_params", None) or {}
        arm = self.carry_arm

        if not getattr(self, "_prerolled", False):  # table mode: grasp from the table
            self._grasp_and_level(arm, v_safe, knobs, eparams)
        # 3. rest of the lift + the knob tilt (about the axis across the carry)
        seg = self.target_xy - self.start_xy
        d = seg / float(np.linalg.norm(seg))
        tilt_axis = np.array([-d[1], d[0], 0.0])
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        up_pose = np.concatenate([ee[:3] + [0.0, 0.0, max(0.0, height - LIFTOFF)], ee[3:]])
        with risky(self):
            self._steps_to(arm, self._rotated_about_tcp(up_pose, _rot_axis(tilt_axis, tilt))
                           if abs(tilt) > 1e-6 else up_pose, v_risky, min_t=t_min)
        # 4. carry over the pad — risky scale
        bowl_xy = np.asarray(self.container.get_pose().p, dtype=np.float64)[:2]
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        dxy = self.target_xy - bowl_xy
        with risky(self):
            self._steps_to(arm, np.concatenate([ee[:3] + [dxy[0], dxy[1], 0.0], ee[3:]]), v_risky, min_t=t_min)
        # 5. undo the knob tilt (re-level) and lower onto the pad — safe scale;
        # leveling rotates about the pinch and shifts the bowl: re-centre it
        self._level(arm, v_safe)
        err = self.target_xy - np.asarray(self.container.get_pose().p, dtype=np.float64)[:2]
        if float(np.linalg.norm(err)) > PLACE_XY_TOL:
            ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
            self._steps_to(arm, np.concatenate([ee[:3] + [err[0], err[1], 0.0], ee[3:]]), v_safe)
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
        # spilled = outside the rule's footprint column (outer radius + pad) —
        # the same criterion as particles_inside (heaped beads above the rim
        # of a brim-full held bowl still count as in the bowl)
        T = np.asarray(self.container.get_pose().to_transformation_matrix(), dtype=np.float64)
        r_col = container_region_radius(self._src_geom, pad=CONTAINER_REGION_PAD)
        n_out = 0
        for b in self.beads:
            p_loc = T[:3, :3].T @ (np.asarray(b.get_pose().p, dtype=np.float64) - T[:3, 3])
            n_out += int(np.hypot(p_loc[0], p_loc[2]) > r_col)  # file y = up
        xy = T[:2, 3]
        dist = float(np.linalg.norm(xy - self.target_xy))
        flag = "" if n_out == 0 else "!"
        return [
            f"{flag}beads spilled {n_out} / {self.n_beads}  ({getattr(self, 'carry_mode', 'table')} mode)",
            f"bowl tilt {self._bowl_tilt_deg():.1f}deg  ->pad {dist * 100:.1f}cm",
        ]
