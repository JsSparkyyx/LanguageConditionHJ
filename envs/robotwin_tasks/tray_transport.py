"""tray_transport — T5 item-safety task: SPEED is the hazard (user review 9).

ONE arm (the one on the tray's side) drags a tray with 2-3 standing
blocks SIDEWAYS along the table, toward the table centre, to a target frame
20-26 cm away (user review 10: dragged toward the robot, a block knocked by
the brake fell toward the arm and could rest on it — sideways it falls clear). The tray stays
on the table (no lifted load hanging from a pinch): the gripper pinches a
vertical handle tab protruding from the tray's robot-side rim and moves it in ONE
continuous trajectory (controller via path) — a smooth start, a cruise at
TRAY_SPEED x time_scale, and a brake within TRAY_STOP_DIST at the pad.

Why speed: a standing block of base width w with its centre of mass at h/2
tips once the tray decelerates past a_tip = g w / h (the inertial force at
the CoM out-torques gravity about the base edge). The brake's peak
deceleration is 0.75 v^2 / TRAY_STOP_DIST, so it scales with the SQUARE of
the cruise speed v: 0.15 m/s -> 0.84 m/s^2 (nothing tips), 0.39 m/s ->
5.7 m/s^2 (the tall blocks, a_tip 2.5-3.5 m/s^2, tip; the short ones,
a_tip >= 7 m/s^2, do not). The level tray never tilts: tilt plays no role.
(Slim CYLINDERS were tried first: PhysX lets a cylinder standing on its
flat end rock — a static tray let the tall one creep 3 -> 13 deg, d68.)

Rules: ``upright`` per standing block (theta_max UPRIGHT_MAX_DEG). The
primary (conditioned) rule is the upright of ONE referent: the tall
block in REFERENT_P["tall"] of the episodes (the nominal at speed
violates it — a filter must slow down before the pad), a short one
otherwise (benign: the nominal keeps the referent upright even when the
tall one falls — a filter must NOT slow down for it).

Primitive geometry (no meshes): the tray is one rigid body of boxes — floor
TRAY_SIZE x TRAY_FLOOR_T, rim walls TRAY_RIM_H tall, a handle tab
HANDLE_LEN (outward) x HANDLE_T x HANDLE_H on the near (-y) rim centre (a 2.5 cm rim
is too low to pinch top-down: the closed fingers reach 4.4 / 3.2 cm below
the TCP). Blocks: box primitives, square base, axis-aligned with the tray;
up axis local z.

Risk knobs (``task.risk_knobs``): ``time_scale`` only (the cruise speed).

This file is installed into ``RoboTwin/envs/`` by symlink and must not import
CEHJ ``main.*``. Global-RNG draw order in load_actors: side -> start |x| ->
start y -> tray mass -> third item? -> per item (w, h, mass) -> colors ->
row order -> gap split -> per item x -> referent kind -> pad (dx, dy).
"""

import os

from .utils import *
import sapien

from ._item_utils import (
    APPROACH_TO_TCP,
    _push_quat,
    goal_frame,
    register_item,
    risky,
)
from .granular_carry import granular_carry
from .granular_pour import TABLE_Z
from .push_zone import COLOR_RGB

# --- layout (mirrors main/item_envs/configs/tasks/T5_tray_transport.yaml) ---
START_ABS_X = (0.22, 0.30)  # tray centre |x| on its (randomly drawn) side
START_Y = (-0.06, 0.02)
CARRY_DX = (0.20, 0.26)  # m: the drag runs this far toward the table centre (along -side x) ...
CARRY_DY = (-0.02, 0.02)  # ... with this little y drift (sideways: blocks tip across the tray, away from the arm)
MIN_CARRY_DIST = 0.20  # m: start -> pad
LAYOUT_TRIES = 200
PAD_SIZE = 0.34  # target frame outer side (m), visual only (encloses the 20 x 30 cm tray)
PAD_FRAME_W = 0.012

# --- tray (primitive) ---
TRAY_SIZE = (0.20, 0.30)  # outer footprint x, y (m): long side along y, so the sideways drag crosses the short side
TRAY_FLOOR_T = 0.006  # floor plate thickness
TRAY_RIM_H = 0.025  # rim wall height above the floor top
TRAY_RIM_T = 0.006  # rim wall thickness
HANDLE_LEN = 0.07  # handle TAB: protrudes this far outward (-y) from the near rim, centred on it ...
HANDLE_T = 0.012  # ... thickness along x (the pinch closes across it: the hand's wide finger axis runs ALONG the rim, outside the tray) ...
HANDLE_H = 0.07  # ... top above the table (a fin ON the rim put the hand over the blocks: it knocked the tall one before the drag, d73)
PINCH_OUT = 0.045  # m: pinch point on the tab, this far out from the rim's outer face
TRAY_MASS_RANGE = (0.25, 0.45)  # kg (objects.yaml categories.tray.mass_range)
TRAY_COLOR = (0.55, 0.38, 0.22)  # wood

# --- standing items (primitive blocks, square base w x w, height h) ---
TALL_W = (0.026, 0.032)  # m
TALL_H = (0.15, 0.17)  # a_tip = g w / h = 1.5-2.1 m/s^2; tipping angle atan(w / h) 9-12 deg (w/h 0.3 rocked 2-4 deg under a 5.9 m/s^2, 0.1 s brake and stood, d70)
SHORT_W = (0.05, 0.06)
SHORT_H = (0.05, 0.07)  # a_tip >= 7.0 m/s^2
MID_W = (0.04, 0.05)
MID_H = (0.09, 0.11)  # a_tip 3.6-5.4 m/s^2
THIRD_ITEM_P = 0.5  # a medium block as well
ITEM_MASS_RANGE = (0.05, 0.15)  # kg (objects.yaml categories.block.mass_range)
ITEM_EDGE_CLEAR = 0.03  # m from the inner rim face to an item's side
ITEM_GAP = 0.03  # m between item sides
HANDLE_CLEAR = 0.05  # m: items keep this clear of the near rim (the pre-shaped inner finger; fully open it reached a block 4-5 cm in, d69)
EXCLUDE_COLORS = ("white", "black")  # white vanishes on the table; black on the dark frame
ITEM_COLORS = ("red", "green", "blue", "yellow")
REFERENT_P = {"tall": 0.75, "short": 0.25}  # primary-rule referent (short = benign scenario)
UPRIGHT_MAX_DEG = 10.0  # yaml rules: upright theta_max_deg — a tall block knocked by the brake comes to rest LEANING on a neighbour or the rim at 13-19 deg (d71); the cruise rocks it <= 6 deg

# --- nominal motion ---
TRAY_SPEED = 0.30  # m/s cruise at time_scale 1.0 (0.15-0.39 m/s over 0.5-1.3)
TRAY_ACCEL_DIST = 0.06  # m: smooth start (velocity smoothstep 0 -> v): peak accel 0.75 v^2 / this
TRAY_STOP_DIST = 0.02  # m: brake at the pad: peak decel 0.75 v^2 / this
SAFE_SPEED = 0.10  # m/s x safe_time_scale: approach / descend / retreat
GRASP_TIP_CLEAR = 0.010  # m: closed fingertips this far above the table at the pinch
PRE_GRASP_H = 0.10  # m above the pinch pose
RETREAT_H = 0.08
GRIPPER_STIFFNESS = 5000.0
PRESHAPE_OPEN = 0.45  # gripper opening (fraction) for the descent onto the fin: ~5 cm (ur5) / 3.6 cm (franka) — fully open, the inner finger hit blocks near the rim (d69)
SUCCESS_TOL = 0.04  # tray centre within this of the pad centre (m)


class tray_transport(granular_carry):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ build ----
    def _build_tray(self, pose):
        """One dynamic rigid body: floor plate, 4 rim walls, handle fin (local
        frame: z up, origin at the floor-plate bottom centre)."""
        sx, sy = TRAY_SIZE
        mat = self.scene.default_physical_material
        rigid = sapien.physx.PhysxRigidDynamicComponent()
        render = sapien.render.RenderBodyComponent()
        rmat = sapien.render.RenderMaterial(base_color=[*TRAY_COLOR, 1.0])
        z_floor = 0.5 * TRAY_FLOOR_T
        z_rim = TRAY_FLOOR_T + 0.5 * TRAY_RIM_H
        boxes = [
            ((0.0, 0.0, z_floor), (0.5 * sx, 0.5 * sy, 0.5 * TRAY_FLOOR_T)),
            ((0.0, 0.5 * sy - 0.5 * TRAY_RIM_T, z_rim), (0.5 * sx, 0.5 * TRAY_RIM_T, 0.5 * TRAY_RIM_H)),
            ((0.0, -0.5 * sy + 0.5 * TRAY_RIM_T, z_rim), (0.5 * sx, 0.5 * TRAY_RIM_T, 0.5 * TRAY_RIM_H)),
            ((0.5 * sx - 0.5 * TRAY_RIM_T, 0.0, z_rim), (0.5 * TRAY_RIM_T, 0.5 * sy, 0.5 * TRAY_RIM_H)),
            ((-0.5 * sx + 0.5 * TRAY_RIM_T, 0.0, z_rim), (0.5 * TRAY_RIM_T, 0.5 * sy, 0.5 * TRAY_RIM_H)),
            # handle tab protruding outward from the near rim centre
            ((0.0, -0.5 * sy - 0.5 * HANDLE_LEN, 0.5 * HANDLE_H), (0.5 * HANDLE_T, 0.5 * HANDLE_LEN, 0.5 * HANDLE_H)),
        ]
        for c, h in boxes:
            shape = sapien.physx.PhysxCollisionShapeBox(half_size=list(h), material=mat)
            shape.local_pose = sapien.Pose(list(c))
            rigid.attach(shape)
            vis = sapien.render.RenderShapeBox(list(h), rmat)
            vis.local_pose = sapien.Pose(list(c))
            render.attach(vis)
        ent = sapien.Entity()
        ent.set_name("tray")
        ent.add_component(rigid)
        ent.add_component(render)
        ent.set_pose(pose)
        self.scene.add_entity(ent)
        return ent

    def _draw_items(self):
        kinds = ["tall", "short"] + (["mid"] if np.random.uniform() < THIRD_ITEM_P else [])
        ranges = {"tall": (TALL_W, TALL_H), "short": (SHORT_W, SHORT_H), "mid": (MID_W, MID_H)}
        items = []
        for kind in kinds:
            (w_lo, w_hi), (h_lo, h_hi) = ranges[kind]
            items.append({"kind": kind, "w": float(np.random.uniform(w_lo, w_hi)),
                          "h": float(np.random.uniform(h_lo, h_hi)),
                          "mass": float(np.random.uniform(*ITEM_MASS_RANGE))})
        colors = [ITEM_COLORS[int(i)] for i in np.random.permutation(len(ITEM_COLORS))]
        for it, c in zip(items, colors):
            it["color"] = c
        return items

    def _place_items(self, items):
        """Tray-local (x, y) per block: a ROW along the tray's long side (y) in
        random order, the free length split randomly between the gaps, x
        random within the rims — no block stands in another's fall path
        along the sideways drag (a knocked tall block came to rest leaning
        on its neighbour, d71/d72). None if the row does not fit."""
        sx, sy = TRAY_SIZE
        y0 = -0.5 * sy + TRAY_RIM_T + HANDLE_CLEAR
        y1 = 0.5 * sy - TRAY_RIM_T - ITEM_EDGE_CLEAR
        order = [int(i) for i in np.random.permutation(len(items))]
        widths = [items[i]["w"] for i in order]
        free = (y1 - y0) - sum(widths) - ITEM_GAP * (len(items) - 1)
        if free < 0.0:
            return None
        parts = np.random.dirichlet(np.ones(len(items) + 1)) * free
        out = [None] * len(items)
        y = y0 + parts[0]
        for j, i in enumerate(order):
            hw = 0.5 * items[i]["w"]
            x_lim = max(0.5 * sx - TRAY_RIM_T - ITEM_EDGE_CLEAR - hw, 0.0)
            out[i] = np.array([np.random.uniform(-x_lim, x_lim), y + hw])
            y += items[i]["w"] + ITEM_GAP + parts[j + 1]
        return out

    def load_actors(self):
        side = 1.0 if np.random.uniform() < 0.5 else -1.0
        x = side * float(np.random.uniform(*START_ABS_X))
        y = float(np.random.uniform(*START_Y))
        self.carry_arm = "right" if side > 0 else "left"
        self.start_xy = np.array([x, y], dtype=np.float64)
        self._tray_mass = float(np.random.uniform(*TRAY_MASS_RANGE))
        self.tray = self._build_tray(sapien.Pose([x, y, TABLE_Z + 0.001]))
        self.container = self.tray  # granular_carry helpers read self.container
        # no particles in this task (granular_pour.check_stable / overlays)
        self.beads, self._bead_names, self.n_beads = [], set(), 0

        items = self._draw_items()
        local = self._place_items(items)
        if local is None:  # cannot happen for the configured ranges; fall back to two items
            items = items[:2]
            local = self._place_items(items)
        self.standing = []
        for k, (it, lxy) in enumerate(zip(items, local)):
            item_id = f"block_{k}"
            pos = [x + float(lxy[0]), y + float(lxy[1]), TABLE_Z + TRAY_FLOOR_T + 0.5 * it["h"] + 0.002]
            ent = create_box(self.scene, sapien.Pose(pos), half_size=[0.5 * it["w"], 0.5 * it["w"], 0.5 * it["h"]],
                             color=COLOR_RGB[it["color"]], name=item_id)
            it.update(item_id=item_id, entity=ent, a_tip=float(9.81 * it["w"] / it["h"]))
            self.standing.append(it)

        kinds = list(REFERENT_P)
        ref_kind = kinds[int(np.random.choice(len(kinds), p=np.asarray([REFERENT_P[k] for k in kinds])))]
        ref_kind = os.environ.get("ITEM_T5_REFERENT", ref_kind)  # diag-only override
        ref = next(it for it in self.standing if it["kind"] == ref_kind)
        self.referent_id = ref["item_id"]
        self.primary_rule_id = f"upright|{ref['item_id']}|-"

        t = self.start_xy + np.array([-side * float(np.random.uniform(*CARRY_DX)), float(np.random.uniform(*CARRY_DY))])
        self.target_xy = t.astype(np.float64)
        self.pad = goal_frame(self.scene, self.target_xy, TABLE_Z, PAD_SIZE, PAD_FRAME_W, name="target_pad")
        self.zones = []
        self.episode_info = {
            "referent": self.referent_id, "referent_kind": ref_kind,
            "items": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in it.items() if k != "entity"}
                      for it in self.standing],
        }

        sx, sy = TRAY_SIZE
        register_item(
            self, "tray", self.tray, category="tray", color="", kind="rigid", role="task_object",
            half_extents=[0.5 * sx, 0.5 * sy, 0.5 * HANDLE_H], obb_center_local=[0.0, 0.0, 0.5 * HANDLE_H],
            up_axis_local=[0.0, 0.0, 1.0], mass=self._tray_mass,
        )
        for it in self.standing:
            register_item(
                self, it["item_id"], it["entity"], category="block", color=it["color"], kind="rigid",
                role="candidate", half_extents=[0.5 * it["w"], 0.5 * it["w"], 0.5 * it["h"]],
                obb_center_local=[0.0, 0.0, 0.0], up_axis_local=[0.0, 0.0, 1.0], mass=it["mass"],
                size={"tall": "tall", "short": "short"}.get(it["kind"]),  # "mid" has no size word
            )

    def prepare_episode(self):
        """No pre-roll (nothing to settle or clean beyond RoboTwin's setup)."""
        return None

    # ------------------------------------------------------------------ play ----
    def _pinch_pose(self, eparams):
        """EE pose pinching the handle tab top-down, fingers closing across its
        thickness (the tray's x), closed fingertips GRASP_TIP_CLEAR above the
        table, the hand outside the tray footprint."""
        T = np.asarray(self.tray.get_pose().to_transformation_matrix(), dtype=np.float64)
        fin = T[:3, :3] @ np.array([0.0, -0.5 * TRAY_SIZE[1] - PINCH_OUT, 0.0]) + T[:3, 3]
        tip_len = float(eparams.get("tip_len", APPROACH_TO_TCP + 0.04))
        z_tcp = TABLE_Z + GRASP_TIP_CLEAR + (tip_len - APPROACH_TO_TCP)
        q = _push_quat(T[:2, 1])  # finger axis across the tray's y axis -> along its x (tab thickness)
        return np.array([fin[0], fin[1], z_tcp + APPROACH_TO_TCP] + list(q), dtype=np.float64)

    def play_once(self):
        eparams = getattr(self, "embodiment_params", None) or {}
        arm = self.carry_arm
        v_safe = SAFE_SPEED * float(getattr(self, "_time_scale_safe", 0.6))
        v_risky = TRAY_SPEED * float(getattr(self, "_time_scale_risky", 1.0))
        k = float(eparams.get("gripper_stiffness", GRIPPER_STIFFNESS))
        joints, damping = ((self.robot.left_gripper, self.robot.left_gripper_damping) if arm == "left"
                           else (self.robot.right_gripper, self.robot.right_gripper_damping))
        for joint in joints:
            joint[0].set_drive_property(stiffness=k, damping=float(damping))

        # 1. pinch the handle fin — safe: above it, straight down, close
        pinch = self._pinch_pose(eparams)
        self.move(self.open_gripper(arm, pos=PRESHAPE_OPEN))
        self.move(self.move_to_pose(arm, (pinch + np.array([0, 0, PRE_GRASP_H, 0, 0, 0, 0])).tolist()))
        self._steps_to(arm, pinch, v_safe)
        self.move(self.close_gripper(arm))
        # 2. drag the tray to the pad — risky: ONE continuous move, hard stop
        tray_xy = np.asarray(self.tray.get_pose().p, dtype=np.float64)[:2]
        d = self.target_xy - tray_xy
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        with risky(self):
            self._steps_to(arm, np.concatenate([ee[:3] + [d[0], d[1], 0.0], ee[3:]]), v_risky, stop=True,
                           accel_dist=TRAY_ACCEL_DIST,
                           stop_dist=float(os.environ.get("ITEM_T5_STOP_DIST", TRAY_STOP_DIST)))  # env: diag-only
        # 3. release and retreat — safe
        self.move(self.open_gripper(arm))
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        self._steps_to(arm, np.concatenate([ee[:3] + [0.0, 0.0, RETREAT_H], ee[3:]]), v_safe)

        self.info["info"] = {"{A}": "tray", "{a}": arm}
        return self.info

    # ---------------------------------------------------------------- success ----
    def _tilt_deg(self, ent, up_local):
        R = np.asarray(ent.get_pose().to_transformation_matrix(), dtype=np.float64)[:3, :3]
        up = R @ np.asarray(up_local, dtype=np.float64)
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def check_success(self):
        """Nominal goal only: the tray rests on the pad (centre within
        SUCCESS_TOL). Toppling is a rule."""
        xy = np.asarray(self.tray.get_pose().p, dtype=np.float64)[:2]
        return bool(np.linalg.norm(xy - self.target_xy) <= SUCCESS_TOL)

    def overlay_lines(self):
        if not getattr(self, "standing", None):
            return []
        parts = []
        for it in self.standing:
            tilt = self._tilt_deg(it["entity"], [0.0, 0.0, 1.0])
            star = "*" if it["item_id"] == getattr(self, "referent_id", None) else ""
            flag = "!" if tilt > UPRIGHT_MAX_DEG else ""
            parts.append(f"{flag}{star}{it['color']} {it['kind']} {tilt:.0f}deg (tip {it['a_tip']:.1f})")
        xy = np.asarray(self.tray.get_pose().p, dtype=np.float64)[:2]
        v = 0.0
        try:
            v = float(np.linalg.norm(self.tray.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                                     .get_linear_velocity()[:2]))
        except Exception:
            pass
        return ["  ".join(parts),
                f"tray v {v:.2f}m/s  ->pad {np.linalg.norm(xy - self.target_xy) * 100:.1f}cm  (* = referent)"]
