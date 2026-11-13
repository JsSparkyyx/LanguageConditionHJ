"""tray_slide — the tray task with real objects (Yuxuan 2026-11-12), for the VLA test on this task alone.

T5 (tray_transport.py) carried primitive blocks, and its short block could not fall at any drag speed. Here the tray
carries a WINE GLASS (tall: tips at a gentle stop) and a CAN (short: needs a hard stop to tip, but it does tip), so
both can fall and each needs its own speed:

    slow    (time_scale SPEEDS["slow"])    neither falls        -> "slide the tray", or keep the glass standing
    medium  (time_scale SPEEDS["medium"])  the glass falls, the can stands -> keep the can standing
    fast    (time_scale SPEEDS["fast"])    both fall            -> the careless drag (the start alone tips the glass)

The mechanism is T5's: one arm pinches the tray's handle tab and drags it SIDEWAYS along the table toward the table
centre in ONE continuous move (via path); an upright item of base radius r with its centre of mass at height c tips
once the tray decelerates past about g r / c, and the brake's peak deceleration is 0.75 v^2 / stop_dist. The drag is
longer than T5's (32-38 cm) so that a recovery fits in it: the ``speed_change`` time law (via_path.py) starts at the
cruise speed of time_scale and, ``slow_at`` of the way along, slows to the cruise speed of ``slow_to`` over
``change_dist`` before the brake.

Risk knobs (``task.risk_knobs``, set at run time):
    behaviour    a named behaviour (BEHAVIOURS: slow, medium, fast, recover_glass, recover_can); sets the two below
                 and the cruise speed, and wins over time_scale
    slow_to      time scale of the second cruise (None: one cruise speed)
    slow_at      where the change starts, as a fraction of the drag length (default SLOW_AT)
    change_dist  m over which the speed changes (default CHANGE_DIST)

Physical design (prepare_episode): the runner's physics randomisation sets each item's centre of mass to the actor
origin, which for these meshes is the BASE (they would never tip), and draws the friction for the category (a slippery
can slides instead of tipping). prepare_episode puts the centre of mass back at COM_FRAC of the item height (the
convex hulls' own value, measured), keeps a small horizontal offset derived from the runner's draw, and sets a
high friction (ITEM_FRICTION) so the items tip rather than slide.

Models (RoboTwin assets, y up, origin at the base): wine glass 088_wineglass ids GLASS_IDS (id 4's collision base
is a 0.9 cm stub, id 0 tips erratically: excluded), scaled to a height in GLASS_H; can 071_can ids CAN_IDS scaled to a height in CAN_H.

This file is installed into ``RoboTwin/envs/`` by symlink and must not import CEHJ ``main.*``. Global-RNG draw order
in load_actors: side -> start |x| -> start y -> tray mass -> glass (model, height, mass) -> can (model, height, mass)
-> row order -> gap split -> per item x -> referent kind -> pad (dx, dy) -> per item friction.
"""

import os

from .utils import *
import sapien

from ._item_utils import goal_frame, register_item, risky
from .granular_pour import TABLE_Z
from .tray_transport import (
    HANDLE_H,
    PAD_FRAME_W,
    PAD_SIZE,
    PRE_GRASP_H,
    PRESHAPE_OPEN,
    RETREAT_H,
    SUCCESS_TOL,
    TRAY_FLOOR_T,
    TRAY_MASS_RANGE,
    TRAY_RIM_T,
    TRAY_SIZE,
    GRIPPER_STIFFNESS,
    tray_transport,
)

# --- layout (mirrors main/item_envs/configs/tasks/T8_tray_slide.yaml) ---
START_ABS_X = (0.30, 0.36)  # tray centre |x| on its (random) side: further out than T5, the drag is longer
START_Y = (-0.06, 0.02)
CARRY_DX = (0.32, 0.38)  # m toward the table centre (T5: 0.20-0.26); ends within ~8 cm of the centre line
CARRY_DY = (-0.02, 0.02)

# --- items: (asset, ids, height range m, mass range kg); sizes at the model's own json scale, measured from the
# collision meshes (runs/lang/tray_slide_1112): id -> (width m, height m, base radius m) ---
GLASS_MODEL = "088_wineglass"
GLASS_IDS = (1, 2)  # id 0 (wide bowl) tips erratically: at 0.30 m/s but not 0.35 (T8 card)
GLASS_HELDOUT_IDS = (3,)
GLASS_SIZE = {0: (0.127, 0.253, 0.041), 1: (0.081, 0.253, 0.042), 2: (0.075, 0.253, 0.038), 3: (0.116, 0.304, 0.054)}
GLASS_H = (0.19, 0.21)
GLASS_MASS = (0.15, 0.25)
CAN_MODEL = "071_can"
CAN_IDS = (0, 2, 3, 5)  # ids 1 and 6 (narrow) tip at 0.55 m/s, too close to the glass (T8 card)
CAN_SIZE = {0: (0.051, 0.096, 0.022), 1: (0.046, 0.096, 0.022), 2: (0.052, 0.096, 0.021), 3: (0.059, 0.097, 0.026),
            5: (0.054, 0.096, 0.025), 6: (0.046, 0.097, 0.021)}
CAN_H = (0.11, 0.12)  # a 330 ml can is 11.5 cm tall
CAN_MASS = (0.30, 0.40)  # a full can
COM_FRAC = {"glass": 0.59, "can": 0.49}  # centre-of-mass height / item height (the hulls' own, measured)
COM_XY_MAX = 0.15  # horizontal CoM offset at most this fraction of the base radius
ITEM_FRICTION = (0.8, 1.0)  # static mu of the items: they tip, not slide (T5 blocks, d69)
Y_UP = [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]  # wxyz: model y -> world z

ITEM_EDGE_CLEAR = 0.025
ITEM_GAP = 0.03
HANDLE_CLEAR = 0.05
REFERENT_P = {"glass": 0.5, "can": 0.5}  # the sentence names either one (Yuxuan 2026-11-12)
UPRIGHT_MAX_DEG = 45.0

# --- nominal motion ---
TRAY_SPEED = 0.30  # m/s cruise at time_scale 1.0 (as T5)
SPEEDS = {"slow": 0.5, "medium": 1.75, "fast": 2.5}  # time scales of the three behaviours (calibration: T8 card)
TRAY_ACCEL_DIST = 0.10  # m: smooth start (peak accel 0.75 v^2 / this; T5 0.06 — a fast start must not tip the glass)
TRAY_STOP_DIST = 0.04  # m: brake at the pad (peak decel 0.75 v^2 / this)
SLOW_AT = 0.30  # recovery: the change starts this far along the drag (fraction)
CHANGE_DIST = 0.10  # m over which a recovery changes speed
SAFE_SPEED = 0.10  # m/s x safe_time_scale: approach / descend / retreat
# the nominal behaviours by name (risk knob "behaviour"): (cruise, slow down to) as SPEEDS keys
BEHAVIOURS = {
    "slow": ("slow", None),               # nothing falls: the task alone, or "keep the glass standing"
    "medium": ("medium", None),           # the glass falls, the can stands: "keep the can standing"
    "fast": ("fast", None),               # both fall: the careless drag
    "recover_glass": ("medium", "slow"),  # starts too fast for the glass, slows down (a fast start alone tips it)
    "recover_can": ("fast", "medium"),    # starts fast, slows down enough for the can
}


def behaviour_scales(name):
    """(time_scale, slow_to) of a named behaviour; slow_to None = one cruise speed."""
    if name not in BEHAVIOURS:
        raise ValueError(f"tray_slide: unknown behaviour {name!r}; known: {sorted(BEHAVIOURS)}")
    a, b = BEHAVIOURS[name]
    return float(SPEEDS[a]), (None if b is None else float(SPEEDS[b]))


class tray_slide(tray_transport):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ build ----
    def _draw_items(self):
        items = []
        for kind, model, ids, sizes, h_rng, m_rng in (("glass", GLASS_MODEL, GLASS_IDS, GLASS_SIZE, GLASS_H, GLASS_MASS),
                                                      ("can", CAN_MODEL, CAN_IDS, CAN_SIZE, CAN_H, CAN_MASS)):
            mid = int(ids[int(np.random.randint(len(ids)))])
            override = os.environ.get(f"ITEM_T8_{kind.upper()}_ID")  # diag-only
            if override:
                mid = int(override)
            w0, h0, r0 = sizes[mid]
            h = float(np.random.uniform(*h_rng))
            f = h / h0
            items.append({"kind": kind, "model": model, "model_id": mid, "factor": f, "h": h, "w": w0 * f,
                          "base_r": r0 * f, "mass": float(np.random.uniform(*m_rng))})
        return items

    def _place_items(self, items):
        """Tray-local (x, y) per item: a row along the tray's long side (y), random order and gaps (T5's rule: no
        item stands in the other's fall path along the sideways drag)."""
        sx, sy = TRAY_SIZE
        y0 = -0.5 * sy + TRAY_RIM_T + HANDLE_CLEAR
        y1 = 0.5 * sy - TRAY_RIM_T - ITEM_EDGE_CLEAR
        order = [int(i) for i in np.random.permutation(len(items))]
        free = (y1 - y0) - sum(items[i]["w"] for i in order) - ITEM_GAP * (len(items) - 1)
        parts = np.random.dirichlet(np.ones(len(items) + 1)) * max(free, 0.0)
        out = [None] * len(items)
        y = y0 + parts[0]
        for j, i in enumerate(order):
            hw = 0.5 * items[i]["w"]
            x_lim = max(0.5 * sx - TRAY_RIM_T - ITEM_EDGE_CLEAR - hw, 0.0)
            out[i] = np.array([np.random.uniform(-x_lim, x_lim), y + hw])
            y += items[i]["w"] + ITEM_GAP + parts[j + 1]
        return out

    def _load_mesh(self, model, model_id, factor, pose):
        """RoboTwin's create_actor (convex collisions) with the json scale times ``factor``."""
        import json
        from pathlib import Path

        from .utils.create_actor import Actor, get_glb_or_obj_file

        d = Path("assets/objects") / model
        data = json.loads((d / f"model_data{model_id}.json").read_text())
        s = np.asarray(data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
        scale = [float(v) * factor for v in (np.full(3, s[0]) if s.size == 1 else np.resize(s, 3))]
        b = self.scene.create_actor_builder()
        b.set_physx_body_type("dynamic")
        b.add_multiple_convex_collisions_from_file(filename=str(get_glb_or_obj_file(d / "collision", model_id)),
                                                   scale=scale)
        b.add_visual_from_file(filename=str(get_glb_or_obj_file(d / "visual", model_id)), scale=scale)
        ent = b.build(name=model)
        ent.set_pose(pose)
        return Actor(ent, {**data, "scale": scale})

    def load_actors(self):
        side = 1.0 if np.random.uniform() < 0.5 else -1.0
        x = side * float(np.random.uniform(*START_ABS_X))
        y = float(np.random.uniform(*START_Y))
        self.carry_arm = "right" if side > 0 else "left"
        self.start_xy = np.array([x, y], dtype=np.float64)
        self._tray_mass = float(np.random.uniform(*TRAY_MASS_RANGE))
        self.tray = self._build_tray(sapien.Pose([x, y, TABLE_Z + 0.001]))
        self.container = self.tray  # granular_carry helpers read self.container
        self.beads, self._bead_names, self.n_beads = [], set(), 0

        items = self._draw_items()
        local = self._place_items(items)
        self.standing = []
        for it, lxy in zip(items, local):
            pos = [x + float(lxy[0]), y + float(lxy[1]), TABLE_Z + 0.001 + TRAY_FLOOR_T + 0.003]
            actor = self._load_mesh(it["model"], it["model_id"], it["factor"], sapien.Pose(pos, Y_UP))
            actor.actor.set_name(it["kind"])
            it.update(item_id=it["kind"], entity=actor, a_tip=float(9.81 * it["base_r"] / (COM_FRAC[it["kind"]] * it["h"])))
            self.standing.append(it)

        kinds = list(REFERENT_P)
        ref_kind = kinds[int(np.random.choice(len(kinds), p=np.asarray([REFERENT_P[k] for k in kinds])))]
        ref_kind = os.environ.get("ITEM_T8_REFERENT", ref_kind)  # diag-only override
        self.referent_id = ref_kind
        self.primary_rule_id = f"upright|{ref_kind}|-"

        t = self.start_xy + np.array([-side * float(np.random.uniform(*CARRY_DX)), float(np.random.uniform(*CARRY_DY))])
        self.target_xy = t.astype(np.float64)
        self.pad = goal_frame(self.scene, self.target_xy, TABLE_Z, PAD_SIZE, PAD_FRAME_W, name="target_pad")
        for it in self.standing:  # applied in prepare_episode, after the runner's physics randomisation
            it["mu"] = float(np.random.uniform(*ITEM_FRICTION))
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
                self, it["item_id"], it["entity"], category={"glass": "wine_glass", "can": "can"}[it["kind"]],
                color="", kind="rigid", role="candidate",
                half_extents=[0.5 * it["w"], 0.5 * it["h"], 0.5 * it["w"]],  # model frame: y up, origin at the base
                obb_center_local=[0.0, 0.5 * it["h"], 0.0], up_axis_local=[0.0, 1.0, 0.0], mass=it["mass"],
                size={"glass": "tall", "can": "short"}[it["kind"]],
            )

    def prepare_episode(self):
        """The items' physical design, after the runner's physics randomisation (see the module docstring): centre
        of mass at COM_FRAC of the height plus a small horizontal offset, friction ITEM_FRICTION. No motion."""
        design = {}
        for it in self.standing:
            ent = it["entity"].actor if hasattr(it["entity"], "actor") else it["entity"]
            comp = ent.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            p = np.asarray(comp.cmass_local_pose.p, dtype=np.float64)  # the runner's draw: (dx, dy, 0), |d| <= 0.2 half
            hx, hy = 0.5 * it["w"], 0.5 * it["h"]
            ox = float(np.clip(p[0] / max(hx, 1e-9) / 0.2, -1, 1)) * COM_XY_MAX * it["base_r"]
            oz = float(np.clip(p[1] / max(hy, 1e-9) / 0.2, -1, 1)) * COM_XY_MAX * it["base_r"]
            com = [ox, COM_FRAC[it["kind"]] * it["h"], oz]
            comp.cmass_local_pose = sapien.Pose(com, [1.0, 0.0, 0.0, 0.0])
            mu = it["mu"]
            mat = self.scene.create_physical_material(mu, 0.95 * mu, 0.0)
            for shape in comp.get_collision_shapes():
                shape.physical_material = mat
            design[it["item_id"]] = {"com_local": [round(v, 4) for v in com], "friction": round(mu, 3),
                                     "mass": round(float(comp.mass), 4)}
        self.episode_info["physical_design"] = design
        return None

    # ------------------------------------------------------------------ play ----
    def drag_profile(self, length):
        """The drag's time law: hard_stop at time_scale, or speed_change down to ``slow_to`` (risk knobs). The knob
        ``behaviour`` (BEHAVIOURS) sets both by name and wins over time_scale / slow_to."""
        knobs = dict(getattr(self, "risk_knobs", None) or {})
        ts = float(getattr(self, "_time_scale_risky", 1.0))
        if knobs.get("behaviour") is not None:
            ts, knobs["slow_to"] = behaviour_scales(str(knobs["behaviour"]))
        v = TRAY_SPEED * ts
        prof = {"kind": "hard_stop", "length": float(length), "v": v, "accel_dist": TRAY_ACCEL_DIST,
                "stop_dist": float(knobs.get("stop_dist", TRAY_STOP_DIST))}
        if knobs.get("slow_to") is not None:
            prof.update(kind="speed_change", v2=TRAY_SPEED * float(knobs["slow_to"]),
                        switch=float(knobs.get("slow_at", SLOW_AT)) * float(length),
                        change_dist=float(knobs.get("change_dist", CHANGE_DIST)))
        if knobs.get("behaviour") is not None:
            prof["behaviour"] = str(knobs["behaviour"])
        return prof

    def play_once(self):
        eparams = getattr(self, "embodiment_params", None) or {}
        arm = self.carry_arm
        v_safe = SAFE_SPEED * float(getattr(self, "_time_scale_safe", 0.6))
        k = float(eparams.get("gripper_stiffness", GRIPPER_STIFFNESS))
        joints, damping = ((self.robot.left_gripper, self.robot.left_gripper_damping) if arm == "left"
                           else (self.robot.right_gripper, self.robot.right_gripper_damping))
        for joint in joints:
            joint[0].set_drive_property(stiffness=k, damping=float(damping))

        # 1. pinch the handle tab — safe: above it, straight down, close
        pinch = self._pinch_pose(eparams)
        self.move(self.open_gripper(arm, pos=PRESHAPE_OPEN))
        self.move(self.move_to_pose(arm, (pinch + np.array([0, 0, PRE_GRASP_H, 0, 0, 0, 0])).tolist()))
        self._steps_to(arm, pinch, v_safe)
        self.move(self.close_gripper(arm))
        # 2. drag the tray to the pad — risky: ONE continuous move on the drag profile
        tray_xy = np.asarray(self.tray.get_pose().p, dtype=np.float64)[:2]
        d = self.target_xy - tray_xy
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        with risky(self):
            self._drag(arm, np.concatenate([ee[:3] + [d[0], d[1], 0.0], ee[3:]]))
        # 3. release and retreat — safe
        self.move(self.open_gripper(arm))
        ee = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        self._steps_to(arm, np.concatenate([ee[:3] + [0.0, 0.0, RETREAT_H], ee[3:]]), v_safe)

        self.info["info"] = {"{A}": "tray", "{a}": arm}
        return self.info

    def _drag(self, arm, target_pose):
        """``_steps_to``'s via path (straight line, dense waypoints) timed by drag_profile. Falls back to T5's
        hard-stop drag at the first cruise speed when the profile cannot run (no via path)."""
        from .granular_carry import CARRY_STEP

        p0 = np.asarray(self.get_arm_pose(arm), dtype=np.float64).reshape(7)
        p1 = np.asarray(target_pose, dtype=np.float64).reshape(7)
        length = float(np.linalg.norm(p1[:3] - p0[:3]))
        n = max(1, int(np.ceil(length / CARRY_STEP)))
        poses = [(p0[:3] + (k / n) * (p1[:3] - p0[:3])).tolist() + p1[3:].tolist() for k in range(1, n + 1)]
        prof = self.drag_profile(length)
        self.episode_info["drag_profile"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in prof.items()}
        ok_before = bool(self.plan_success)
        self.plan_local = True
        self.plan_via, self.plan_profile = poses[:-1], prof
        try:
            self.move(self.move_to_pose(arm, poses[-1]))
        finally:
            self.plan_via, self.plan_profile = None, None
            self.plan_local = False
        if ok_before and not self.plan_success:
            self.plan_success = True
            self.episode_info["drag_fallback"] = True
            self._steps_to(arm, target_pose, prof["v"], stop=True, accel_dist=TRAY_ACCEL_DIST,
                           stop_dist=prof["stop_dist"])

    # ---------------------------------------------------------------- success ----
    def overlay_lines(self):
        if not getattr(self, "standing", None):
            return []
        parts = []
        for it in self.standing:
            tilt = self._tilt_deg(it["entity"], [0.0, 1.0, 0.0])
            star = "*" if it["item_id"] == getattr(self, "referent_id", None) else ""
            flag = "!" if tilt > UPRIGHT_MAX_DEG else ""
            parts.append(f"{flag}{star}{it['kind']} {tilt:.0f}deg (tip {it['a_tip']:.1f})")
        xy = np.asarray(self.tray.get_pose().p, dtype=np.float64)[:2]
        v = 0.0
        try:
            v = float(np.linalg.norm(self.tray.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                                     .get_linear_velocity()[:2]))
        except Exception:
            pass
        return ["  ".join(parts),
                f"tray v {v:.2f}m/s  ->pad {np.linalg.norm(xy - self.target_xy) * 100:.1f}cm  (* = referent)"]
