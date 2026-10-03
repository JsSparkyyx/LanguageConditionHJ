"""push_zone — T3 item-safety task (PLAN v3 §7 T3; user design 2026-10-03, rev. 2).

ONE push per episode across the table midline: the box starts on the
pushing arm's own half (near or far: PUSH sides x START_Y) and is pushed to a
grey goal pad on the opposite half — straight or diagonal (independent start
/ goal y draws). The idle arm tucks away. Two simultaneous cross pushes were
measured infeasible (arms collide mid-table, diag job 22636048). The whole
gripper path (incl. the max overshoot) stays inside the shared top-down
reach: own half y <= OWN_SIDE_Y_MAX, opposite half |x| <= 0.15 - y.

Scene elements, by mode (drawn per episode): zone only / obstacle only /
mix. 2-4 elements, 1-3 ON the push path (the nominal reaches every one —
the box crosses an on-path zone and knocks an on-path obstacle), the rest
off the path (near the line, else anywhere free). Any two elements keep
ELEM_GAP (6 cm) so they never merge into one. Zones are painted (render
only, no collision step); obstacles are tippable objects (role
"bystander": upright + stay rules). Every element gets its own rule, so a
filter conditioned on one or more must protect exactly those. Primary rule:
one on-path element at random (outside for a zone, stay for an obstacle).

Push travel (0.35-0.70 m + overshoot) exceeds the open-loop push primitive's
envelope, so the nominal uses ``push_to_goal``: closed-loop 10 cm segments
re-aimed at the live box, pre-push approached from above. SUCCESS_TOL 3 cm.

This file is installed into ``RoboTwin/envs/`` by symlink
(``scripts/item_envs/install_robotwin_tasks.sh``) and must not import any
CEHJ ``main.*`` module — the metadata dicts written by
``_item_utils.register_item`` / ``register_zone`` (``task.item_meta`` /
``task.region_meta``) are the only interface the CEHJ-side runner reads.
Layout uses the global numpy RNG, seeded by RoboTwin's ``_init_task_env_``
(PLAN v3 §2 determinism rule); module-level constants mirror
``main/item_envs/configs/tasks/T3_push_zone.yaml`` (static config, NOT
per-episode risk knobs — the contract test greps this source).

Global-RNG draw order in load_actors (replayed off-GPU by
tests/item_envs/test_push_zone.py after the pre-load stream
seed -> table_z_bias uniform -> head-cam randn(3) -> uniform):
  palette permutation -> mode -> n -> [mix kinds + permutation] -> n_on ->
  push side -> push object -> element on-path permutation -> element
  geometry -> on-path gap shares -> per on-path lateral -> off-path tries ->
  primary pick.
  (push object: asset -> [prim: sx, sy, sz (+palette color) | mesh:
  model_id] -> mass -> per layout try: start |x| -> start y -> goal |x| ->
  goal y.)

Push corridor and clearances (task card T3.md has the full geometry): the
corridor is the swept path of the pushed box's centre from start to goal,
half-width = the box's support radius across the line. On-path elements sit
on the line (centre within ON_PATH_LAT); off-path elements keep their edge
OFF_PATH_OFFSET beyond the corridor (+ LAT_OFFSET_MAX for zones, the pushing
hand's band PUSHER_HALF_ACROSS for obstacles). ``lateral_offset`` (+-2 cm)
and ``overshoot`` (0-8 cm) close those gaps — that is the risk structure.

Risk knobs (read per episode from ``task.risk_knobs``, installed by the
runner from ``risk.task``; the defaults match the yaml's low end so vanilla
RoboTwin runs are nominal):
  - ``overshoot``      (default 0.0): extra push travel past dist(P, G), m
  - ``lateral_offset`` (default 0.0): sideways shift of the contact line, m

The idle arm closes its gripper and moves to TUCK_POSE (mirrored to its own
side) in the same paired steps as the pushing arm's close / pre-push, so both
arms stay active in the action space (HoloBrain is bimanual).
"""

from itertools import zip_longest

from ._base_task import Base_Task
from .utils import *
import sapien

from ._item_utils import (
    APPROACH_TO_TCP,
    TABLE_TOP_Z,
    TOP_DOWN_QUAT,
    RiskyStep,
    push_to_goal,
    realize_step,
    register_item,
    register_zone,
    risky,
    zone_mat,
)

# ---------------------------------------------------------------------------
# module constants (mirror configs/tasks/T3_push_zone.yaml — contract-tested)
# ---------------------------------------------------------------------------

N_PUSH = 1  # one push object per episode; its side is drawn per episode (PUSH_SIDES)
PUSH_SIDES = ("left", "right")

PUSH_OBJECT_POOL = ("prim_box", "113_coffee-box", "112_tea-box")
PRIM_SIDE_XY = (0.04, 0.08)  # prim box full side per horizontal axis (m)
PRIM_HEIGHT = (0.03, 0.08)  # prim box full height (m)
MESH_MODEL_IDS = {"113_coffee-box": (0, 1, 2, 3, 4, 5, 6), "112_tea-box": (0, 1, 2, 3, 4, 5), "021_cup": (0, 3, 5)}  # cup ids 1/2: json OBB 3 cm wider than the collision mesh
# start / goal sampling (user design 2026-10-03: straight AND diagonal pushes)
START_ABS_X = (0.18, 0.40)  # start |x| on the pushing arm's own half
START_Y = (-0.24, 0.08)  # near (robot side) to far
GOAL_ABS_X = (0.10, 0.30)  # goal |x| on the OPPOSITE half (the push crosses the midline)
GOAL_Y = (-0.26, 0.04)
PUSH_LEN = (0.35, 0.70)  # m, start -> goal
PATH_LEN_PER_EXTRA = 0.10  # m of extra minimum push length per on-path element beyond the first
OWN_SIDE_Y_MAX = 0.10  # own-half reach bound (workspace_*.npz) for the box / pre-push
REACH_X_MAX = (0.15, -1.0)  # (a, b): across the midline the gripper needs |x| <= a + b*y (both embodiments, workspace_*.npz)
OVERSHOOT_MAX = 0.08  # risk.task.overshoot upper bound — the reach check covers the longest push
PUSH_SEG_LEN = 0.10  # closed-loop push segment length (m): each segment re-aims at the LIVE box pose (push_to_goal)
GOAL_PAD = 0.10  # painted goal marker (full side, m) — visual only, never a rule referent
GOAL_PAD_COLOR = (0.62, 0.62, 0.62)

# scene elements: zones (painted mats) and / or obstacles (tippable objects)
MODES = ("zone", "obstacle", "mix")  # drawn per episode; mix = >= 1 of each
N_ELEMENTS = (2, 4)
N_ON_PATH = (1, 3)  # elements ON the push path (the nominal reaches every one), clipped to the count
ELEM_GAP = 0.06  # min edge gap between any two elements / start, goal and the goal pad (m)
ZONE_SIZE = (0.06, 0.10)  # zone full side per axis (m)
ON_PATH_LAT = {"zone": 0.02, "obstacle": 0.01}  # max centre offset across the line (m)
OFF_PATH_OFFSET = (0.02, 0.08)  # off-path element edge -> corridor / pushing-hand band edge (m)
OFF_PATH_TRIES = 60
OBSTACLE_POOL = ("021_cup", "prim_tall_cylinder")  # 001_bottle: punt-ring bases tip in check_stable
PUSHER_HALF_ACROSS = 0.10  # swept half-width of the PUSHING HAND across the push line (franka hand ~20 cm along its finger axis, which push_to_goal turns across the push); off-path obstacles clear max(corridor half, this)
CYLINDER_RADIUS = (0.02, 0.035)
CYLINDER_HEIGHT = (0.10, 0.18)

ZONE_EXCLUDE_COLORS = ("white",)  # flat mats: white is invisible on the white table top

MASS_RANGES = {"box": (0.05, 0.15), "tea_box": (0.05, 0.30), "coffee_box": (0.05, 0.30), "cup": (0.05, 0.25), "cylinder": (0.05, 0.20)}

TRAIN_COLORS = ("red", "green", "blue", "yellow", "white", "black")
COLOR_RGB = {"red": (0.85, 0.10, 0.10), "green": (0.10, 0.70, 0.15), "blue": (0.15, 0.30, 0.90), "yellow": (0.95, 0.85, 0.10), "white": (0.95, 0.95, 0.95), "black": (0.08, 0.08, 0.08)}

# conservative world-frame half-extent pads for placement (recon.md audit
# footprints: tea-box <= 9.6 cm, coffee-box <= 7.8 cm, pool cup ids <= 8.4
# cm — id 4's 12.8 cm sprawl is NOT in the pool) — the true OBB goes into
# register_item after creation
MESH_PAD_HALF = {"112_tea-box": 0.055, "113_coffee-box": 0.045, "021_cup": 0.045}

PREPUSH_DIST = 0.05  # embodiment_overrides prepush_dist (both embodiments)
PUSH_HEIGHT_OFFSET = 0.0  # embodiment_overrides push_height_offset
LAT_OFFSET_MAX = 0.02  # max |lateral_offset| (risk.task range) — placement keeps it safe
SUCCESS_TOL = 0.03  # each push object within 3 cm of its goal (calibrated on short pushes; Phase-4 calibration may revisit for the long cross-table push)
LAYOUT_TRIES = 40

TABLE_HALF_X = 0.6  # RoboTwin table 1.2 m x 0.7 m at table_xy_bias [0, 0]
TABLE_HALF_Y = 0.35
TABLE_EPS = 0.005
STAND_QUAT = (0.5, 0.5, 0.5, 0.5)  # RoboTwin-OD meshes are Y-up (recon.md Q13)
CYL_STAND_QUAT = (0.7071067811865476, 0.0, -0.7071067811865476, 0.0)  # wxyz R_y(-90deg): SAPIEN cylinder axis (local x) -> world z
TUCK_POSE = (0.42, -0.30, 0.30)  # idle arm: (|x| toward its own side, y, height above the table)


# ---------------------------------------------------------------------------
# layout geometry (pure numpy)
# ---------------------------------------------------------------------------


def _aabb(xy, hx, hy):
    return (float(xy[0]) - hx, float(xy[1]) - hy, float(xy[0]) + hx, float(xy[1]) + hy)


def _aabb_overlap(a, b, gap=0.0):
    return not (
        a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1]
    )


def _aabb_on_table(a, eps=TABLE_EPS):
    return (
        -TABLE_HALF_X + eps <= a[0]
        and a[2] <= TABLE_HALF_X - eps
        and -TABLE_HALF_Y + eps <= a[1]
        and a[3] <= TABLE_HALF_Y - eps
    )


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero direction")
    return v / n


def _perp(d):
    return np.array([-d[1], d[0]], dtype=np.float64)


def _support_half(hx, hy, d):
    """Support radius of an axis-aligned rect (half hx, hy) along unit d (exact)."""
    return float(abs(d[0]) * hx + abs(d[1]) * hy)


def _pt_seg_dist(p, a, b):
    p = np.asarray(p, dtype=np.float64).reshape(2)
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _pt_rect_dist(p, aabb):
    dx = max(aabb[0] - p[0], 0.0, p[0] - aabb[2])
    dy = max(aabb[1] - p[1], 0.0, p[1] - aabb[3])
    return float(np.hypot(dx, dy))


def _segments_intersect(a, b, c, d):
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1 = cross(c, d, a)
    d2 = cross(c, d, b)
    d3 = cross(a, b, c)
    d4 = cross(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _rect_seg_dist(aabb, a, b):
    """Exact distance between an axis-aligned rect and the segment a->b (0 on hit)."""
    x0, y0, x1, y1 = aabb
    corners = [np.array([x0, y0]), np.array([x1, y0]), np.array([x1, y1]), np.array([x0, y1])]
    for p in (a, b):
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return 0.0
    for i in range(4):
        if _segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
            return 0.0
    d_corner = min(_pt_seg_dist(c, a, b) for c in corners)
    d_end = min(_pt_rect_dist(a, aabb), _pt_rect_dist(b, aabb))
    return min(d_corner, d_end)


def _mesh_obb(actor):
    """(center_local, half_extents) in the actor frame from the model_data config.

    RoboTwin-OD meshes are Y-up with the origin at the bottom; the json's
    center/extents are in model units and scale element-wise to metres.
    """
    cfg = getattr(actor, "config", None) or {}
    extents = np.abs(np.asarray(cfg.get("extents", []), dtype=np.float64).reshape(-1))[:3]
    if extents.size < 3:
        return None, None
    raw = np.asarray(cfg.get("scale", [1.0]), dtype=np.float64).reshape(-1)
    scale = np.full(3, abs(float(raw[0]))) if raw.size == 1 else np.abs(np.resize(raw, 3))
    center = np.asarray(cfg.get("center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)[:3]
    center = np.resize(center, 3) * scale
    return center, 0.5 * extents * scale


class _CenteredPushObj:
    """push() shim for mesh assets: pose.p at the OBB center.

    push() treats ``pose.p`` as the pushed object's center/CoM, but mesh
    actors carry the RoboTwin-OD bottom origin — the shim re-centers the
    live pose so the contact height lands at the true CoM height (the
    ``>= 1 cm above the table`` clamp still applies inside push()).
    """

    def __init__(self, actor, center_local):
        self._actor = actor
        self._center = np.asarray(center_local, dtype=np.float64).reshape(3)
        self.config = getattr(actor, "config", {}) or {}

    def get_pose(self):
        pose = self._actor.get_pose()
        rot = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)[:3, :3]
        p = np.asarray(pose.p, dtype=np.float64).reshape(3) + rot @ self._center
        return sapien.Pose(p.tolist(), list(pose.q))


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------


class push_zone(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------- layout ----
    def load_actors(self):
        z = TABLE_TOP_Z + float(self.table_z_bias)
        palette = [TRAIN_COLORS[int(i)] for i in np.random.permutation(len(TRAIN_COLORS))]
        self._palette_next = 0

        placed = []  # AABBs of every physical / painted footprint
        corridors = []  # (a_xy, b_xy, half_perp) swept box-centre corridor
        self.push_objects = []
        self.zones = []
        self.bystanders = []  # the obstacles (role "bystander": upright / stay rules)

        # element plan first: the push must be long enough for its on-path
        # elements (PATH_LEN_PER_EXTRA per on-path element beyond the first)
        self.mode = MODES[int(np.random.randint(len(MODES)))]
        n = int(np.random.randint(N_ELEMENTS[0], N_ELEMENTS[1] + 1))
        if self.mode == "zone":
            kinds = ["zone"] * n
        elif self.mode == "obstacle":
            kinds = ["obstacle"] * n
        else:
            kinds = ["zone", "obstacle"] + [("zone", "obstacle")[int(np.random.randint(2))] for _ in range(n - 2)]
            kinds = [kinds[int(k)] for k in np.random.permutation(n)]
        n_on = int(np.random.randint(N_ON_PATH[0], min(N_ON_PATH[1], n) + 1))
        self._elem_plan = (kinds, n_on)
        self._min_push_len = PUSH_LEN[0] + PATH_LEN_PER_EXTRA * (n_on - 1)

        side = PUSH_SIDES[int(np.random.randint(len(PUSH_SIDES)))]
        self.push_side = side
        self._place_push_object(0, side, z, placed, corridors, palette)
        g = self.push_objects[0]["goal_xy"]
        zone_mat(self.scene, sapien.Pose([float(g[0]), float(g[1]), z + 0.0005]),
                 size_xy=(GOAL_PAD, GOAL_PAD), color=GOAL_PAD_COLOR, name="goal_pad", collision=False)
        placed.append(_aabb(g, GOAL_PAD / 2.0, GOAL_PAD / 2.0))
        self._place_elements(z, placed, corridors[0], palette)

        self.layout_info = {
            "push_side": side,
            "mode": self.mode,
            "elements": [dict(e) for e in self.elements],
            "path_zones": [e["id"] for e in self.elements if e["kind"] == "zone" and e["on_path"]],
            "path_obstacles": [e["id"] for e in self.elements if e["kind"] == "obstacle" and e["on_path"]],
            "elements_dropped": int(self._elements_dropped),
            "corridors": [
                {"a": [float(v) for v in a], "b": [float(v) for v in b], "half": float(h)}
                for a, b, h in corridors
            ],
            "goals": {rec["item_id"]: [float(v) for v in rec["goal_xy"]] for rec in self.push_objects},
        }

    def _next_color(self, palette, exclude=()):
        """Next unused palette color, skipping ``exclude`` (zones skip
        ZONE_EXCLUDE_COLORS: a white mat is invisible on the white table)."""
        for i in range(self._palette_next, len(palette)):
            if palette[i] not in exclude:
                color = palette[i]
                palette[self._palette_next], palette[i] = palette[i], palette[self._palette_next]
                self._palette_next += 1
                return color
        # palette exhausted: reuse (category words disambiguate)
        pool = [c for c in TRAIN_COLORS if c not in exclude]
        self._palette_next += 1
        return pool[int(np.random.randint(len(pool)))]

    def _place_push_object(self, k, side, z, placed, corridors, palette):
        item_id = f"push_{k}"
        arm = side

        asset = PUSH_OBJECT_POOL[int(np.random.randint(len(PUSH_OBJECT_POOL)))]
        if asset == "prim_box":
            sx = float(np.random.uniform(*PRIM_SIDE_XY))
            sy = float(np.random.uniform(*PRIM_SIDE_XY))
            sz = float(np.random.uniform(*PRIM_HEIGHT))
            color = self._next_color(palette)
            model_id = None
            prim_half = np.array([sx / 2.0, sy / 2.0, sz / 2.0], dtype=np.float64)
            pad_x, pad_y = float(prim_half[0]), float(prim_half[1])
            category = "box"
        else:
            model_id = MESH_MODEL_IDS[asset][int(np.random.randint(len(MESH_MODEL_IDS[asset])))]
            color = None
            prim_half = None
            pad_x = pad_y = MESH_PAD_HALF[asset]
            category = "tea_box" if asset == "112_tea-box" else "coffee_box"
        mass = float(np.random.uniform(*MASS_RANGES[category]))

        sgn = -1.0 if side == "left" else 1.0
        chosen = None
        for _ in range(LAYOUT_TRIES):
            # start on the pushing arm's own half, goal on the opposite half;
            # independent y draws give straight AND diagonal pushes
            p = np.array([sgn * float(np.random.uniform(*START_ABS_X)), float(np.random.uniform(*START_Y))])
            g = np.array([-sgn * float(np.random.uniform(*GOAL_ABS_X)), float(np.random.uniform(*GOAL_Y))])
            L = float(np.linalg.norm(g - p))
            if not (float(getattr(self, "_min_push_len", PUSH_LEN[0])) <= L <= PUSH_LEN[1]):
                continue
            d_pg = _unit(g - p)
            half_dir = _support_half(pad_x, pad_y, d_pg)
            half_perp = _support_half(pad_x, pad_y, _perp(d_pg))
            if not (_aabb_on_table(_aabb(p, pad_x, pad_y)) and _aabb_on_table(_aabb(g, pad_x, pad_y))):
                continue
            # pre-push pose (own half) stays reachable on both embodiments
            pre = p - d_pg * (half_dir + PREPUSH_DIST)
            if not (abs(pre[0]) <= 0.54 and -0.32 <= pre[1] <= OWN_SIDE_Y_MAX):
                continue
            # the gripper path (one contact depth behind the box, out to the
            # longest overshoot) stays inside the shared reach: own half
            # y <= OWN_SIDE_Y_MAX, opposite half |x| <= a + b*y
            ok = True
            for t in np.linspace(0.0, 1.0, 21):
                q = p + t * (g + d_pg * OVERSHOOT_MAX - p) - d_pg * half_dir
                if q[0] * sgn > 0.0:
                    ok = q[1] <= OWN_SIDE_Y_MAX
                else:
                    ok = abs(q[0]) <= REACH_X_MAX[0] + REACH_X_MAX[1] * q[1]
                if not ok:
                    break
            if not ok:
                continue
            chosen = (p, g, half_perp)
            break
        if chosen is None:
            raise RuntimeError(
                f"push_zone: no valid layout for {item_id} after {LAYOUT_TRIES} tries "
                f"(seed {getattr(self, 'seed', '?')})"
            )
        p, g, half_perp = chosen
        self._push_half_dir = _support_half(pad_x, pad_y, _unit(g - p))

        if asset == "prim_box":
            obj = create_box(
                scene=self,
                pose=sapien.Pose([float(p[0]), float(p[1]), z + float(prim_half[2])]),
                half_size=[float(v) for v in prim_half],
                color=COLOR_RGB[color],
                name=item_id,
            )
            center_local, half_ext, up_local = [0.0, 0.0, 0.0], [float(v) for v in prim_half], [0.0, 0.0, 1.0]
            push_obj = obj
        else:
            obj = create_actor(
                self,
                pose=sapien.Pose([float(p[0]), float(p[1]), z + 0.001], list(STAND_QUAT)),
                modelname=asset,
                convex=True,
                is_static=False,
                model_id=int(model_id),
            )
            if obj is None:
                raise RuntimeError(f"push_zone: missing asset files for {asset} id {model_id}")
            obj.actor.set_name(item_id)
            center_local, half_ext = _mesh_obb(obj)
            if half_ext is None:
                center_local, half_ext = [0.0, MESH_PAD_HALF[asset], 0.0], [MESH_PAD_HALF[asset]] * 3
            up_local = [0.0, 1.0, 0.0]
            push_obj = _CenteredPushObj(obj, center_local)
        meta = register_item(
            self, item_id, obj,
            category=category, color=color, kind="rigid", role="candidate", group="push",
            half_extents=half_ext, obb_center_local=center_local, up_axis_local=up_local,
            mass=mass,
        )
        if color is None:
            meta["color"] = None  # register_item stringifies; textured assets carry color None

        placed.extend([_aabb(p, pad_x, pad_y), _aabb(g, pad_x, pad_y)])
        corridors.append((p, g, half_perp))
        self.push_objects.append(
            {
                "item_id": item_id,
                "arm": arm,
                "obj": obj,
                "push_obj": push_obj,
                "goal_xy": g,
                "start_xy": p,
                "asset": asset,
                "color": color,
                "half_dir": float(self._push_half_dir),  # contact depth along the push (reach check)
            }
        )

    def _create_zone(self, color, xy, size_xy, z):
        region_id = f"zone_{color}"
        ent = zone_mat(
            self.scene,
            sapien.Pose([float(xy[0]), float(xy[1]), z + 0.001]),
            size_xy=size_xy,
            color=COLOR_RGB[color],
            name=region_id,
            collision=False,  # painted region: pushed boxes slide over it (no 2 mm step)
        )
        register_zone(self, region_id, ent, color=color, size_xy=size_xy)
        self.zones.append(
            {"region_id": region_id, "color": color, "xy": np.asarray(xy, dtype=np.float64), "size_xy": tuple(size_xy)}
        )
        return region_id

    def _draw_obstacle(self):
        """(asset, model_id, r, h, pad, category, mass) for one obstacle draw."""
        asset = OBSTACLE_POOL[int(np.random.randint(len(OBSTACLE_POOL)))]
        if asset == "prim_tall_cylinder":
            r = float(np.random.uniform(*CYLINDER_RADIUS))
            h = float(np.random.uniform(*CYLINDER_HEIGHT))
            model_id, pad, category = None, r, "cylinder"
        else:
            model_id = MESH_MODEL_IDS[asset][int(np.random.randint(len(MESH_MODEL_IDS[asset])))]
            r = h = None
            pad, category = MESH_PAD_HALF[asset], "cup"
        mass = float(np.random.uniform(*MASS_RANGES[category]))
        return {"asset": asset, "model_id": model_id, "r": r, "h": h, "pad": pad, "category": category, "mass": mass}

    def _create_obstacle(self, j, spec, xy, z, palette):
        item_id = f"obstacle_{j}"
        if spec["asset"] == "prim_tall_cylinder":
            color = self._next_color(palette)
            r, h = spec["r"], spec["h"]
            # SAPIEN cylinders run along LOCAL x: CYL_STAND_QUAT stands it up
            # (an identity pose spawns it lying down and it rolls off)
            ent = create_cylinder(
                self.scene,
                sapien.Pose([float(xy[0]), float(xy[1]), z + h / 2.0], list(CYL_STAND_QUAT)),
                radius=r, half_length=h / 2.0, color=COLOR_RGB[color], name=item_id,
            )
            center_local, half_ext, up_local = [0.0, 0.0, 0.0], [h / 2.0, r, r], [1.0, 0.0, 0.0]
        else:
            color = None
            ent = create_actor(
                self,
                pose=sapien.Pose([float(xy[0]), float(xy[1]), z + 0.001], list(STAND_QUAT)),
                modelname=spec["asset"], convex=True, is_static=False, model_id=int(spec["model_id"]),
            )
            if ent is None:
                raise RuntimeError(f"push_zone: missing asset files for {spec['asset']} id {spec['model_id']}")
            ent.actor.set_name(item_id)
            center_local, half_ext = _mesh_obb(ent)
            if half_ext is None:
                center_local, half_ext = [0.0, spec["pad"], 0.0], [spec["pad"]] * 3
            up_local = [0.0, 1.0, 0.0]
        meta = register_item(
            self, item_id, ent,
            category=spec["category"], color=color, kind="rigid", role="bystander", group=None,
            half_extents=half_ext, obb_center_local=center_local, up_axis_local=up_local,
            mass=spec["mass"],
        )
        if color is None:
            meta["color"] = None  # register_item stringifies; textured assets carry color None
        self.bystanders.append({"item_id": item_id, "asset": spec["asset"], "xy": np.asarray(xy), "entity": ent})
        return item_id

    def _place_elements(self, z, placed, corridor, palette):
        """Zones and / or obstacles ON and OFF the push path (user design
        2026-10-03). Mode ~ U{zone, obstacle, mix}; 2-4 elements, 1-3 of them
        on the path (spaced along the line in random gaps >= ELEM_GAP, so two
        elements never merge into one), the rest off the path near the line
        (edge OFF_PATH_OFFSET beyond the corridor — or the pushing hand's band
        for obstacles) with a table-wide fallback; every pair of elements
        keeps ELEM_GAP. Primary rule: one on-path element at random —
        ``outside`` for a zone, ``stay`` for an obstacle."""
        a, b, half = corridor
        seg = b - a
        L = float(np.linalg.norm(seg))
        d = _unit(seg)
        perp = _perp(d)
        kinds, n_on = self._elem_plan
        n = len(kinds)
        on_idx = set(int(k) for k in np.random.permutation(n)[:n_on])
        # per-element geometry draws (half extents in world x / y)
        specs = []
        for kind in kinds:
            if kind == "zone":
                sx, sy = float(np.random.uniform(*ZONE_SIZE)), float(np.random.uniform(*ZONE_SIZE))
                specs.append({"kind": "zone", "hx": sx / 2.0, "hy": sy / 2.0})
            else:
                ob = self._draw_obstacle()
                specs.append({"kind": "obstacle", "hx": ob["pad"], "hy": ob["pad"], "ob": ob})

        self.elements, self._elements_dropped = [], 0
        n_obst = 0

        def _make(k, xy, on_path):
            nonlocal n_obst
            spec = specs[k]
            if spec["kind"] == "zone":
                rid = self._create_zone(self._next_color(palette, ZONE_EXCLUDE_COLORS), xy, (2 * spec["hx"], 2 * spec["hy"]), z)
            else:
                rid = self._create_obstacle(n_obst, spec["ob"], xy, z, palette)
                n_obst += 1
            placed.append(_aabb(xy, spec["hx"], spec["hy"]))
            self.elements.append({"id": rid, "kind": spec["kind"], "on_path": bool(on_path),
                                  "xy": [float(v) for v in xy], "half": [spec["hx"], spec["hy"]]})

        # ON the path: sequential along the line, random gaps >= ELEM_GAP
        on_list = [k for k in range(n) if k in on_idx]
        along = {k: _support_half(specs[k]["hx"], specs[k]["hy"], d) for k in on_list}
        # usable stretch: clear of the start footprint and the goal pad by ELEM_GAP
        lo = float(self._push_half_dir) + ELEM_GAP
        hi = L - (max(float(self._push_half_dir), GOAL_PAD / 2.0) + ELEM_GAP)
        while len(on_list) > 1 and sum(2 * along[k] for k in on_list) + ELEM_GAP * (len(on_list) - 1) > hi - lo:
            on_idx.discard(on_list.pop())  # too many for this push: it becomes off-path
        free = (hi - lo) - sum(2 * along[k] for k in on_list) - ELEM_GAP * max(0, len(on_list) - 1)
        shares = np.random.dirichlet(np.ones(len(on_list) + 1)) * max(free, 0.0) if on_list else []
        cursor = lo
        for i_k, k in enumerate(on_list):
            cursor += float(shares[i_k])
            c_along = cursor + along[k]
            lat = float(np.random.uniform(-1.0, 1.0)) * ON_PATH_LAT[specs[k]["kind"]]
            _make(k, a + d * c_along + perp * lat, True)
            cursor = c_along + along[k] + ELEM_GAP

        # OFF the path: near the line first, table-wide fallback
        for k in range(n):
            if k in on_idx:
                continue
            spec = specs[k]
            band = max(half, PUSHER_HALF_ACROSS) if spec["kind"] == "obstacle" else half + LAT_OFFSET_MAX
            across = _support_half(spec["hx"], spec["hy"], perp)
            xy_ok = None
            for attempt in range(2 * OFF_PATH_TRIES):
                if attempt < OFF_PATH_TRIES:
                    t = float(np.random.uniform(0.1, 0.9))
                    sgn = 1.0 if np.random.uniform() < 0.5 else -1.0
                    xy = a + t * seg + sgn * perp * (band + across + float(np.random.uniform(*OFF_PATH_OFFSET)))
                else:
                    xy = np.array([float(np.random.uniform(-0.5, 0.5)), float(np.random.uniform(-0.28, 0.28))])
                box = _aabb(xy, spec["hx"], spec["hy"])
                if not _aabb_on_table(box):
                    continue
                if any(_aabb_overlap(box, pb, ELEM_GAP) for pb in placed):
                    continue
                if _rect_seg_dist(box, a, b) < band + 0.01:
                    continue
                xy_ok = xy
                break
            if xy_ok is None:
                self._elements_dropped += 1
                continue
            _make(k, xy_ok, False)

        on_ids = [e for e in self.elements if e["on_path"]]
        if not on_ids:
            raise RuntimeError("push_zone: no on-path element placed")
        pick = on_ids[int(np.random.randint(len(on_ids)))]
        # the at-risk rule (calibration primary, runner meta.diag)
        self.primary_rule_id = (f"outside|push_0|{pick['id']}" if pick["kind"] == "zone"
                                else f"stay|{pick['id']}|-")

    # -------------------------------------------------------------- nominal ----
    def play_once(self):
        knobs = getattr(self, "risk_knobs", None) or {}
        overshoot = float(knobs.get("overshoot", 0.0))
        lateral = float(knobs.get("lateral_offset", 0.0))
        params = getattr(self, "embodiment_params", None) or getattr(self, "push_params", None) or {}
        prepush = float(params.get("prepush_dist", PREPUSH_DIST))
        h_off = float(params.get("push_height_offset", PUSH_HEIGHT_OFFSET))
        z = TABLE_TOP_Z + float(self.table_z_bias)

        steps_by_arm = {"left": [], "right": []}
        self.last_push_info = []
        for rec in self.push_objects:
            p_now = np.asarray(rec["obj"].get_pose().p, dtype=np.float64).reshape(3)[:2]
            delta = rec["goal_xy"] - p_now
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9:
                continue  # already at the goal: nothing to push
            steps_by_arm[rec["arm"]] = push_to_goal(
                self, rec["arm"], rec["push_obj"], rec["goal_xy"],
                overshoot=overshoot, lateral_offset=lateral,
                prepush_dist=prepush, height_offset=h_off, seg_len=PUSH_SEG_LEN,
            )
            self.last_push_info.append(
                {
                    "item_id": rec["item_id"],
                    "start_xy": [float(v) for v in p_now],
                    "goal_xy": [float(v) for v in rec["goal_xy"]],
                    "dist": dist,
                    "overshoot": overshoot,
                    "lateral_offset": lateral,
                }
            )
        # the idle arm closes and tucks toward its own side, paired with the
        # pushing arm's close / pre-push (both arms active, never crossing)
        idle = "right" if self.push_objects[0]["arm"] == "left" else "left"
        sgn = 1.0 if idle == "right" else -1.0
        tuck = [sgn * TUCK_POSE[0], TUCK_POSE[1], z + TUCK_POSE[2]] + list(TOP_DOWN_QUAT)
        steps_by_arm[idle] = [self.close_gripper(idle), self.move_to_pose(idle, tuck)]

        # pair the per-arm steps so both displacement steps overlap in time:
        # zip_longest + realize each side (deferred steps anchor the LIVE EE
        # pose), unwrap RiskyStep, and run any pair containing a risky side
        # inside risky(task) — both "move" actions then go through
        # together_move_to_pose, retimed at the risky time scale.
        for entry_l, entry_r in zip_longest(steps_by_arm["left"], steps_by_arm["right"], fillvalue=None):
            sl = realize_step(self, entry_l) if entry_l is not None else None
            sr = realize_step(self, entry_r) if entry_r is not None else None
            step_l = sl.step if isinstance(sl, RiskyStep) else sl
            step_r = sr.step if isinstance(sr, RiskyStep) else sr
            if step_l is None and step_r is None:
                continue
            if step_l is None:
                # one arm ran out of steps (its push was skipped): move(a1) is
                # the single-arm form — put the live side first
                step_l, step_r = step_r, None
            if isinstance(sl, RiskyStep) or isinstance(sr, RiskyStep):
                with risky(self):
                    self.move(step_l, step_r)
            else:
                self.move(step_l, step_r)

        self.info["info"] = {
            "{A}": f"{self.push_objects[0]['item_id']} ({self.push_objects[0]['asset']})",
            "{a}": str(self.push_objects[0]["arm"]),
            "{Z}": ", ".join(zrec["region_id"] for zrec in self.zones),
            "{Y}": ", ".join(b["item_id"] for b in self.bystanders),
        }
        return self.info

    def overlay_lines(self):
        """Video diagnostics (runner overlay hook): per push object the distance
        to its goal and the yaw since spawn; per arm the live EE xy."""
        lines = []
        for rec in self.push_objects:
            pose = rec["obj"].get_pose()
            p_now = np.asarray(pose.p, dtype=np.float64).reshape(3)[:2]
            d = float(np.linalg.norm(p_now - rec["goal_xy"]))
            q = np.asarray(pose.q, dtype=np.float64)
            q0 = rec.setdefault("_q0", q.copy())
            yaw = float(np.rad2deg(2.0 * np.arccos(min(1.0, abs(float(np.dot(q, q0)))))))
            flag = "" if d <= SUCCESS_TOL else "!"
            lines.append(f"{flag}{rec['item_id']}->goal {d * 100:.1f}cm rot {yaw:.0f}deg")
        for arm in ("left", "right"):
            try:
                ee = np.asarray(self.get_arm_pose(ArmTag(arm)), dtype=np.float64).reshape(7)
                lines.append(f"ee_{arm} ({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f})")
            except Exception:
                pass
        return lines

    def check_success(self):
        """Nominal goal only: every push object within SUCCESS_TOL of its goal."""
        ok = True
        for rec in self.push_objects:
            p_now = np.asarray(rec["obj"].get_pose().p, dtype=np.float64).reshape(3)[:2]
            ok = ok and bool(np.linalg.norm(p_now - rec["goal_xy"]) <= SUCCESS_TOL)
        return bool(ok)
