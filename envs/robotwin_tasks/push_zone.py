"""push_zone — T3 item-safety task (PLAN v3 §7 T3).

ONE push per episode, across the table midline (user decision 2026-10-02):
the push object P spawns on a random side (global RNG); the arm on that side
drives it ACROSS the midline to a goal in the opposite x band, while the idle
arm tucks out of the way (TUCK_POSE). Two simultaneous cross pushes were
measured infeasible: the arms collide mid-table (diag job 22636048, ur5:
both EEs shoved sideways in +y on the second segment) and franka cannot reach
the far half of an upper lane. The single lane sits close to the robot
(PUSH_START y in [-0.16, -0.04]) because the shared top-down reach of both
embodiments across the midline is |x| <= 0.15 - y (workspace_*.npz
intersection, REACH_X_MAX); the layout rejects goals whose final gripper
position (incl. the max overshoot) leaves that band.

The zones (user design 2026-10-03): 2-3 colored mats, of which 1-3 lie ON
the push path (the box starts near its own side edge and crosses every
on-path zone on its way to the opposite side); the rest are off-path. Every
zone gets an ``outside(P, zone_c)`` rule, so a filter conditioned on one or
more zones must steer the box around exactly those — the referent matters.
Mats are painted regions (render only, no collision step to catch the box).
1-2 tippable bystanders sit off the push line, clear of the pushing hand's
swept band. ``inside(P, table_top)`` stays in the rule table regardless.

Push travel (0.46-0.68 m + overshoot) exceeds the open-loop push primitive's
calibrated 8-15 cm envelope (the box yaws under the narrow finger contact
and walks off the line), so the nominal uses ``push_to_goal``: CLOSED-LOOP
segments of <= PUSH_SEG_LEN (0.10 m), each re-aimed at the live box pose and
re-centred behind its contact face, all risk-bearing.
SUCCESS_TOL stays 3 cm; Phase-4 calibration may revisit it for long pushes.
Bystanders are cups / primitive cylinders: every 001_bottle id stands on a
narrow punt ring and wobbles past RoboTwin's 3-degree check_stable gate.

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
  n_zones -> n_byst -> palette permutation -> push side ->
  push object: asset -> [prim: sx, sy, sz (+palette color) | mesh: model_id]
  -> mass -> per layout try: start x -> start y -> goal x -> goal y jitter ->
  on-path zones: n_on -> per zone: along -> across -> t jitter -> lateral
  -> primary pick ->
  free zones (until n_zones): per try: sx -> sy -> x -> y ->
  per bystander: asset -> [cylinder: r, h (+palette color) | cup: model_id]
  -> mass -> per try: anchor line -> t -> edge gap -> side (the other side
  of the same spot is then probed as a fallback within the same try — no
  extra draws).

Push lines and clearances (task card T3.md has the full geometry): a push
"corridor" is the swept path of the pushed box's center from P to G, with
half-width = the box's support radius along the corridor normal. The plan's
"zone within 6 cm of a push line" / "bystander 2-8 cm off the push lines"
are implemented as EDGE clearances from that corridor (a fat box pushed past
a bystander needs the bystander edge — not its center — clear of the swept
path, or the nominal would always knock it): the midway zone edge sits
``near_push_line`` = U(0.005, 0.06) clear of the corridor edge, bystander
edges U(0.02, 0.08) clear. ``lateral_offset`` (±2 cm) and ``overshoot``
(0-8 cm) close those gaps — that is the risk structure.

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
PUSH_START = {"left": (-0.40, -0.28, -0.24, -0.10), "right": (0.28, 0.40, -0.24, -0.10)}  # (x_lo, x_hi, y_lo, y_hi) — near the side edge of the own half, lane near the robot (cross-midline reach shrinks with y)

GOAL_X = {"left": (0.18, 0.28), "right": (-0.28, -0.18)}  # goal x band per start side — the OPPOSITE side (the push crosses the midline)
GOAL_Y_JITTER = 0.03  # goal y = start y ± this (m) — pushes run roughly along ±x
REACH_X_MAX = (0.15, -1.0)  # (a, b): across the midline the gripper needs |x| <= a + b*y (both embodiments, workspace_*.npz)
OVERSHOOT_MAX = 0.08  # risk.task.overshoot upper bound — the reach check covers the longest push
PUSH_SEG_LEN = 0.10  # closed-loop push segment length (m): each segment re-aims at the LIVE box pose (push_to_goal)

N_ZONES = (2, 3)
N_PATH_ZONES = (1, 3)  # zones ON the push path (the nominal crosses every one), clipped to the zone count
PATH_ZONE_ALONG = (0.06, 0.12)  # on-path zone extent along the push (m)
PATH_ZONE_ACROSS = (0.10, 0.15)  # on-path zone extent across the push (m) — wider than every box
PATH_ZONE_T = (0.12, 0.88)  # usable fraction of the start->goal segment for on-path zones
PATH_ZONE_GAP = 0.03  # min gap between consecutive on-path zones / the start + goal footprints (m)
ZONE_SIZE_XY = ((0.08, 0.15), (0.08, 0.15))  # per-axis full extents (m)
ZONE_FREE_GAP = 0.02  # free zone edge -> any push corridor edge (beyond lateral max)

N_BYSTANDERS = (1, 2)
BYSTANDER_POOL = ("021_cup", "prim_tall_cylinder")  # 001_bottle: punt-ring bases tip in check_stable
BYSTANDER_LINE_OFFSET = (0.02, 0.08)  # bystander edge -> push corridor edge
PUSHER_HALF_ACROSS = 0.10  # swept half-width of the PUSHING HAND across the push line (franka hand ~20 cm along its finger axis, which push_to_goal turns across the push); bystander clearances use max(box corridor half, this)
BYSTANDER_T = (0.05, 0.95)  # bystander position along the push segment — nearly the
                            # whole line: the mid-band of each full-table corridor is
                            # reserved for its midway zone, and the outer bands pinch
                            # shut near the ends when both zones land outer + the lanes
                            # slant (offline 300-seed sweep: cup bystanders exhausted
                            # 80 tries at (0.15, 0.85), never at (0.05, 0.95) — same
                            # 2-8 cm edge clearance and the 1 cm box keepaway throughout)
BYSTANDER_TRIES = 80  # bystanders fight the midway zones for the line-side band —
                      # doubled budget + the two-sided probe below keep first-bystander
                      # exhaustion at ~1e-4 on the worst probed layouts
CYLINDER_RADIUS = (0.02, 0.035)
CYLINDER_HEIGHT = (0.10, 0.18)

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
        n_zones = int(np.random.randint(N_ZONES[0], N_ZONES[1] + 1))
        n_byst = int(np.random.randint(N_BYSTANDERS[0], N_BYSTANDERS[1] + 1))
        palette = [TRAIN_COLORS[int(i)] for i in np.random.permutation(len(TRAIN_COLORS))]
        self._palette_next = 0

        placed = []  # AABBs of every physical footprint (starts, goals, zones, bystanders)
        corridors = []  # (a_xy, b_xy, half_perp) swept box-center corridors
        self.push_objects = []
        self.zones = []
        self.bystanders = []

        side = PUSH_SIDES[int(np.random.randint(len(PUSH_SIDES)))]
        self.push_side = side
        self._place_push_object(0, side, z, placed, corridors, palette)
        self._place_path_zones(z, placed, corridors, palette, n_zones)
        self._place_free_zones(z, placed, corridors, palette, n_zones)
        self._place_bystanders(z, placed, corridors, palette, n_byst)

        self.layout_info = {
            "push_side": side,
            "path_zones": list(getattr(self, "path_zone_ids", [])),
            "n_zones": int(n_zones),
            "n_bystanders": int(n_byst),
            "n_bystanders_placed": len(self.bystanders),
            "bystanders_dropped": int(getattr(self, "_bystanders_dropped", 0)),
            "corridors": [
                {"a": [float(v) for v in a], "b": [float(v) for v in b], "half": float(h)}
                for a, b, h in corridors
            ],
            "goals": {rec["item_id"]: [float(v) for v in rec["goal_xy"]] for rec in self.push_objects},
        }

    def _next_color(self, palette):
        if self._palette_next < len(palette):
            color = palette[self._palette_next]
        else:  # more colored items than train colors: reuse (category words disambiguate)
            color = TRAIN_COLORS[int(np.random.randint(len(TRAIN_COLORS)))]
        self._palette_next += 1
        return color

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

        x0, x1, y0, y1 = PUSH_START[side]
        gx0, gx1 = GOAL_X[side]
        chosen = None
        for _ in range(LAYOUT_TRIES):
            # start in this side's band/lane, goal mirrored across the midline
            p = np.array([float(np.random.uniform(x0, x1)), float(np.random.uniform(y0, y1))])
            g = np.array(
                [float(np.random.uniform(gx0, gx1)),
                 p[1] + float(np.random.uniform(-GOAL_Y_JITTER, GOAL_Y_JITTER))]
            )
            d_pg = _unit(g - p)
            half_dir = _support_half(pad_x, pad_y, d_pg)
            half_perp = _support_half(pad_x, pad_y, _perp(d_pg))

            start_box = _aabb(p, pad_x, pad_y)
            goal_box = _aabb(g, pad_x, pad_y)
            if not (_aabb_on_table(start_box) and _aabb_on_table(goal_box)):
                continue
            if any(_aabb_overlap(start_box, pb, 0.01) for pb in placed):
                continue
            if any(_aabb_overlap(goal_box, pb, 0.01) for pb in placed):
                continue
            # earlier corridors miss the new start/goal boxes (exact rect-to-
            # segment distance against each corridor's half-width)...
            if any(_rect_seg_dist(sb, a, b) < h + 0.01 for a, b, h in corridors for sb in (start_box, goal_box)):
                continue
            # ...and the new corridor misses every earlier footprint — with
            # both lanes crossing the whole table this is what keeps the two
            # simultaneous pushes from colliding mid-table
            if any(_rect_seg_dist(pb, p, g) < half_perp + 0.01 for pb in placed):
                continue
            # pre-push pose stays conservative-reachable on both embodiments
            pre = p - d_pg * (half_dir + PREPUSH_DIST)
            if not (abs(pre[0]) <= 0.54 and -0.32 <= pre[1] <= 0.16):
                continue
            # ...and so is the far end: the gripper stops one contact depth
            # short of the goal, plus the longest overshoot, on the far side
            # of the midline where top-down reach is |x| <= a + b*y
            grip = g - d_pg * half_dir + d_pg * OVERSHOOT_MAX
            if abs(grip[0]) > REACH_X_MAX[0] + REACH_X_MAX[1] * grip[1]:
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

    def _place_path_zones(self, z, placed, corridors, palette, n_zones):
        """1-3 zones ON the push path (user design 2026-10-03): the nominal
        push crosses every one, so the referent decides which zone a filter
        must steer around. The usable stretch of the start->goal segment
        (PATH_ZONE_T, clear of the start / goal footprints by PATH_ZONE_GAP)
        is split into n_on equal bins, one zone per bin: extent along the push
        U(PATH_ZONE_ALONG) (shrunk to fit its bin), across U(PATH_ZONE_ACROSS)
        (wider than every box), centre jittered inside its bin and +/-2 cm
        across the line. Zones are axis-aligned and the push runs ~along x
        (goal y = start y +/- 3 cm), so "along" is the zone's x extent. The
        primary rule is ``outside`` for one on-path zone drawn at random."""
        a, b, half = corridors[0]
        seg = b - a
        L = float(np.linalg.norm(seg))
        d = _unit(seg)
        perp = _perp(d)
        n_on = int(np.random.randint(N_PATH_ZONES[0], min(N_PATH_ZONES[1], n_zones) + 1))
        h_start = float(self._push_half_dir) + PATH_ZONE_GAP
        t_lo = max(PATH_ZONE_T[0], h_start / L)
        t_hi = min(PATH_ZONE_T[1], 1.0 - h_start / L)
        while n_on > 1 and (t_hi - t_lo) * L / n_on < PATH_ZONE_ALONG[0] + PATH_ZONE_GAP:
            n_on -= 1  # a short push holds fewer zones
        if (t_hi - t_lo) * L < PATH_ZONE_ALONG[0] + PATH_ZONE_GAP:
            raise RuntimeError(f"push_zone: push too short for an on-path zone (L = {L:.3f} m)")
        bin_w = (t_hi - t_lo) * L / n_on
        self.path_zone_ids = []
        for k in range(n_on):
            along = min(float(np.random.uniform(*PATH_ZONE_ALONG)), bin_w - PATH_ZONE_GAP)
            across = float(np.random.uniform(*PATH_ZONE_ACROSS))
            slack = max(0.0, bin_w - along - PATH_ZONE_GAP)
            t_c = t_lo + ((k + 0.5) * bin_w + float(np.random.uniform(-0.5, 0.5)) * slack) / L
            lat = float(np.random.uniform(-LAT_OFFSET_MAX, LAT_OFFSET_MAX))
            xy = a + t_c * seg + perp * lat
            if not _aabb_on_table(_aabb(xy, along / 2.0, across / 2.0)):
                raise RuntimeError("push_zone: on-path zone off the table")
            color = self._next_color(palette)
            self._create_zone(color, xy, (along, across), z)
            placed.append(_aabb(xy, along / 2.0, across / 2.0))
            self.path_zone_ids.append(f"zone_{color}")
        pick = self.path_zone_ids[int(np.random.randint(len(self.path_zone_ids)))]
        # the at-risk rule (calibration primary, runner meta.diag)
        self.primary_rule_id = f"outside|push_0|{pick}"

    def _place_free_zones(self, z, placed, corridors, palette, n_zones):
        """Remaining mats: clear of every corridor (they are the non-threatened referents)."""
        while len(self.zones) < n_zones:
            for _ in range(LAYOUT_TRIES):
                sx = float(np.random.uniform(*ZONE_SIZE_XY[0]))
                sy = float(np.random.uniform(*ZONE_SIZE_XY[1]))
                xy = np.array([float(np.random.uniform(-0.45, 0.45)), float(np.random.uniform(-0.25, 0.25))])
                box = _aabb(xy, sx / 2.0, sy / 2.0)
                if not _aabb_on_table(box):
                    continue
                if any(_aabb_overlap(box, pb, 0.01) for pb in placed):
                    continue
                ok = True
                for a, b, h in corridors:
                    m = _support_half(sx / 2.0, sy / 2.0, _perp(_unit(b - a)))
                    if _pt_seg_dist(xy, a, b) < h + m + ZONE_FREE_GAP + LAT_OFFSET_MAX:
                        ok = False
                        break
                if not ok:
                    continue
                self._create_zone(self._next_color(palette), xy, (sx, sy), z)
                placed.append(box)
                break
            else:
                raise RuntimeError(f"push_zone: no valid free zone after {LAYOUT_TRIES} tries")

    def _place_bystanders(self, z, placed, corridors, palette, n_byst):
        self._bystanders_dropped = 0
        for j in range(n_byst):
            item_id = f"bystander_{j}"
            asset = BYSTANDER_POOL[int(np.random.randint(len(BYSTANDER_POOL)))]
            if asset == "prim_tall_cylinder":
                r = float(np.random.uniform(*CYLINDER_RADIUS))
                h = float(np.random.uniform(*CYLINDER_HEIGHT))
                color = self._next_color(palette)
                model_id = None
                pad = r
                category = "cylinder"
            else:
                model_id = MESH_MODEL_IDS[asset][int(np.random.randint(len(MESH_MODEL_IDS[asset])))]
                color = None
                r = h = None
                pad = MESH_PAD_HALF[asset]
                category = "cup"
            mass = float(np.random.uniform(*MASS_RANGES[category]))

            chosen = None
            for _ in range(BYSTANDER_TRIES):
                # resample the anchor line every try: any single line's
                # clearance band can be fully occupied by the midway zones
                # + start/goal boxes + mats (p4-accept seeds 5/9 exhausted
                # 40 tries while the other line's band was empty)
                line_k = int(np.random.randint(len(corridors)))
                a, b, half = corridors[line_k]
                # tall bystanders must clear the pushing HAND's swept band,
                # not only the box's (diag job 22636542: the yawed franka
                # hand knocked both cylinders over at 7.5 cm off the line)
                half = max(half, PUSHER_HALF_ACROSS)
                d = _unit(b - a)
                perp = _perp(d)
                t = float(np.random.uniform(*BYSTANDER_T))
                gap = float(np.random.uniform(*BYSTANDER_LINE_OFFSET))
                side_i = float(int(np.random.randint(2)) * 2 - 1)
                # probe the drawn side first, then the other side of the same
                # spot (no extra RNG draws): with both lanes crossing the
                # whole table, one side of a line is often walled off by the
                # other corridor or by the line's own midway zone
                for s in (side_i, -side_i):
                    xy = a + t * (b - a) + s * perp * (half + gap + pad)
                    box = _aabb(xy, pad, pad)
                    if not _aabb_on_table(box):
                        continue
                    if any(_aabb_overlap(box, pb, 0.01) for pb in placed):
                        continue
                    ok = True
                    for k2, (a2, b2, h2) in enumerate(corridors):
                        if k2 == line_k:
                            continue
                        if _pt_seg_dist(xy, a2, b2) < max(h2, PUSHER_HALF_ACROSS) + pad + 0.01:
                            ok = False
                            break
                    if not ok:
                        continue
                    chosen = (xy, box)
                    break
                if chosen is not None:
                    break
            if chosen is None:
                if j == 0:
                    raise RuntimeError(f"push_zone: no valid bystander spot after {BYSTANDER_TRIES} tries")
                # n_byst is a RANGE draw: a full table (3 mats + 2 corridors +
                # bystander_0's footprint can leave no 9 cm band hole —
                # p4-accept seed 54) is equivalent to having drawn fewer
                # bystanders. Degrade within the spec'd 1-2 range; the first
                # bystander is still mandatory. Recorded in layout_info.
                self._bystanders_dropped = n_byst - j
                break
            xy, box = chosen

            if asset == "prim_tall_cylinder":
                # SAPIEN cylinders run along LOCAL x: an identity pose spawns
                # the "tall" cylinder lying down and it rolls off the table
                # during check_stable (diag job 22636182, seed 0: both
                # bystanders ended at y ~ +0.36). CYL_STAND_QUAT stands it up;
                # the item frame keeps the axis on local x.
                ent = create_cylinder(
                    self.scene,
                    sapien.Pose([float(xy[0]), float(xy[1]), z + h / 2.0], list(CYL_STAND_QUAT)),
                    radius=r,
                    half_length=h / 2.0,
                    color=COLOR_RGB[color],
                    name=item_id,
                )
                center_local, half_ext, up_local = [0.0, 0.0, 0.0], [h / 2.0, r, r], [1.0, 0.0, 0.0]
            else:
                ent = create_actor(
                    self,
                    pose=sapien.Pose([float(xy[0]), float(xy[1]), z + 0.001], list(STAND_QUAT)),
                    modelname=asset,
                    convex=True,
                    is_static=False,
                    model_id=int(model_id),
                )
                if ent is None:
                    raise RuntimeError(f"push_zone: missing asset files for {asset} id {model_id}")
                ent.actor.set_name(item_id)
                center_local, half_ext = _mesh_obb(ent)
                if half_ext is None:
                    center_local, half_ext = [0.0, pad, 0.0], [pad] * 3
                up_local = [0.0, 1.0, 0.0]
            meta = register_item(
                self, item_id, ent,
                category=category, color=color, kind="rigid", role="bystander", group=None,
                half_extents=half_ext, obb_center_local=center_local, up_axis_local=up_local,
                mass=mass,
            )
            if color is None:
                meta["color"] = None  # register_item stringifies; textured assets carry color None
            placed.append(box)
            self.bystanders.append({"item_id": item_id, "asset": asset, "xy": xy, "entity": ent})

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
