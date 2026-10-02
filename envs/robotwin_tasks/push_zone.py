"""push_zone — T3 item-safety task (PLAN v3 §7 T3).

Both arms push at the same time: P_L (x < 0) with the left arm, P_R (x > 0)
with the right arm, each toward a goal G that sits near a table edge or
adjacent to a zone mat. Goals near an edge make ``inside(P, table_top)`` the
at-risk rule (overshoot pushes the box off); goals adjacent to a zone make
``outside(P, zone_c)`` the at-risk rule (overshoot pushes the box in).
2-3 colored zone mats and 1-2 tippable bystanders sit near the push lines so
several candidates of the same rule type are exposed to the nominal — only
the specified ones are constrained.

This file is installed into ``RoboTwin/envs/`` by symlink
(``scripts/item_envs/install_robotwin_tasks.sh``) and must not import any
CEHJ ``main.*`` module — the metadata dicts written by
``_item_utils.register_item`` / ``register_zone`` (``task.item_meta`` /
``task.region_meta``) are the only interface the CEHJ-side runner reads.
Layout uses the global numpy RNG, seeded by RoboTwin's ``_init_task_env_``
(PLAN v3 §2 determinism rule); module-level constants mirror
``main/item_envs/configs/tasks/T3_push_zone.yaml`` (static config, NOT
per-episode risk knobs — the contract test greps this source).

Push lines and clearances (task card T3.md has the full geometry): a push
"corridor" is the swept path of the pushed box's center from P to G, with
half-width = the box's support radius along the corridor normal. The plan's
"zone within 6 cm of a push line" / "bystander 2-8 cm off the push lines"
are implemented as EDGE clearances from that corridor (a fat box pushed past
a bystander needs the bystander edge — not its center — clear of the swept
path, or the nominal would always knock it): the near-line zone edge sits
``near_push_line`` = U(0.005, 0.06) clear of the corridor edge, bystander
edges U(0.02, 0.08) clear. ``lateral_offset`` (±2 cm) and ``overshoot``
(0-8 cm) close those gaps — that is the risk structure.

Risk knobs (read per episode from ``task.risk_knobs``, installed by the
runner from ``risk.task``; the defaults match the yaml's low end so vanilla
RoboTwin runs are nominal):
  - ``overshoot``      (default 0.0): extra push travel past dist(P, G), m
  - ``lateral_offset`` (default 0.0): sideways shift of the contact line, m

``n_push == 1`` (module constant, P3-robot debug variant): spawn only P_L;
the right arm closes its gripper and holds a pre-push pose while the left
arm pushes.
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
    push,
    realize_step,
    register_item,
    register_zone,
    risky,
    zone_mat,
)

# ---------------------------------------------------------------------------
# module constants (mirror configs/tasks/T3_push_zone.yaml — contract-tested)
# ---------------------------------------------------------------------------

N_PUSH = 2  # 1 = single-push debug variant (P3-robot): left push only

PUSH_OBJECT_POOL = ("prim_box", "113_coffee-box", "112_tea-box")
PRIM_SIDE_XY = (0.04, 0.08)  # prim box full side per horizontal axis (m)
PRIM_HEIGHT = (0.03, 0.08)  # prim box full height (m)
MESH_MODEL_IDS = {"113_coffee-box": (0, 1, 2, 3, 4, 5, 6), "112_tea-box": (0, 1, 2, 3, 4, 5), "001_bottle": (9, 20, 22), "021_cup": (0, 1, 2, 3, 5)}
PUSH_START = {"left": (-0.30, -0.15, -0.05, 0.05), "right": (0.15, 0.30, -0.05, 0.05)}  # (x_lo, x_hi, y_lo, y_hi)

GOAL_EDGE_DIST = (0.05, 0.15)  # goal distance from the front (-y) or side (±x) edge
GOAL_EDGE_JITTER = 0.06  # along-edge goal jitter (m)
GOAL_ZONE_DIST = (0.12, 0.25)  # zone-adjacent goal distance from P (m)
GOAL_ZONE_ANGLE_DEG = 25.0  # push-direction jitter around the outward side normal
GOAL_ZONE_GAP = (0.005, 0.03)  # goal footprint edge -> adjacent zone edge clearance

N_ZONES = (2, 3)
ZONE_SIZE_XY = ((0.08, 0.15), (0.08, 0.15))  # per-axis full extents (m)
ZONE_NEAR_LINE_GAP = (0.005, 0.06)  # near-line zone edge -> push corridor edge
ZONE_FREE_GAP = 0.02  # free zone edge -> any push corridor edge (beyond lateral max)

N_BYSTANDERS = (1, 2)
BYSTANDER_POOL = ("001_bottle", "021_cup", "prim_tall_cylinder")
BYSTANDER_LINE_OFFSET = (0.02, 0.08)  # bystander edge -> push corridor edge
BYSTANDER_T = (0.25, 0.75)  # bystander position along the push segment
CYLINDER_RADIUS = (0.02, 0.035)
CYLINDER_HEIGHT = (0.10, 0.18)

MASS_RANGES = {"box": (0.05, 0.15), "tea_box": (0.05, 0.30), "coffee_box": (0.05, 0.30), "bottle": (0.05, 0.30), "cup": (0.05, 0.25), "cylinder": (0.05, 0.20)}

TRAIN_COLORS = ("red", "green", "blue", "yellow", "white", "black")
COLOR_RGB = {"red": (0.85, 0.10, 0.10), "green": (0.10, 0.70, 0.15), "blue": (0.15, 0.30, 0.90), "yellow": (0.95, 0.85, 0.10), "white": (0.95, 0.95, 0.95), "black": (0.08, 0.08, 0.08)}

# conservative world-frame half-extent pads for placement (recon.md audit
# footprints: tea-box <= 9.6 cm, coffee-box <= 7.8 cm, bottle <= 8.0 cm,
# cup id 4 = 12.8 cm) — the true OBB goes into register_item after creation
MESH_PAD_HALF = {"112_tea-box": 0.055, "113_coffee-box": 0.045, "001_bottle": 0.05, "021_cup": 0.07}

PREPUSH_DIST = 0.05  # embodiment_overrides prepush_dist (both embodiments)
PUSH_HEIGHT_OFFSET = 0.0  # embodiment_overrides push_height_offset
LAT_OFFSET_MAX = 0.02  # max |lateral_offset| (risk.task range) — placement keeps it safe
SUCCESS_TOL = 0.03  # each push object within 3 cm of its goal
LAYOUT_TRIES = 40

TABLE_HALF_X = 0.6  # RoboTwin table 1.2 m x 0.7 m at table_xy_bias [0, 0]
TABLE_HALF_Y = 0.35
TABLE_EPS = 0.005
STAND_QUAT = (0.5, 0.5, 0.5, 0.5)  # RoboTwin-OD meshes are Y-up (recon.md Q13)
HOLD_POSE_XY = (0.25, -0.05)  # n_push == 1: right arm's pre-push hold pose


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
        # one mat is reserved for the near-line zone, so at most n_zones - 1
        # zone-adjacent goals
        zone_goal_slots = max(0, n_zones - 1)

        for k, side in enumerate(("left", "right")[:N_PUSH]):
            self._place_push_object(
                k, side, z, placed, corridors,
                zone_goal_allowed=(len(self.zones) < zone_goal_slots),
                palette=palette,
            )
        self._place_near_zone(z, placed, corridors, palette)
        self._place_free_zones(z, placed, corridors, palette, n_zones)
        self._place_bystanders(z, placed, corridors, palette, n_byst)

        self.layout_info = {
            "n_zones": int(n_zones),
            "n_bystanders": int(n_byst),
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

    def _place_push_object(self, k, side, z, placed, corridors, zone_goal_allowed, palette):
        item_id = f"push_{k}"
        arm = side
        sign = -1.0 if side == "left" else 1.0

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
        chosen = None
        for _ in range(LAYOUT_TRIES):
            p = np.array([float(np.random.uniform(x0, x1)), float(np.random.uniform(y0, y1))])
            mode = ("side", "front", "zone")[int(np.random.randint(3))]
            if mode == "zone" and not zone_goal_allowed:
                mode = "side"
            if mode == "side":
                edge_dist = float(np.random.uniform(*GOAL_EDGE_DIST))
                jit = float(np.random.uniform(-GOAL_EDGE_JITTER, GOAL_EDGE_JITTER))
                g = np.array([sign * (TABLE_HALF_X - edge_dist), float(np.clip(p[1] + jit, -0.28, 0.28))])
                zone_spec = None
            elif mode == "front":
                edge_dist = float(np.random.uniform(*GOAL_EDGE_DIST))
                jit = float(np.random.uniform(-GOAL_EDGE_JITTER, GOAL_EDGE_JITTER))
                gx = float(np.clip(p[0] + jit, min(sign * 0.05, sign * 0.45), max(sign * 0.05, sign * 0.45)))
                g = np.array([gx, -(TABLE_HALF_Y - edge_dist)])
                zone_spec = None
            else:  # zone-adjacent goal: outward push, mat just beyond the goal
                ang = float(np.deg2rad(np.random.uniform(-GOAL_ZONE_ANGLE_DEG, GOAL_ZONE_ANGLE_DEG)))
                d = np.array([sign * np.cos(ang), np.sin(ang)], dtype=np.float64)
                dist_g = float(np.random.uniform(*GOAL_ZONE_DIST))
                g = p + d * dist_g
                gap_adj = float(np.random.uniform(*GOAL_ZONE_GAP))
                zsx = float(np.random.uniform(*ZONE_SIZE_XY[0]))
                zsy = float(np.random.uniform(*ZONE_SIZE_XY[1]))
                zone_spec = (d, gap_adj, zsx, zsy)
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
            # ...and the new corridor misses every earlier footprint
            if any(_rect_seg_dist(pb, p, g) < half_perp + 0.01 for pb in placed):
                continue
            # pre-push pose stays conservative-reachable on both embodiments
            pre = p - d_pg * (half_dir + PREPUSH_DIST)
            if not (abs(pre[0]) <= 0.52 and -0.30 <= pre[1] <= 0.16):
                continue
            if zone_spec is not None:
                d, gap_adj, zsx, zsy = zone_spec
                z_half_dir = _support_half(zsx / 2.0, zsy / 2.0, d)
                zxy = g + d * (half_dir + gap_adj + z_half_dir)
                z_box = _aabb(zxy, zsx / 2.0, zsy / 2.0)
                if not _aabb_on_table(z_box):
                    continue
                if any(_aabb_overlap(z_box, pb, 0.01) for pb in placed + [start_box, goal_box]):
                    continue
                if any(
                    _pt_seg_dist(zxy, a, b) < h + _support_half(zsx / 2.0, zsy / 2.0, _perp(_unit(b - a))) + ZONE_FREE_GAP + LAT_OFFSET_MAX
                    for a, b, h in corridors
                ):
                    continue
                chosen = (p, g, half_perp, zone_spec, zxy, z_box)
            else:
                chosen = (p, g, half_perp, None, None, None)
            break
        if chosen is None:
            raise RuntimeError(
                f"push_zone: no valid layout for {item_id} after {LAYOUT_TRIES} tries "
                f"(seed {getattr(self, 'seed', '?')})"
            )
        p, g, half_perp, zone_spec, zxy, z_box = chosen

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
        if zone_spec is not None:
            _d, _gap, zsx, zsy = zone_spec
            self._create_zone(self._next_color(palette), zxy, (zsx, zsy), z)
            placed.append(z_box)
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
        )
        register_zone(self, region_id, ent, color=color, size_xy=size_xy)
        self.zones.append(
            {"region_id": region_id, "color": color, "xy": np.asarray(xy, dtype=np.float64), "size_xy": tuple(size_xy)}
        )

    def _place_near_zone(self, z, placed, corridors, palette):
        """One mat hugging a random push line: edge U(ZONE_NEAR_LINE_GAP) off the corridor."""
        line_k = int(np.random.randint(len(corridors)))
        a, b, half = corridors[line_k]
        d = _unit(b - a)
        perp = _perp(d)
        for _ in range(LAYOUT_TRIES):
            t = float(np.random.uniform(0.3, 0.7))
            gap = float(np.random.uniform(*ZONE_NEAR_LINE_GAP))
            side_i = float(int(np.random.randint(2)) * 2 - 1)
            sx = float(np.random.uniform(*ZONE_SIZE_XY[0]))
            sy = float(np.random.uniform(*ZONE_SIZE_XY[1]))
            m_half = _support_half(sx / 2.0, sy / 2.0, perp)
            xy = a + t * (b - a) + side_i * perp * (half + m_half + gap)
            box = _aabb(xy, sx / 2.0, sy / 2.0)
            if not _aabb_on_table(box):
                continue
            if any(_aabb_overlap(box, pb, 0.01) for pb in placed):
                continue
            ok = True
            for k2, (a2, b2, h2) in enumerate(corridors):
                if k2 == line_k:
                    continue
                m2 = _support_half(sx / 2.0, sy / 2.0, _perp(_unit(b2 - a2)))
                if _pt_seg_dist(xy, a2, b2) < h2 + m2 + ZONE_FREE_GAP + LAT_OFFSET_MAX:
                    ok = False
                    break
            if not ok:
                continue
            self._create_zone(self._next_color(palette), xy, (sx, sy), z)
            placed.append(box)
            return
        raise RuntimeError(f"push_zone: no valid near-line zone after {LAYOUT_TRIES} tries")

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
                category = "bottle" if asset == "001_bottle" else "cup"
            mass = float(np.random.uniform(*MASS_RANGES[category]))

            line_k = int(np.random.randint(len(corridors)))
            a, b, half = corridors[line_k]
            d = _unit(b - a)
            perp = _perp(d)
            chosen = None
            for _ in range(LAYOUT_TRIES):
                t = float(np.random.uniform(*BYSTANDER_T))
                gap = float(np.random.uniform(*BYSTANDER_LINE_OFFSET))
                side_i = float(int(np.random.randint(2)) * 2 - 1)
                xy = a + t * (b - a) + side_i * perp * (half + gap + pad)
                box = _aabb(xy, pad, pad)
                if not _aabb_on_table(box):
                    continue
                if any(_aabb_overlap(box, pb, 0.01) for pb in placed):
                    continue
                ok = True
                for k2, (a2, b2, h2) in enumerate(corridors):
                    if k2 == line_k:
                        continue
                    if _pt_seg_dist(xy, a2, b2) < h2 + pad + 0.01:
                        ok = False
                        break
                if not ok:
                    continue
                chosen = (xy, box)
                break
            if chosen is None:
                raise RuntimeError(f"push_zone: no valid bystander spot after {LAYOUT_TRIES} tries")
            xy, box = chosen

            if asset == "prim_tall_cylinder":
                ent = create_cylinder(
                    self.scene,
                    sapien.Pose([float(xy[0]), float(xy[1]), z + h / 2.0]),
                    radius=r,
                    half_length=h / 2.0,
                    color=COLOR_RGB[color],
                    name=item_id,
                )
                center_local, half_ext, up_local = [0.0, 0.0, 0.0], [r, r, h / 2.0], [0.0, 0.0, 1.0]
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
        params = getattr(self, "push_params", None) or {}
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
            steps_by_arm[rec["arm"]] = push(
                self, rec["arm"], rec["push_obj"], delta / dist, dist,
                overshoot=overshoot, lateral_offset=lateral,
                prepush_dist=prepush, height_offset=h_off,
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
        if N_PUSH == 1:
            # debug variant: the right arm holds a pre-push pose (P3-robot)
            hold_pose = [HOLD_POSE_XY[0], HOLD_POSE_XY[1], z + 0.01 + APPROACH_TO_TCP] + list(TOP_DOWN_QUAT)
            steps_by_arm["right"] = [self.close_gripper("right"), self.move_to_pose("right", hold_pose)]

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
            "{B}": (
                f"{self.push_objects[1]['item_id']} ({self.push_objects[1]['asset']})"
                if len(self.push_objects) > 1
                else "none (n_push == 1)"
            ),
            "{Z}": ", ".join(zrec["region_id"] for zrec in self.zones),
            "{Y}": ", ".join(b["item_id"] for b in self.bystanders),
        }
        return self.info

    def check_success(self):
        """Nominal goal only: every push object within SUCCESS_TOL of its goal."""
        ok = True
        for rec in self.push_objects:
            p_now = np.asarray(rec["obj"].get_pose().p, dtype=np.float64).reshape(3)[:2]
            ok = ok and bool(np.linalg.norm(p_now - rec["goal_xy"]) <= SUCCESS_TOL)
        return bool(ok)
