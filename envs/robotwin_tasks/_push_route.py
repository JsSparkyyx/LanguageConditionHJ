"""T3 push_zone safe mode: the route planner (pure numpy; nominal_mode="safe", Yuxuan 2026-10-27).

The careless push runs straight from the box to the goal frame, across the mats and past the cylinders on the
line, on purpose. The safe push follows a polyline of via points that keeps every PROTECTED element clear:

- a mat: the box footprint stays ``mat_clear`` outside it (the hand passes over mats);
- a cylinder: the box stays ``cyl_clear_box`` and the WSG hand ``cyl_clear_hand`` away from it, so it is neither
  touched, tipped nor moved; the hand counts by the layers that reach the cylinder's height (the 14.6 cm base,
  turned across the push, the closed fingers, the wrist);
- the table: the box footprint stays ``table_clear`` inside its edges, and every EE position is reachable.

Unprotected elements are ignored (a named rule protects its own elements only; the demo collector names none, and
then every element is protected).

Checked, not trusted: the box centre is sampled every ``ds`` along each leg at every yaw it may take there (a push
turns the box until a face lies flush with the finger pair, so the yaws run from its yaw at the leg start to the
face-aligned one, +- ``yaw_allow_deg``); the hand sits where ``_item_utils.push_via`` puts it: behind the box
during a push (backed off up to ``backoff``), backed off and lifted over the box at a corner that turns more than
``lift_turn_deg`` (backed off and swung at push height below it), lowered ``descend_gap`` behind the new face.
Distances are exact (rect / rect by separating axes and vertex distances, rect / disk).

Search: the straight route first. Otherwise every protected element the straight route breaks gets a detour lane,
a line parallel to the push at the lateral offset that clears it (neighbours passed on the same side may share one
lane), entered and left at via points on a grid along the line, every turn <= ``max_turn_deg``. The candidates are
sorted by cost (length + ``w_turn`` x the summed squared turns + ``w_corner`` per via, ties to the near side) and
checked in that order; the first that passes is the route. Deterministic, no RNG.

Lives in robotwin_tasks/ (push_zone imports it in RoboTwin's process): no main.* import, no sapien.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field, fields

import numpy as np

TABLE_HALF_XY = (0.6, 0.35)   # push_zone TABLE_HALF_X / TABLE_HALF_Y (contract-tested)
OWN_SIDE_Y_MAX = 0.10         # push_zone's linear reach bound (contract-tested): own half y <= 0.10,
REACH_X_MAX = (0.15, -1.0)    # far half |x| <= 0.15 - y
YAW_STEP = math.radians(7.5)  # box yaw sampling step (a 9 cm diagonal moves < 1 mm between samples)
ALIGN_NEAR = math.radians(40.0)  # a push this far off every face normal meets a corner: the box may turn either way
ALIGN_LEN = 0.06              # a leg this long turns the box flush with the fingers; after shorter ones both yaws stay
LIFT_BAND_DEG = 10.0          # corners this close to lift_turn_deg are checked both ways (live turns differ a little)


# ---------------------------------------------------------------------------
# scene description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Elem:
    """A scene element: a painted mat ("zone", axis-aligned, ``half`` = its half sides) or a standing cylinder
    ("obstacle", ``half`` = (r, r), ``height`` above the table)."""

    id: str
    kind: str
    xy: tuple
    half: tuple
    height: float = 0.0


@dataclass(frozen=True)
class Layer:
    """A height band of the hand in top view: ``z_lo``..``z_hi`` above the EE origin (below it < 0); a rect centred
    on the EE (``a`` half along the finger axis, ``b`` half across it) or a disk of radius ``a``."""

    name: str
    z_lo: float
    z_hi: float
    shape: str
    a: float
    b: float = 0.0


@dataclass(frozen=True)
class Hand:
    """The pushing hand: its layers and the push mode, "wide" (push_to_goal's: the finger axis across the push,
    both fingers on the box) or "narrow" (the finger axis along the push, one finger on the box)."""

    mode: str
    layers: tuple

    def layer(self, name: str) -> Layer:
        return next(lay for lay in self.layers if lay.name == name)

    @property
    def contact(self) -> float:
        """The finger front face, ahead of the EE origin along the push (m)."""
        f = self.layer("fingers")
        return f.b if self.mode == "wide" else f.a

    @property
    def tip_len(self) -> float:
        return -self.layer("fingers").z_lo

    def across(self, lay: Layer) -> float:
        """The layer's half width across the push."""
        if lay.shape == "disk":
            return lay.a
        return lay.a if self.mode == "wide" else lay.b

    def finger_angle(self, phi):
        """World angle of the finger axis for a push heading ``phi``."""
        return phi + math.pi / 2.0 if self.mode == "wide" else phi

    @property
    def radius(self) -> float:
        """The hand's farthest point from the EE axis in top view (m): a turn of t rad moves it t x radius."""
        return max(lay.a if lay.shape == "disk" else math.hypot(lay.a, lay.b) for lay in self.layers)


def wsg_hand(mode: str = "wide", tip_len: float = 0.164) -> Hand:
    """ur5-wsg's WSG-50 hand in top view (safe_plan.HAND_BOXES and ur5_wsg_gripper.urdf). RoboTwin's EE pose
    (get_arm_pose) sits 1 cm above the WSG base link: it is ee_link moved gripper_bias - 0.12 = -0.01 m along the
    approach axis (robot._trans_endpose), and wsg_base_joint has its origin on ee_link. From the EE origin: the base
    14.6 x 5 cm from 1 cm down to 8.25 cm down; the closed fingers with their guides 4.5 x 3 cm from 6.94 cm down to
    the fingertips ``tip_len`` down (the yaml's push_tip_len, measured: 0.164 = 0.01 + 0.023 + 0.1305 by the URDF);
    the wrist_3 link, a 4.3 cm disk from 0.95 cm down to 2.5 cm up (its collision mesh). The arm above stays above
    every cylinder: wrist_2 starts 2.5 cm above the EE origin, which sits 17.2 cm above the table during a push,
    and the cylinders are at most 18 cm tall."""
    if mode not in ("wide", "narrow"):
        raise ValueError(f"hand mode must be 'wide' or 'narrow', got {mode!r}")
    return Hand(mode, (
        Layer("base", -0.0825, -0.01, "rect", 0.073, 0.025),
        Layer("fingers", -float(tip_len), -0.0694, "rect", 0.0227, 0.015),
        Layer("wrist", -0.0095, 0.025, "disk", 0.0432),
    ))


@dataclass
class RouteCfg:
    """The planner's and the executor's knobs (T3_push_zone.yaml safe_mode.route; unknown keys are ignored)."""

    mat_clear: float = 0.015       # box footprint -> protected mat (m)
    cyl_clear_box: float = 0.02    # box footprint -> protected cylinder
    cyl_clear_hand: float = 0.015  # every hand layer at the cylinder's height -> protected cylinder
    table_clear: float = 0.02      # box footprint -> table edge
    guard_clear: float = 0.008     # push_via's live re-check before every move: the hard floor of the four above
    lane_slack: float = 0.005      # detour lanes this much wider than the clearances ask (tracking error)
    yaw_allow_deg: float = 15.0    # box yaw spread on top of the face-aligned turn
    max_turn_deg: float = 90.0     # largest turn at a via point (above lift_turn_deg the hand lifts over the box)
    max_first_deg: float = 90.0    # largest first-leg angle off the straight start -> goal line (90 = sideways)
    lift_turn_deg: float = 30.0    # corners turning more lift the hand over the box (push_via)
    min_leg: float = 0.04          # shortest leg
    hand: str = "wide"
    reach: str = "npz"             # npz | linear | either | both | none (make_reach)
    reach_erode: int = 0           # npz cells peeled off the reach border
    seg_len: float = 0.06          # executor: push segment length (each re-aimed at the live box)
    via_tol: float = 0.012         # executor: a via point counts as reached within this
    lane_tol: float = 0.02         # executor: the box this far off its leg line re-plans
    approach_gap: float = 0.015    # executor: finger gap to the box face before a push
    descend_gap: float = 0.03      # executor: the hand comes down this far behind the new face
    backoff: float = 0.03          # executor: back-off before a corner
    realign_backoff: float = 0.02  # executor: back-off before a small re-alignment at push height
    corner_lift: float = 0.02      # executor: lifted fingertips this far above the box top
    approach_lift: float = 0.10    # executor: first approach from this far above the push pose
    tip_clear: float = 0.008       # fingertip bottoms above the table during a push (push_to_goal's)
    z_clear: float = 0.01          # a hand layer meets a cylinder when it reaches below its top + this
    ds: float = 0.005              # sampling step along legs and moves
    grid: int = 9                  # via positions per lane end
    w_turn: float = 0.05           # m per rad^2 of turn
    w_corner: float = 0.02         # m per via point
    w_side: float = 0.05           # m per element passed on another side than ``prefer`` (re-plans)
    max_evals: int = 900           # legs evaluated (not cached) before giving up; one plan uses at most a quarter

    @classmethod
    def from_dict(cls, d: dict | None) -> "RouteCfg":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class Route:
    """plan_route's answer. ``sides``: per element the route goes around, "near" / "far" (its -y / +y side; "left"
    / "right" of the push for pushes along y). ``detail``: the search (blocking elements, lane checks, failures)."""

    ok: bool
    via: list = field(default_factory=list)
    sides: dict = field(default_factory=dict)
    hand: str = "wide"
    turns_deg: list = field(default_factory=list)
    first_deg: float = 0.0
    length: float = 0.0
    min_clear: dict = field(default_factory=dict)
    reason: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-ready (mm / 0.1 deg precision), for task.episode_info["route"]."""
        return {
            "ok": bool(self.ok),
            "via": [[round(float(v[0]), 4), round(float(v[1]), 4)] for v in self.via],
            "sides": dict(self.sides),
            "hand": self.hand,
            "turns_deg": [round(float(t), 1) for t in self.turns_deg],
            "first_deg": round(float(self.first_deg), 1),
            "length": round(float(self.length), 4),
            "min_clear": {k: round(float(v), 4) for k, v in self.min_clear.items()},
            "reason": self.reason,
            "detail": _jsonable(self.detail),
        }


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    return v


# ---------------------------------------------------------------------------
# reach
# ---------------------------------------------------------------------------


def reach_linear(arm: str):
    """push_zone._reach_ok, the layout's bound: own half y <= 0.10, far half |x| <= 0.15 - y."""
    sgn = -1.0 if arm == "left" else 1.0

    def ok(xy):
        xy = np.asarray(xy, dtype=np.float64)
        x, y = xy[..., 0], xy[..., 1]
        return np.where(x * sgn > 0.0, y <= OWN_SIDE_Y_MAX, np.abs(x) <= REACH_X_MAX[0] + REACH_X_MAX[1] * y)

    return ok


def reach_npz(path, arm: str, erode: int = 0, heights=(0.10, 0.20)):
    """Top-down reach from workspace_<embodiment>.npz (scripts/item_envs/workspace_sweep.py: a cuRobo plan from the
    home pose to a top-down pose, 2 cm cells, the LEFT arm; the right arm is its mirror about x = 0). A cell counts
    when every layer of ``heights`` (m above the table; the push EE sits at 0.17, a corner lift near 0.25) is
    feasible; ``erode`` cells are peeled off the border (edge cells replicated). Returns ok(xy [..., 2])."""
    z = np.load(str(path))
    xs, ys = np.asarray(z["x"], dtype=np.float64), np.asarray(z["y"], dtype=np.float64)
    hts = np.asarray(z["heights_above_table"], dtype=np.float64)
    feas = np.asarray(z["feasible"], dtype=bool)
    m = np.logical_and.reduce([feas[int(np.argmin(np.abs(hts - h)))] for h in heights])
    for _ in range(int(erode)):
        p = np.pad(m, 1, mode="edge")
        m = m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    x0, dx, y0, dy = float(xs[0]), float(xs[1] - xs[0]), float(ys[0]), float(ys[1] - ys[0])
    ny, nx = m.shape
    mirror = -1.0 if arm == "right" else 1.0

    def ok(xy):
        xy = np.asarray(xy, dtype=np.float64)
        ix = np.rint((mirror * xy[..., 0] - x0) / dx).astype(np.int64)
        iy = np.rint((xy[..., 1] - y0) / dy).astype(np.int64)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.zeros(xy.shape[:-1], dtype=bool)
        out[inside] = m[iy[inside], ix[inside]]
        return out

    return ok


def make_reach(kind: str, arm: str, npz=None, erode: int = 0):
    """The reach model: "npz" (the measured grid; "linear" when the file is missing), "linear" (the layout's bound),
    "either" / "both" (their union / intersection), "none" (no reach check)."""
    if kind == "none":
        return None
    lin = reach_linear(arm)
    if kind == "linear" or npz is None:
        return lin
    grid = reach_npz(npz, arm, erode=erode)
    if kind == "npz":
        return grid
    if kind == "either":
        return lambda xy: grid(xy) | lin(xy)
    if kind == "both":
        return lambda xy: grid(xy) & lin(xy)
    raise ValueError(f"unknown reach model {kind!r}")


# ---------------------------------------------------------------------------
# exact 2D distances (numpy broadcasting over every argument)
# ---------------------------------------------------------------------------


def _rect_pt_dist(px, py, cx, cy, ang, a, b):
    """Distance from points to rects (centre cx, cy; ``a`` half along the axis at angle ``ang``, ``b`` half across
    it), 0 inside."""
    dx, dy = px - cx, py - cy
    c, s = np.cos(ang), np.sin(ang)
    qa = np.abs(dx * c + dy * s) - a
    qb = np.abs(dy * c - dx * s) - b
    return np.hypot(np.maximum(qa, 0.0), np.maximum(qb, 0.0))


def _rect_corners(cx, cy, ang, a, b):
    """x, y of the 4 rect corners, shape [..., 4]."""
    cx, cy, ang, a, b = (np.asarray(v, dtype=np.float64)[..., None] for v in (cx, cy, ang, a, b))
    la = np.array([1.0, -1.0, -1.0, 1.0]) * a
    lb = np.array([1.0, 1.0, -1.0, -1.0]) * b
    c, s = np.cos(ang), np.sin(ang)
    return cx + la * c - lb * s, cy + la * s + lb * c


def _support(ang_u, ang_r, a, b):
    """Half extent along the direction ``ang_u`` of a rect whose ``a`` axis lies at angle ``ang_r``."""
    t = ang_u - ang_r
    return a * np.abs(np.cos(t)) + b * np.abs(np.sin(t))


def _rect_rect_dist(r1, r2):
    """Exact distance between rects r = (cx, cy, ang, a, b), 0 when they overlap: separating axes decide the
    overlap, else the closest vertex-to-rect pair is the distance (for convex polygons one end is a vertex)."""
    x1, y1, t1, a1, b1 = (np.asarray(v, dtype=np.float64) for v in r1)
    x2, y2, t2, a2, b2 = (np.asarray(v, dtype=np.float64) for v in r2)
    k1x, k1y = _rect_corners(x1, y1, t1, a1, b1)
    k2x, k2y = _rect_corners(x2, y2, t2, a2, b2)
    e = lambda v: v[..., None]  # noqa: E731
    dist = np.minimum(_rect_pt_dist(k1x, k1y, e(x2), e(y2), e(t2), e(a2), e(b2)).min(-1),
                      _rect_pt_dist(k2x, k2y, e(x1), e(y1), e(t1), e(a1), e(b1)).min(-1))
    dx, dy = x2 - x1, y2 - y1
    sep = np.zeros(dist.shape, dtype=bool)
    for t in (t1, t1 + np.pi / 2.0, t2, t2 + np.pi / 2.0):
        sep |= np.abs(dx * np.cos(t) + dy * np.sin(t)) > _support(t, t1, a1, b1) + _support(t, t2, a2, b2)
    return np.where(sep, dist, 0.0)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _aligned(theta_ref: float, phi: float) -> list:
    """Box yaws with a face flush with the fingers for a push heading ``phi``: phi + k * 90 deg nearest
    ``theta_ref``, and the other neighbour too when the push meets a corner (> ALIGN_NEAR off the face normals)."""
    q = math.pi / 2.0
    t = phi + q * round((theta_ref - phi) / q)
    out = [t]
    if abs(theta_ref - t) > ALIGN_NEAR:
        out.append(t + math.copysign(q, theta_ref - t))
    return out


def quat_finger_angle(q) -> float:
    """World angle of the finger axis (the EE y axis, horizontal part) of an EE quaternion (wxyz)."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(1.0 - 2.0 * (x * x + z * z), 2.0 * (x * y - w * z))


# ---------------------------------------------------------------------------
# samples and checks
# ---------------------------------------------------------------------------


class _Ctx:
    """What one planning call shares: config, hand, box, protected elements, reach, heights."""

    def __init__(self, cfg, hand, box_half_xy, box_h, reach_ok, elems, protect, ee_height):
        self.cfg = cfg
        self.hand = hand
        self.bx, self.by = (float(v) for v in np.asarray(box_half_xy, dtype=np.float64).reshape(-1)[:2])
        self.box_h = float(box_h)
        self.reach_ok = reach_ok
        want = None if protect is None else {str(i) for i in protect}
        self.elems = [e for e in elems if want is None or e.id in want]
        self.ee_h = float(ee_height) if ee_height is not None else hand.tip_len + float(cfg.tip_clear)
        self.lift_h = max(self.ee_h, self.box_h + float(cfg.corner_lift) + hand.tip_len)
        self.plan_clear = {"mat": cfg.mat_clear, "cyl_box": cfg.cyl_clear_box, "cyl_hand": cfg.cyl_clear_hand,
                           "table": cfg.table_clear}
        self.guard_clear = _floors(cfg.guard_clear)


class _Acc:
    """Sample accumulator: box poses (centre, yaw), hand poses (EE xy, finger angle, pads along / across the finger
    axis, EE height above the table), reach points; each tagged with its leg."""

    def __init__(self):
        self.box = []
        self.hand = []
        self.reach = []

    def add_box(self, x, y, th, leg):
        """Centres x, y [n] times yaws th [m]."""
        n, m = x.size, th.size
        a = np.empty((n * m, 4))
        a[:, 0], a[:, 1], a[:, 2], a[:, 3] = np.repeat(x, m), np.repeat(y, m), np.tile(th, n), leg
        self.box.append(a)

    def add_hand(self, x, y, ang, pa, pb, z, leg):
        a = np.empty((x.size, 7))
        a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6] = x, y, ang, pa, pb, z, leg
        self.hand.append(a)

    def add_reach(self, x, y, leg):
        a = np.empty((x.size, 3))
        a[:, 0], a[:, 1], a[:, 2] = x, y, leg
        self.reach.append(a)

    def arrays(self):
        cat = lambda parts, n: np.concatenate(parts, axis=0) if parts else np.zeros((0, n))  # noqa: E731
        return cat(self.box, 4), cat(self.hand, 7), cat(self.reach, 3)


def _n_samples(ctx, dist, dang):
    """Samples of a hand move: the EE and the hand's farthest point (Hand.radius) step at most ds."""
    return max(2, int(math.ceil(float(dist) / ctx.cfg.ds)) + 1,
               int(math.ceil(abs(float(dang)) * ctx.hand.radius / ctx.cfg.ds)) + 1)


def _move_samples(acc, ctx, p0, p1, ang0, ang1, pad, z, leg, long=False):
    """Hand samples of one straight EE move p0 -> p1 at height z, the finger axis turning ang0 -> ang1 the short
    way (the hand is symmetric under 180 deg); ``long``: the other way round (push_via's 180-deg yaw switch at a
    lifted corner, the wrist_3 limit)."""
    dang = float((ang1 - ang0 + math.pi / 2.0) % math.pi - math.pi / 2.0)
    if long:
        dang -= math.copysign(math.pi, dang) if dang != 0.0 else math.pi
    t = np.linspace(0.0, 1.0, _n_samples(ctx, np.linalg.norm(p1 - p0), dang))
    x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
    acc.add_hand(x, y, ang0 + t * dang, pad, pad, z, leg)
    acc.add_reach(x, y, leg)


def _leg(ctx, a, b, prev, lo, hi, k, acc=None):
    """Leg k, a -> b, with the corner at ``a`` (``prev``: the previous leg's (heading, mid contact depth, depth
    spread, finger angle); None on the first leg) and the box yaw range lo..hi at its start: its samples go into
    ``acc`` (none when None); returns (prev, lo, hi) for the next leg."""
    cfg, hand = ctx.cfg, ctx.hand
    alpha = math.radians(cfg.yaw_allow_deg)
    length = float(np.linalg.norm(b - a))
    d = (b - a) / length
    phi = math.atan2(d[1], d[0])
    al = _aligned(0.5 * (lo + hi), phi)

    def yaws(t_lo, t_hi):
        t_lo, t_hi = t_lo - alpha, t_hi + alpha
        return np.linspace(t_lo, t_hi, max(3, int(math.ceil((t_hi - t_lo) / YAW_STEP)) + 1))

    # the first ALIGN_LEN of the leg: any yaw from the start range to flush with the fingers; after it, flush
    parts = [(0.0, min(length, ALIGN_LEN), yaws(min(lo, *al), max(hi, *al)))]
    if length > ALIGN_LEN:
        parts.append((ALIGN_LEN, length, yaws(min(al), max(al))))
    deps = [np.abs(np.cos(phi - th)) * ctx.bx + np.abs(np.sin(phi - th)) * ctx.by for _, _, th in parts]
    dmin, dmax = float(min(dp.min() for dp in deps)), float(max(dp.max() for dp in deps))  # contact depths
    mid, spread = 0.5 * (dmin + dmax), 0.5 * (dmax - dmin)
    fa = hand.finger_angle(phi)
    if acc is not None:
        contact = hand.contact
        back = max(cfg.realign_backoff, cfg.backoff, cfg.descend_gap, cfg.approach_gap)
        for (s0, s1, th), dp in zip(parts, deps):
            s = np.linspace(s0, s1, max(2, int(math.ceil((s1 - s0) / cfg.ds)) + 1))
            cx, cy = a[0] + d[0] * s, a[1] + d[1] * s
            acc.add_box(cx, cy, th, k)
            # the hand behind the box: EE = centre - d (contact + depth), backed off up to ``back`` (re-alignments,
            # the corner back-off, the descent behind the face): one rect stretched along the push covers them
            p_lo, p_hi = float(dp.min()), float(dp.max())
            off = contact + 0.5 * (p_lo + p_hi) + 0.5 * back
            ext = 0.5 * (p_hi - p_lo) + 0.5 * back
            pa, pb = (0.0, ext) if hand.mode == "wide" else (ext, 0.0)
            acc.add_hand(cx - d[0] * off, cy - d[1] * off, fa, pa, pb, ctx.ee_h, k)
            for depth in (contact + p_lo, contact + p_hi + back):
                acc.add_reach(cx - d[0] * depth, cy - d[1] * depth, k)
        if prev is not None:   # the corner at a: back off, then a lifted swing or a low re-alignment
            d_p, mid_p, spread_p, fa_p = prev
            turn = math.degrees(math.acos(float(np.clip(d_p @ d, -1.0, 1.0))))
            e_back = a - d_p * (contact + mid_p + cfg.backoff)
            pad = max(spread_p, spread)
            if turn > cfg.lift_turn_deg - LIFT_BAND_DEG:
                far = a - d * (contact + mid + cfg.descend_gap)
                _move_samples(acc, ctx, e_back, far, fa_p, fa, pad, ctx.lift_h, k)
                # push_via may take the other of the two yaws here (the wrist_3 limit): it then turns the hand the
                # long way round in place, lifted, before it moves across
                _move_samples(acc, ctx, e_back, e_back, fa_p, fa, pad, ctx.lift_h, k, long=True)
                _move_samples(acc, ctx, e_back, far, fa, fa, pad, ctx.lift_h, k)
            if turn <= cfg.lift_turn_deg + LIFT_BAND_DEG:
                _move_samples(acc, ctx, e_back, a - d * (contact + mid + cfg.approach_gap), fa_p, fa, pad, ctx.ee_h, k)
    lo, hi = (min(al), max(al)) if length >= ALIGN_LEN else (min(lo, *al), max(hi, *al))
    return (d, mid, spread, fa), lo, hi


def _check_route(ctx, pts, yaw0, clear, cache=None, collect=False, first_clear=None):
    """(ok, fails, min_clear) of the route through ``pts`` ([start, via ..., goal], world xy), leg by leg, the
    first failure ends it unless ``collect``. ``cache``: leg -> result, shared within one search (the candidates
    share their legs); its "evals" entry counts the legs actually evaluated. ``first_clear``: the floors of the
    first leg (a re-plan from a box that drifted close to an element, plan_route's ``start_floor``)."""
    lo = hi = float(yaw0)
    prev = None
    fails, minc = [], {}
    for k in range(len(pts) - 1):
        a, b = np.asarray(pts[k], dtype=np.float64), np.asarray(pts[k + 1], dtype=np.float64)
        if float(np.linalg.norm(b - a)) < 1e-9:
            continue
        key = None
        if cache is not None and not collect:
            key = (k, *np.round(np.concatenate([a, b]), 6), round(lo, 6), round(hi, 6),
                   None if prev is None else (*np.round(prev[0], 6), round(prev[1], 6), round(prev[2], 6)))
        res = cache.get(key) if key is not None else None
        if res is None:
            acc = _Acc()
            nxt = _leg(ctx, a, b, prev, lo, hi, k, acc)
            leg_clear = first_clear if (k == 0 and first_clear is not None) else clear
            res = (*_check(ctx, acc.arrays(), leg_clear, collect), nxt)
            if key is not None:
                cache[key] = res
                cache["evals"] = cache.get("evals", 0) + 1
        ok_k, f_k, m_k, nxt = res
        for name, v in m_k.items():
            minc[name] = min(minc.get(name, np.inf), v)
        fails += f_k
        if not ok_k and not collect:
            return False, fails, minc
        prev, lo, hi = nxt
    return not fails, fails, minc


def _check(ctx, samples, clear, collect=False):
    """(ok, fails, min_clear) of box / hand / reach samples. ``fails``: [{"check", "elem", "leg"}], the first only
    unless ``collect``; ``min_clear``: per protected element and "table", the lowest clearance evaluated (m).
    ``clear``: the floors "mat", "cyl_box", "cyl_hand", "table"; optional "elem_box" / "elem_hand" {element id:
    floor} replace one element's box / hand floor (the drift relaxations of check_motion and plan_route)."""
    box, hnd, rch = samples
    fails, minc = [], {}
    over_box, over_hand = clear.get("elem_box") or {}, clear.get("elem_hand") or {}

    def fail(check, elem, leg):
        fails.append({"check": check, "elem": elem, "leg": int(leg)})

    X, Y, TH, LEG = (box[:, i] for i in range(4))
    for e in ctx.elems:
        ex, ey = float(e.xy[0]), float(e.xy[1])
        if e.kind == "zone":
            dist = _rect_rect_dist((X, Y, TH, ctx.bx, ctx.by), (ex, ey, 0.0, float(e.half[0]), float(e.half[1])))
            i = int(np.argmin(dist)) if dist.size else 0
            minc[e.id] = float(dist[i]) if dist.size else np.inf
            if dist.size and dist[i] < over_box.get(e.id, clear["mat"]):
                fail("box_mat", e.id, LEG[i])
        else:
            r = float(e.half[0])
            db = _rect_pt_dist(ex, ey, X, Y, TH, ctx.bx, ctx.by) - r
            i = int(np.argmin(db)) if db.size else 0
            best_b = float(db[i]) if db.size else np.inf
            best_h, leg_h = np.inf, 0
            for lay in ctx.hand.layers:
                if not hnd.size:
                    break
                m = hnd[:, 5] + lay.z_lo < float(e.height) + ctx.cfg.z_clear   # the layer reaches below the top
                if not m.any():
                    continue
                hx, hy, ha, pa, pb, _, hl = (hnd[m, j] for j in range(7))
                if lay.shape == "rect":
                    dh = _rect_pt_dist(ex, ey, hx, hy, ha, lay.a + pa, lay.b + pb) - r
                else:
                    dh = np.hypot(ex - hx, ey - hy) - (lay.a + np.maximum(pa, pb)) - r
                j = int(np.argmin(dh))
                if dh[j] < best_h:
                    best_h, leg_h = float(dh[j]), hl[j]
            minc[e.id] = min(best_b, best_h)
            if best_b < over_box.get(e.id, clear["cyl_box"]):
                fail("box_cyl", e.id, LEG[i])
            if best_h < over_hand.get(e.id, clear["cyl_hand"]):
                fail("hand_cyl", e.id, leg_h)
        if fails and not collect:
            return False, fails, minc
    if box.size:
        kx, ky = _rect_corners(X, Y, TH, ctx.bx, ctx.by)
        marg = np.minimum(TABLE_HALF_XY[0] - np.abs(kx), TABLE_HALF_XY[1] - np.abs(ky)).min(axis=1)
        i = int(np.argmin(marg))
        minc["table"] = float(marg[i])
        if marg[i] < clear["table"]:
            fail("table", None, LEG[i])
            if not collect:
                return False, fails, minc
    if ctx.reach_ok is not None and rch.size:
        ok = np.asarray(ctx.reach_ok(rch[:, :2]), dtype=bool)
        if not ok.all():
            fail("reach", None, rch[int(np.argmin(ok)), 2])
    return not fails, fails, minc


def _pose_samples(ctx, xy, yaw):
    """The box alone at ``xy`` (yaw +- the allowance): the start / goal pre-check."""
    alpha = math.radians(ctx.cfg.yaw_allow_deg)
    acc = _Acc()
    acc.add_box(np.array([float(xy[0])]), np.array([float(xy[1])]), np.linspace(yaw - alpha, yaw + alpha, 5), 0)
    return acc.arrays()


def _floors(f):
    """push_via's guard floors from one distance ``f``: mats and cylinders f, the table edge f / 2."""
    return {"mat": float(f), "cyl_box": float(f), "cyl_hand": float(f), "table": 0.5 * float(f)}


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def _partitions(seq):
    """Every split of ``seq`` into consecutive groups."""
    n = len(seq)
    for cuts in itertools.product((False, True), repeat=max(0, n - 1)):
        groups, cur = [], [seq[0]]
        for item, cut in zip(seq[1:], cuts):
            if cut:
                groups.append(cur)
                cur = [item]
            else:
                cur.append(item)
        groups.append(cur)
        yield groups


def _shapes(m):
    """Via layouts for m lanes: at each boundary between lanes keep both ends (0), skip the first lane's exit (1)
    or the next lane's entry (2) — the crossing leg then passes that element itself."""
    return list(itertools.product((0, 1, 2), repeat=max(0, m - 1)))


def _lane_kept(shape, j, m):
    return (j == 0 or shape[j - 1] != 2) and (j == m - 1 or shape[j] != 1)


def _side_name(s, n0):
    y = s * float(n0[1])
    if abs(y) > 0.2:
        return "near" if y < 0.0 else "far"
    return "left" if s > 0 else "right"


def plan_route(start_xy, goal_xy, box_half_xy, box_yaw=0.0, elems=(), protect=None, cfg=None, reach_ok=None,
               box_h=0.05, hand=None, ee_height=None, prefer=None, start_floor=None) -> Route:
    """The safe push route from the box at ``start_xy`` (yaw ``box_yaw``, footprint half sides ``box_half_xy``,
    height ``box_h``) to ``goal_xy``: via points (the goal last) keeping every protected element of ``elems``
    clear (``protect``: element ids, None = all). ``reach_ok``: make_reach(...) or None. ``ee_height``: the push EE
    height above the table (default the hand's tip_len + tip_clear). ``prefer``: {element id: +1 / -1} lane sides
    of an earlier route (re-plans keep them when they can). ``start_floor``: a re-plan from a box that drifted
    close to an element (push_zone's, at the guard's floor guard_clear): the start pose needs only that floor (the
    table half of it), and on the first leg an element the start pose already has inside its clearance (the table
    edge too) keeps only that floor, box and hand, so the route can lead away from it."""
    cfg = cfg if isinstance(cfg, RouteCfg) else RouteCfg.from_dict(cfg)
    hand = hand if hand is not None else wsg_hand(cfg.hand)
    ctx = _Ctx(cfg, hand, box_half_xy, box_h, reach_ok, elems, protect, ee_height)
    s = np.asarray(start_xy, dtype=np.float64).reshape(2)
    g = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    yaw0 = float(box_yaw)
    base = {"protect": "all" if protect is None else sorted(str(i) for i in protect)}
    total = float(np.linalg.norm(g - s))
    if total < 1e-6:
        return Route(True, via=[g.tolist()], hand=hand.mode, detail=base)
    d0 = (g - s) / total
    n0 = np.array([-d0[1], d0[0]])

    first_clear = None
    for tag, xy in (("start", s), ("goal", g)):
        relax = tag == "start" and start_floor is not None
        ok, fails, m_pose = _check(ctx, _pose_samples(ctx, xy, yaw0), _floors(start_floor) if relax else ctx.plan_clear,
                                   collect=relax)
        if not ok:
            return Route(False, hand=hand.mode, reason=f"{tag}_blocked", detail={**base, "fail": fails[0]})
        if relax:
            f = _floors(start_floor)
            near = [e.id for e in ctx.elems
                    if m_pose.get(e.id, np.inf) < (cfg.mat_clear if e.kind == "zone" else cfg.cyl_clear_box)]
            table_near = m_pose.get("table", np.inf) < cfg.table_clear
            first_clear = {**ctx.plan_clear, "elem_box": {i: f["mat"] for i in near},
                           "elem_hand": {i: f["cyl_hand"] for i in near}}
            if table_near:
                first_clear["table"] = f["table"]
            base["start_relaxed"] = sorted(near) + (["table"] if table_near else [])

    ok, fails, minc = _check_route(ctx, [s, g], yaw0, ctx.plan_clear, collect=True, first_clear=first_clear)
    if ok:
        return Route(True, via=[g.tolist()], hand=hand.mode, length=total, min_clear=minc, detail=base)
    blocking_ids = {f["elem"] for f in fails if f["elem"] is not None}
    u_of = {e.id: float((np.asarray(e.xy, dtype=np.float64) - s) @ d0) for e in ctx.elems}
    v_of = {e.id: float((np.asarray(e.xy, dtype=np.float64) - s) @ n0) for e in ctx.elems}
    blocking = sorted((e for e in ctx.elems if e.id in blocking_ids), key=lambda e: u_of[e.id])
    detail = {**base, "blocking": [e.id for e in blocking], "straight": fails}
    if not blocking:
        return Route(False, hand=hand.mode, reason="straight_" + fails[0]["check"], detail=detail)

    # lateral clearance from an element's centre to a lane (the box centre line) that passes it
    alpha = math.radians(cfg.yaw_allow_deg)
    bs = max(ctx.bx, ctx.by) * math.cos(alpha) + min(ctx.bx, ctx.by) * math.sin(alpha)
    phi0 = math.atan2(d0[1], d0[0])

    def lane_clear(e):
        if e.kind == "zone":
            return abs(n0[0]) * float(e.half[0]) + abs(n0[1]) * float(e.half[1]) + bs + cfg.mat_clear
        reach_down = [lay for lay in hand.layers if ctx.ee_h + lay.z_lo < float(e.height) + cfg.z_clear]
        hv = max([hand.across(lay) for lay in reach_down] or [0.0])
        return float(e.half[0]) + max(bs + cfg.cyl_clear_box, hv + cfg.cyl_clear_hand)

    need = {e.id: lane_clear(e) + cfg.lane_slack for e in ctx.elems}

    def lane_options(group, side):
        """The tightest lane past every member on ``side``, then pushed past the protected non-members it would
        clip (one at a time)."""
        lam = side * max(side * v_of[e.id] + need[e.id] for e in group)
        out = [lam]
        ids = {e.id for e in group}
        u_lo = min(u_of[e.id] for e in group) - 0.15
        u_hi = max(u_of[e.id] for e in group) + 0.15
        for _ in range(2):
            clip = [f for f in ctx.elems if f.id not in ids and u_lo <= u_of[f.id] <= u_hi
                    and abs(lam - v_of[f.id]) < need[f.id]]
            if not clip:
                break
            lam = side * max(side * v_of[f.id] + need[f.id] for f in clip)
            ids |= {f.id for f in clip}
            out.append(lam)
        return out

    plans = []
    for groups in _partitions(blocking):
        for sides in itertools.product((-1, 1), repeat=len(groups)):
            for lanes in itertools.product(*[lane_options(grp, sd) for grp, sd in zip(groups, sides)]):
                plans.append((groups, sides, lanes))

    first_max = math.radians(cfg.max_first_deg)
    turn_max = math.radians(cfg.max_turn_deg)
    keys = []   # one per (plan, via shape): (cheapest cost, plan, shape, via u [C, P] and costs by cost, via v [P])
    for pi, (groups, sides, lanes) in enumerate(plans):
        m = len(groups)
        extra = 1e-4 * sum(1 for sd in sides if sd * n0[1] > 0.0)   # ties go to the near side (-y)
        if prefer:
            extra += cfg.w_side * sum(1 for grp, sd in zip(groups, sides) for e in grp
                                      if e.id in prefer and int(prefer[e.id]) != sd)
        for shape in _shapes(m):
            grids, vs = [], []
            for j, (grp, lam) in enumerate(zip(groups, lanes)):
                # the lane is entered before its first element and left after its last; the first lane may start
                # abreast of the box (a sideways first leg), the last one end abreast of the goal (a sideways last)
                u_first = min(u_of[e.id] for e in grp)
                u_last = max(u_of[e.id] for e in grp)
                lo_in = 0.0 if j == 0 else max(u_of[e.id] for e in groups[j - 1]) + cfg.min_leg
                hi_out = total if j == m - 1 else min(u_of[e.id] for e in groups[j + 1]) - cfg.min_leg
                g_in = np.linspace(lo_in, u_first, cfg.grid) if u_first > lo_in + 1e-6 else np.array([u_first])
                g_out = np.linspace(u_last, hi_out, cfg.grid) if hi_out > u_last + 1e-6 else np.array([u_last])
                if j == 0 or shape[j - 1] != 2:
                    grids.append(g_in)
                    vs.append(lam)
                if j == m - 1 or shape[j] != 1:
                    grids.append(g_out)
                    vs.append(lam)
            U = np.stack([gg.ravel() for gg in np.meshgrid(*grids, indexing="ij")], axis=1)
            n_c = U.shape[0]
            pu = np.concatenate([np.zeros((n_c, 1)), U, np.full((n_c, 1), total)], axis=1)
            pv = np.concatenate([np.zeros((n_c, 1)), np.broadcast_to(np.asarray(vs), U.shape), np.zeros((n_c, 1))],
                                axis=1)
            du, dv = np.diff(pu, axis=1), np.diff(pv, axis=1)
            seg = np.hypot(du, dv)
            head = np.arctan2(dv, du)
            turns = np.abs(np.diff(head, axis=1))
            keep = (du >= -1e-9).all(axis=1) & (seg >= cfg.min_leg - 1e-9).all(axis=1)
            keep &= (np.abs(head[:, 0]) <= first_max + 1e-9) & (turns <= turn_max + 1e-9).all(axis=1)
            cost = seg.sum(axis=1) + cfg.w_turn * (head[:, 0] ** 2 + (turns ** 2).sum(axis=1)) \
                + cfg.w_corner * U.shape[1] + extra
            idx = np.flatnonzero(keep)
            if idx.size:
                idx = idx[np.argsort(cost[idx], kind="stable")]
                keys.append((float(cost[idx[0]]), pi, shape, U[idx], cost[idx], np.asarray(vs, dtype=np.float64)))
    keys.sort(key=lambda t: t[0])
    detail.update(n_candidates=int(sum(len(k[4]) for k in keys)))
    if not keys:
        return Route(False, hand=hand.mode, reason="no_candidate", detail=detail)

    def lane_check(group, lam):
        """None, or the first failure of the box on the lane abreast of the group (a necessary condition)."""
        u_a = min(u_of[e.id] for e in group)
        u_b = max(max(u_of[e.id] for e in group), u_a + 1e-3)
        a, b = s + u_a * d0 + lam * n0, s + u_b * d0 + lam * n0
        ok_l, fails_l, _ = _check_route(ctx, [a, b], _aligned(yaw0, phi0)[0], ctx.plan_clear)
        return None if ok_l else fails_l[0]

    # branch and bound over the plans, cheapest first: each plan's candidates in cost order until one passes or the
    # plan has used its share of the evaluations; a plan whose cheapest candidate cannot beat the best route found
    # ends the search (so a costlier side still gets tried when the cheap one is blocked)
    lanes_seen, why, cache = {}, Counter(), {}
    n_checked, best = 0, None
    per_plan = max(1, int(cfg.max_evals) // 4)
    for c_min, pi, shape, Us, costs, V in keys:
        if (best is not None and c_min >= best[0]) or cache.get("evals", 0) >= cfg.max_evals:
            break
        groups, sides, lanes = plans[pi]
        bad = False
        for j, (grp, sd, lam) in enumerate(zip(groups, sides, lanes)):
            if not _lane_kept(shape, j, len(groups)):
                continue
            key = f"{'+'.join(e.id for e in grp)}:{_side_name(sd, n0)}:{lam:+.3f}"
            if key not in lanes_seen:
                lanes_seen[key] = lane_check(grp, lam)
            if lanes_seen[key] is not None:
                bad = True
                break
        if bad:
            continue
        evals0 = cache.get("evals", 0)
        for U, cost in zip(Us, costs):
            if (best is not None and cost >= best[0]) or cache.get("evals", 0) - evals0 >= per_plan \
                    or cache.get("evals", 0) >= cfg.max_evals:
                break
            via = [s + u * d0 + v * n0 for u, v in zip(U, V)] + [g]
            ok, fails, _ = _check_route(ctx, [s] + via, yaw0, ctx.plan_clear, cache=cache, first_clear=first_clear)
            n_checked += 1
            if ok:
                best = (float(cost), pi, via)
                break
            why[f"{fails[0]['check']}:{fails[0]['elem']}"] += 1
    detail.update(n_checked=n_checked, n_evals=cache.get("evals", 0), lanes=_lane_report(lanes_seen))
    if best is None:
        detail["fails"] = dict(why.most_common(6))
        if n_checked == 0:
            reason = "lanes_blocked"
        elif cache.get("evals", 0) >= cfg.max_evals:
            reason = "max_evals"
        else:
            reason = "checks_failed"
        return Route(False, hand=hand.mode, reason=reason, detail=detail)
    _, pi, via = best
    groups, sides, _ = plans[pi]
    _, _, minc = _check_route(ctx, [s] + via, yaw0, ctx.plan_clear, collect=True, first_clear=first_clear)
    legs = np.diff(np.stack([s] + via), axis=0)
    heads = np.arctan2(legs[:, 1], legs[:, 0])
    detail["lane_sides"] = {e.id: int(sd) for grp, sd in zip(groups, sides) for e in grp}
    return Route(
        True,
        via=[p.tolist() for p in via],
        sides={e.id: _side_name(sd, n0) for grp, sd in zip(groups, sides) for e in grp},
        hand=hand.mode,
        turns_deg=[math.degrees(abs(float(_wrap(t)))) for t in np.diff(heads)],
        first_deg=math.degrees(abs(float(_wrap(heads[0] - phi0)))),
        length=float(np.linalg.norm(legs, axis=1).sum()),
        min_clear=minc,
        detail=detail,
    )


def _lane_report(lanes_seen):
    return {k: ("ok" if v is None else f"{v['check']}:{v['elem']}") for k, v in lanes_seen.items()}


# ---------------------------------------------------------------------------
# executor guard
# ---------------------------------------------------------------------------


def check_motion(ee_poses, box_from, box_to, box_half_xy, box_yaw, elems, protect=None, cfg=None, hand=None,
                 box_h=0.05, table_z=0.74, guard_yaw_deg=5.0):
    """push_via's guard, before every step: (ok, first failure or None, min clearances). ``ee_poses``: the EE poses
    the step commands, in order, the live pose first (world [x, y, z, qw, qx, qy, qz]); the box slides
    ``box_from`` -> ``box_to`` (equal: it stays). Every protected element keeps ``guard_clear`` from the box and
    from every hand layer at its height, the box stays guard_clear / 2 inside the table. Reach is not checked here
    (an unreachable pose fails its plan; it harms nothing).

    Between two poses the hand turns the short way round of the full circle (the wrist follows the orientation's
    shorter turn), so the other of push_via's two yaws is checked swinging the long way; a turn in place is sampled
    as finely as a move. A box or hand that already sits inside a floor (it drifted there) keeps the clearance it has
    instead: the step passes if it never brings them closer (a move away recovers), unless they already touch."""
    cfg = cfg if isinstance(cfg, RouteCfg) else RouteCfg.from_dict(cfg)
    hand = hand if hand is not None else wsg_hand(cfg.hand)
    ctx = _Ctx(cfg, hand, box_half_xy, box_h, None, elems, protect, None)
    acc = _Acc()
    p0 = np.asarray(box_from, dtype=np.float64).reshape(2)
    p1 = np.asarray(box_to, dtype=np.float64).reshape(2)
    dist = float(np.linalg.norm(p1 - p0))
    ga = math.radians(guard_yaw_deg)
    yaws = [float(box_yaw)]
    if dist > 1e-6:
        yaws += _aligned(float(box_yaw), math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    span = max(yaws) - min(yaws) + 2.0 * ga
    th = np.linspace(min(yaws) - ga, max(yaws) + ga, max(3, int(math.ceil(span / YAW_STEP)) + 1))
    s = np.linspace(0.0, 1.0, max(2, int(math.ceil(dist / cfg.ds)) + 1))
    acc.add_box(p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1]), th, 0)
    poses = [np.asarray(p, dtype=np.float64).reshape(7) for p in ee_poses]
    for k in range(len(poses) - 1):
        a, b = poses[k], poses[k + 1]
        fa, fb = quat_finger_angle(a[3:]), quat_finger_angle(b[3:])
        dang = float(_wrap(fb - fa))
        t = np.linspace(0.0, 1.0, _n_samples(ctx, np.linalg.norm(b[:2] - a[:2]), dang))
        acc.add_hand(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), fa + t * dang, 0.0, 0.0,
                     a[2] + t * (b[2] - a[2]) - float(table_z), k + 1)
    # the current pose: the box where it stands (its yaw +- the guard's), the hand at the live EE pose
    now_box, now_hand = _Acc(), _Acc()
    th_now = th[np.abs(th - float(box_yaw)) <= ga + 1e-9]
    now_box.add_box(p0[:1], p0[1:2], th_now if th_now.size else np.array([float(box_yaw)]), 0)
    if poses:
        a = poses[0]
        now_hand.add_hand(a[:1], a[1:2], quat_finger_angle(a[3:]), 0.0, 0.0, a[2] - float(table_z), 0)
        if len(poses) == 1:
            acc.add_hand(a[:1], a[1:2], quat_finger_angle(a[3:]), 0.0, 0.0, a[2] - float(table_z), 0)
    _, _, c_box = _check(ctx, now_box.arrays(), ctx.guard_clear, collect=True)
    _, _, c_hand = _check(ctx, now_hand.arrays(), ctx.guard_clear, collect=True)
    clear = dict(ctx.guard_clear)
    clear["elem_box"], clear["elem_hand"] = {}, {}
    for e in ctx.elems:
        floor_box = clear["mat"] if e.kind == "zone" else clear["cyl_box"]
        if 0.0 < c_box.get(e.id, np.inf) < floor_box:
            clear["elem_box"][e.id] = c_box[e.id] - 1e-6
        if e.kind != "zone" and 0.0 < c_hand.get(e.id, np.inf) < clear["cyl_hand"]:
            clear["elem_hand"][e.id] = c_hand[e.id] - 1e-6
    if 0.0 < c_box.get("table", np.inf) < clear["table"]:
        clear["table"] = c_box["table"] - 1e-6
    ok, fails, minc = _check(ctx, acc.arrays(), clear)
    return ok, (None if ok else fails[0]), minc
