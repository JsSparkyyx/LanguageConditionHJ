"""RoboTwin-side helpers for item-safety tasks (plan v3 §2.2/§4.1/§4.5).

Lives INSIDE the symlinked ``RoboTwin/envs/`` tree, so it may import
RoboTwin modules — but it must NEVER import CEHJ ``main.*``: task files
run inside RoboTwin's process (which has its own vendored ``envs/``
package and cannot reach the CEHJ repo), and the CEHJ runner re-reads
all metadata back from plain attribute dicts. The metadata dicts this
module writes onto the task instance are the ONLY interface between a
RoboTwin task and ``main.item_envs``:

- ``task.item_meta``   — list of item dicts, exactly the v3 §4.1
  ``items`` schema (``item_id``, ``entity``, ``category``, ``color``,
  ``kind``, ``role``, ``group``, ``particles``, ``half_extents``,
  ``obb_center_local``, ``up_axis_local``, ``mass``). ``entity`` is the
  live sapien Actor/Articulation; the runner derives ``pose_fn`` /
  ``obb`` / ``vel_fn`` from it.
- ``task.region_meta`` — list of region dicts (``region_id``,
  ``entity``, ``kind``, ``color``, plus zone extras ``size_xy`` /
  ``z`` / ``z_band`` / ``nameable``). The runner builds Region objects
  from these (``main.item_envs.regions.zone_region`` reads the
  ``size_xy`` attribute when present and falls back to the actor's
  collision-shape AABB otherwise).

Time-scale convention (T1): the runner sets ``task._time_scale_safe``
and ``task._time_scale_risky`` once after setup. ``risky(task)`` below
flips ``task.active_time_scale`` to the risky value around a motion
primitive and restores the previous value on exit (nesting-safe).
``RiskTickController.plan()`` only READS ``task.active_time_scale``
for its retime factor — the controller's separate
``risky_skill_labels`` path is the ONLY other writer and must not be
combined with this context manager.

Optional live mass override (``register_item(..., mass=...)``) writes
through PhysX immediately, so ``physics.apply_physics`` /
``physics.read_back`` stay the single source of truth for sampled
physics.
"""

import math
from contextlib import contextmanager


def _unwrap(entity):
    """Accept a raw sapien Entity or a RoboTwin Actor wrapper."""
    return getattr(entity, "actor", entity)


GOAL_FRAME_COLOR = (0.10, 0.10, 0.12)  # near-black: high contrast on the white table, unlike any filled zone


def goal_frame(scene, center_xy, z, size, width, color=GOAL_FRAME_COLOR, name="goal_frame"):
    """Hollow square outline (4 render-only strips) marking a goal — visual
    only, never a rule referent; its outline shape tells it apart from the
    filled zone mats. Returns the 4 entities."""
    import sapien

    s, w = float(size), float(width)
    strips = [(0.0, (s - w) / 2.0, s, w), (0.0, -(s - w) / 2.0, s, w),
              ((s - w) / 2.0, 0.0, w, s - 2.0 * w), (-(s - w) / 2.0, 0.0, w, s - 2.0 * w)]
    return [
        zone_mat(scene, sapien.Pose([float(center_xy[0]) + dx, float(center_xy[1]) + dy, float(z) + 0.0006]),
                 size_xy=(sx, sy), color=color, name=f"{name}_{k}", collision=False)
        for k, (dx, dy, sx, sy) in enumerate(strips)
    ]


def zone_mat(scene, pose, size_xy, color, thickness=0.002, name=None, collision=True):
    """Spawn a thin static colored box that marks a zone on the table.

    ``pose`` is the CENTER pose of the mat (sapien convention);
    ``size_xy`` is the full (x, y) extent. Built with the same
    Entity + PhysxRigidStaticComponent idiom as RoboTwin's own
    ``create_entity_box`` (no ActorBuilder). A render body is attached
    ONLY when the scene actually has a render system, queried via
    ``scene.get_system("render")`` (sapien raises RuntimeError when no
    such system is registered — the same name-based lookup
    ``add_entity`` performs, so a false query means ``add_entity``
    would reject the entity). Physics-only scenes get a
    collision-only mat.

    The returned entity is a raw pybind11 ``sapien.Entity`` and carries
    NO Python attributes (pybind11 rejects them). ``size_xy`` reaches
    the CEHJ side through ``register_zone``'s meta dict instead;
    ``regions.zone_region`` derives the same footprint from the mat's
    collision-shape AABB when no attribute is present (so a collision-free
    mat MUST be registered with its size_xy).
    """
    import sapien

    half = [float(size_xy[0]) / 2.0, float(size_xy[1]) / 2.0, float(thickness) / 2.0]
    entity = sapien.Entity()
    entity.set_name(name or f"zone_{int(color[0]*255)}_{int(color[1]*255)}_{int(color[2]*255)}")
    material = getattr(scene, "default_physical_material", None)
    if material is None:
        material = scene.create_physical_material(0.5, 0.5, 0.0)
    if collision:
        rigid = sapien.physx.PhysxRigidStaticComponent()
        rigid.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half, material=material))
        entity.add_component(rigid)
    # collision=False: a PAINTED region (render body only) — push_zone's
    # on-path zones are slid over, and a 2 mm collision step catches the box
    try:  # query, not construction: get_system raises when absent
        scene.get_system("render")
        has_render = True
    except RuntimeError:
        has_render = False
    if has_render:
        visual = sapien.render.RenderMaterial(base_color=[float(color[0]), float(color[1]), float(color[2]), 1.0])
        render = sapien.render.RenderBodyComponent()
        render.attach(sapien.render.RenderShapeBox(half, visual))
        entity.add_component(render)
    entity.set_pose(pose)
    scene.add_entity(entity)
    return entity


@contextmanager
def risky(task, risk=None):
    """Mark the enclosed motion primitive as risk-bearing (plan v3 §2.5 T1).

    Sets ``task.active_time_scale`` to the risky time scale for the
    duration of the block and restores the previous value on exit
    (nesting-safe). The scale comes from ``risk`` when given (a
    RiskParams-like object with ``.time_scale``, or a plain float),
    else from ``task._time_scale_risky`` which the runner installs
    after controller attach; a missing attribute falls back to 1.0 so
    the task also runs under vanilla RoboTwin collection.
    """
    if risk is not None:
        scale = float(getattr(risk, "time_scale", risk))
    else:
        scale = float(getattr(task, "_time_scale_risky", 1.0))
    prev = getattr(task, "active_time_scale", getattr(task, "_time_scale_safe", 1.0))
    task.active_time_scale = scale
    try:
        yield
    finally:
        task.active_time_scale = prev


def register_item(task, item_id, entity, category, color, kind, role, group=None,
                  particles=None, half_extents=None, mass=None,
                  obb_center_local=None, up_axis_local=None, size=None):
    """Append a v3 §4.1 item-metadata dict to ``task.item_meta``.

    ``size`` (optional): a size word the sentence generator may use ("the
    tall block"), e.g. T5 / T7's "tall" / "short" blocks.

    ``mass`` (optional) is applied to the live PhysX body immediately —
    the same write ``physics.apply_physics`` performs — so mass never
    diverges from what the runner later reads back.
    """
    if not hasattr(task, "item_meta"):
        task.item_meta = []
    meta = {
        "item_id": str(item_id),
        "entity": entity,
        "category": str(category),
        "color": str(color),
        "kind": str(kind),
        "role": str(role),
        "group": group,
        "particles": particles,
        "half_extents": None if half_extents is None else [float(v) for v in half_extents],
        "obb_center_local": None if obb_center_local is None else [float(v) for v in obb_center_local],
        "up_axis_local": None if up_axis_local is None else [float(v) for v in up_axis_local],
        "mass": None if mass is None else float(mass),
        "size": None if size is None else str(size),
    }
    task.item_meta.append(meta)
    if mass is not None:
        _apply_mass(entity, float(mass))
    return meta


def register_zone(task, region_id, entity, color, size_xy=None, kind="zone",
                  z=0.0, z_band=0.01, nameable=True, **extra):
    """Append a region-metadata dict to ``task.region_meta``.

    The runner turns each entry into a Region. Raw sapien entities
    cannot carry Python attributes, so this meta dict is the channel
    for ``size_xy``; ``regions.zone_region`` otherwise derives the mat
    footprint from its collision-shape AABB.
    """
    if not hasattr(task, "region_meta"):
        task.region_meta = []
    meta = {
        "region_id": str(region_id),
        "entity": entity,
        "kind": str(kind),
        "color": str(color),
        "size_xy": None if size_xy is None else [float(v) for v in size_xy],
        "z": float(z),
        "z_band": float(z_band),
        "nameable": bool(nameable),
    }
    meta.update(extra)
    task.region_meta.append(meta)
    return meta


def _apply_mass(entity, mass):
    """Set the mass of every dynamic rigid body under ``entity``."""
    import sapien.physx as physx

    raw = _unwrap(entity)
    comp = raw.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if comp is None:
        return
    comp.set_mass(float(mass))


# ---------------------------------------------------------------------------
# push primitive (plan v3 §7 T3)
# ---------------------------------------------------------------------------

# RoboTwin's grasp-pose convention (envs/_base_task.py get_grasp_pose): the
# gripper pose origin sits 0.12 m behind the fingertip contact along the
# approach axis (recon.md Q10, approach_to_tcp_m = 0.12, both embodiments).
APPROACH_TO_TCP = 0.12
# GRASP_DIRECTION_DIC["top_down"] (envs/_GLOBAL_CONFIGS.py) — fingers point
# straight down ("fingers vertical"), same world-frame quat for both arms.
TOP_DOWN_QUAT = [-0.5, 0.5, -0.5, -0.5]
TABLE_TOP_Z = 0.74  # RoboTwin table height; + task.table_z_bias at runtime


class RiskyStep:
    """Marker wrapping one (arm_tag, actions) step of a skill action list.

    ``execute_steps`` runs the wrapped step inside ``risky(task)`` — the
    marking must travel WITH the returned action list because the list is
    executed by the caller (a ``with risky(task):`` block inside push()
    itself would end before the caller ever moves). The wrapped step may be
    a plain tuple or a ``_Deferred`` (resolved by ``realize_step`` before
    ``task.move``). Unpacks like the plain (arm_tag, actions) tuples
    existing tasks compose (plain-tuple contents only).
    """

    def __init__(self, step):
        self.step = step

    def __iter__(self):
        return iter(self.step)

    def __getitem__(self, i):
        return self.step[i]


class _Deferred:
    """Skill step whose action list is built at EXECUTION time.

    RoboTwin's ``move_by_displacement`` reads the LIVE EE pose when it is
    CALLED and bakes an absolute target pose into the returned Action — so
    calling it inside push() (before any motion has executed) would anchor
    the displacement at the home pose instead of the pose the arm holds
    just before the push. ``_Deferred`` stores a builder and calls it in
    ``realize`` — inside ``execute_steps`` right before ``task.move`` — so
    the anchor is the live EE pose at that moment.
    """

    def __init__(self, arm_tag, builder):
        self.arm_tag = arm_tag
        self.builder = builder

    def realize(self, task):
        return self.builder(task)


def realize_step(task, entry):
    """Resolve one step entry to a plain (arm_tag, actions) / RiskyStep pair.

    ``_Deferred`` entries build their actions NOW (live EE anchor); plain
    tuples pass through. The two-arm push_zone nominal zips two push()
    lists and must map both sides of each pair through this before
    ``task.move(step_l, step_r)``, entering ``risky(task)`` when either
    realized side is a RiskyStep.
    """
    if isinstance(entry, RiskyStep):
        step = entry.step
        return RiskyStep(step.realize(task) if isinstance(step, _Deferred) else step)
    if isinstance(entry, _Deferred):
        return entry.realize(task)
    return entry


def execute_steps(task, steps):
    """Play a skill action list built by push(); RiskyStep entries run risky.

    Single-arm execution (one task.move per entry, realized first). The
    two-arm push_zone nominal zips two push() lists instead and calls
    task.move(step_l, step_r) per realized pair, entering ``risky(task)``
    when either side is a RiskyStep.
    """
    for entry in steps:
        entry = realize_step(task, entry)
        if isinstance(entry, RiskyStep):
            with risky(task):
                task.move(entry.step)
        else:
            task.move(entry)


def _obb_half_world(obj):
    """World-frame OBB half extents [3] for a RoboTwin Actor (config-based).

    Mirrors CEHJ's world_center_and_half without importing main.*: mesh JSONs
    carry extents in model units with a scale; create_box copies world
    half_size into BOTH fields (the near-equal special case). Raw entities
    (no config) fall back to None and the caller uses a default.
    """
    import numpy as np

    cfg = getattr(obj, "config", None) or {}
    try:
        extents = np.abs(np.asarray(cfg.get("extents", []), dtype=np.float64).reshape(-1))[:3]
        if extents.size < 3:
            return None
        raw = cfg.get("scale")
        if raw is None:
            scale = np.ones(3)
        else:
            raw = np.asarray(raw, dtype=np.float64).reshape(-1)
            scale = np.full(3, abs(float(raw[0]))) if raw.size == 1 else np.abs(np.resize(raw, 3))
        # create_box: extents == scale == world half_size
        if np.allclose(extents, scale, rtol=0.05, atol=1e-4) and float(np.max(extents)) < 0.4:
            return extents
        return 0.5 * extents * scale
    except Exception:
        return None


def push(task, arm_tag, obj, push_dir_xy, dist, overshoot=0.0, lateral_offset=0.0,
         prepush_dist=0.05, height_offset=0.0, seg_len=None):
    """One straight table-top push with a closed gripper (plan v3 §7 T3).

    Returns the skill action list (RoboTwin skills only) in execution order:

    1. close the gripper,
    2. move_to_pose to the pre-push pose: ``prepush_dist`` behind P's contact
       face along -push_dir, at P's CoM height + ``height_offset`` clamped to
       >= 1 cm above the table (the clamp reads on the FINGERTIP plane — the
       gripper pose origin sits APPROACH_TO_TCP = 0.12 m above the fingertips,
       RoboTwin's grasp-pose convention), fingers vertical (top-down),
    3. RiskyStep(move_by_displacement along push_dir by prepush_dist + dist +
       overshoot: the gripper first closes the ``prepush_dist`` gap to P's
       contact face, then drives P by dist + overshoot — so P travels the
       commanded dist, which is what the P3-robot gate measures) — the whole
       contact line is shifted laterally by ``lateral_offset``. With
       ``seg_len`` set (push_zone's cross-table pushes), the displacement is
       CHAINED in ceil(travel / seg_len) RiskyStep segments of travel/n_seg
       each (every segment <= seg_len): each segment is its own _Deferred
       plan anchored at the live EE pose, so the arm replans/reconfigures
       over the long travel instead of committing one 0.5+ m displacement
       (joint-limit flip risk). The segments run back-to-back at the risky
       time scale (consecutive risky() blocks with the same scale).
    4. retreat 5 cm along -push_dir and lift 5 cm.

    Steps 3 and 4 are ``_Deferred``: ``move_by_displacement`` bakes an
    absolute target from the LIVE EE pose at call time, so their builders
    must run when the arm actually sits at the pre-push / post-push pose,
    not when push() is called (home pose). The list MUST therefore be
    executed through ``execute_steps`` (single arm) or zipped pairwise
    through ``realize_step`` + task.move(a_l, a_r) (push_zone's two-arm
    nominal). The planner's world model holds only the table (the caller
    refreshes update_curobo_world), so plans pass straight through P —
    that is intended, it is how a push works.

    ``obj``: a RoboTwin Actor (create_box/create_actor) or raw entity with
    get_pose(); pose.p is treated as P's center/CoM. The contact-face depth
    along push_dir is the OBB support (exact for boxes) from the actor config,
    defaulting to 3 cm when no config is readable.
    """
    import numpy as np

    direction = np.asarray(push_dir_xy, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(direction))
    if n < 1e-9:
        raise ValueError("push_dir_xy must be nonzero")
    direction /= n
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)

    pose = _unwrap(obj).get_pose()
    center = np.asarray(pose.p, dtype=np.float64).reshape(3)
    rot = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)[:3, :3]
    half = _obb_half_world(obj)
    if half is None:
        half = np.array([0.03, 0.03, 0.03], dtype=np.float64)
    # OBB support along push_dir (exact): sum_i |d . axis_i| * half_i
    depth = float(np.abs(rot.T @ np.array([direction[0], direction[1], 0.0])) @ half)

    table_z = TABLE_TOP_Z + float(getattr(task, "table_z_bias", 0.0))
    z_contact = max(float(center[2]) + float(height_offset), table_z + 0.01)
    face_xy = center[:2] - direction * depth  # contact face on the -push_dir side
    pre_xy = face_xy - direction * float(prepush_dist) + perp * float(lateral_offset)
    pre_pose = [float(pre_xy[0]), float(pre_xy[1]), z_contact + APPROACH_TO_TCP] + list(TOP_DOWN_QUAT)

    # gripper travel = closing gap + commanded box travel (+ overshoot)
    travel = float(prepush_dist) + float(dist) + float(overshoot)

    def _disp_step(dist_m):
        def _step(task):
            return task.move_by_displacement(
                arm_tag, x=float(direction[0] * dist_m), y=float(direction[1] * dist_m)
            )

        return _step

    if seg_len is None:
        push_steps = [RiskyStep(_Deferred(arm_tag, _disp_step(travel)))]
    else:
        n_seg = max(1, int(np.ceil(travel / float(seg_len))))
        push_steps = [
            RiskyStep(_Deferred(arm_tag, _disp_step(travel / n_seg))) for _ in range(n_seg)
        ]

    def _retreat_step(task):
        return task.move_by_displacement(
            arm_tag,
            x=float(-direction[0] * 0.05),
            y=float(-direction[1] * 0.05),
            z=0.05,
        )

    steps = [
        task.close_gripper(arm_tag),
        task.move_to_pose(arm_tag, pre_pose),
        *push_steps,
        _Deferred(arm_tag, _retreat_step),
    ]
    return steps


def _push_quat(push_dir_xy):
    """TOP_DOWN_QUAT yawed about world z so the finger axis (EE y — world x
    under TOP_DOWN) lies across ``push_dir_xy``, using the yaw in
    (-pi/2, pi/2] (the closed gripper is symmetric under 180 deg)."""
    import numpy as np

    d = np.asarray(push_dir_xy, dtype=np.float64).reshape(2)
    perp = np.array([-d[1], d[0]])
    # R_z(psi) @ (-1, 0) == perp  ->  psi = atan2(perp) - pi, folded into (-pi/2, pi/2]
    psi = float(np.arctan2(perp[1], perp[0]) - np.pi)
    psi = (psi + np.pi / 2.0) % np.pi - np.pi / 2.0
    qz = np.array([np.cos(psi / 2.0), 0.0, 0.0, np.sin(psi / 2.0)])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = TOP_DOWN_QUAT
    q = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
    return [float(v) for v in q]


def push_to_goal(task, arm_tag, obj, goal_xy, overshoot=0.0, lateral_offset=0.0,
                 prepush_dist=0.05, height_offset=0.0, seg_len=0.10, approach_gap=0.015,
                 max_segments=12, goal_tol=0.005, realign_tol=0.01,
                 height_frac=0.6, lane_tol=0.05, approach_lift=0.10, tip_len=None, tip_clear=0.008):
    """Closed-loop table-top push of ``obj`` to ``goal_xy`` (plan v3 §7 T3).

    ``tip_len`` (m, per embodiment): EE origin -> BOTTOM of the closed
    fingers. When given, the push height is set from it — fingertip bottoms
    ``tip_clear`` above the table — instead of the TCP model (TCP =
    APPROACH_TO_TCP below the EE at height_frac of the CoM height): that model
    put the fingertips 3-4 cm INTO the table, the closed hand rested on the
    table (EE z floor 0.904 ur5 / 0.892 franka whatever was commanded) and
    dragged over low boxes (diag d36-d41: 4-6 cm boxes stalled after 5-35 cm).

    ``push()`` commands one fixed direction/distance from the pose at
    play_once time; over push_zone's 0.2-0.5 m cross-table travel the box
    yaws under the narrow finger contact and walks 3-5 cm off the line (diag
    job 22636182: ur5 seeds 0/2 ended 3-5 cm off the goal; franka seed 2 the
    rotated box was pinched and launched). Here every segment is a
    ``RiskyStep(_Deferred)`` re-aimed at EXECUTION time from the LIVE box:

      d     = unit(goal - box centre)           (re-aimed every segment)
      face  = centre - d * OBB support along d  (+ lateral_offset sideways)
      pre   = face - d * approach_gap
      push  = pre + d * min(seg_len, remaining + approach_gap [+ overshoot on
              the last segment])

    If the gripper is not already at ``pre`` (box drifted / yawed), it backs
    off 2 cm along the old heading first, so the sideways re-alignment never
    drags the box. Segments stop once the box is within ``goal_tol`` with
    the overshoot spent, or the box fell off the table, or the distance to
    the goal grew on two consecutive segments, or the box left its
    start->goal lane by more than ``lane_tol`` (box lost — never chase it
    around; empty action lists, ``move`` no-ops on them). Contact height is
    ``height_frac`` of the CoM height above the table (below the CoM: less
    tipping torque), >= 1 cm.

    Gripper yaw: under TOP_DOWN_QUAT the finger axis (EE y; franka
    delta_matrix maps it onto panda_finger_joint's axis) is world x, so a
    ±x push touched the box with ONE fingertip and the box yawed off the
    line (diag job 22636290: franka tea box rotated ~80 deg and slid off the
    front edge). Every push pose is yawed so the finger pair lies ACROSS the
    push direction (two-finger line contact), with the smallest wrist
    rotation (the closed gripper is symmetric under 180 deg).

    Returns the skill list: close gripper -> pre-push (``prepush_dist``
    behind the contact face, fingers vertical, CoM height clamped >= 1 cm
    above the table on the fingertip plane) -> max_segments RiskyStep
    segments -> retreat 4 cm back along the initial push direction + 6 cm up. Execute through
    ``execute_steps`` or the zipped ``realize_step`` pairs (push_zone).
    """
    import numpy as np

    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    table_z = TABLE_TOP_Z + float(getattr(task, "table_z_bias", 0.0))
    state = {"overshoot_left": float(overshoot), "done": False, "d": None, "d0": None, "dist": None, "grew": 0,
             "line0": None}

    def _live():
        pose = _unwrap(obj).get_pose()
        center = np.asarray(pose.p, dtype=np.float64).reshape(3)
        rot = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)[:3, :3]
        half = _obb_half_world(obj)
        if half is None:
            half = np.array([0.03, 0.03, 0.03], dtype=np.float64)
        rem = goal - center[:2]
        dist = float(np.linalg.norm(rem))
        d = rem / dist if dist > 1e-9 else (state["d"] if state["d"] is not None else np.array([1.0, 0.0]))
        depth = float(np.abs(rot.T @ np.array([d[0], d[1], 0.0])) @ half)
        # contact BELOW the CoM (height_frac of its height above the table):
        # a fingertip at CoM height rolled franka's tall tea box onto its side
        # (diag job 22636635 seed 0: box z 0.741 -> 0.774, ~170 deg)
        if tip_len is not None:
            # fingertip bottoms tip_clear above the table (z_contact is the TCP
            # height _pose adds APPROACH_TO_TCP to)
            z_contact = table_z + float(tip_clear) + float(tip_len) - APPROACH_TO_TCP + float(height_offset)
        else:
            z_contact = max(table_z + float(height_frac) * (float(center[2]) - table_z) + float(height_offset),
                            table_z + 0.01)
        perp = np.array([-d[1], d[0]], dtype=np.float64)
        face = center[:2] - d * depth + perp * float(lateral_offset)
        return d, dist, face, z_contact

    def _pose(xy, z_contact, d):
        return [float(xy[0]), float(xy[1]), z_contact + APPROACH_TO_TCP] + _push_quat(d)

    def _prepush(task):
        # approach from ABOVE: the planner's world holds only the table, so a
        # direct path to the pre-push pose can sweep through the box and knock
        # it over before the push starts (diag job 22640284, ur5 seed 1: the
        # coffee box ended rotated 98 deg at its start)
        d, _, face, zc = _live()
        state["d"] = state["d0"] = d
        state["line0"] = np.asarray(_unwrap(obj).get_pose().p, dtype=np.float64).reshape(3)[:2].copy()
        pre = face - d * float(prepush_dist)
        above = _pose(pre, zc, d)
        above[2] += float(approach_lift)
        return arm_tag, task.move_to_pose(arm_tag, above)[1] + task.move_to_pose(arm_tag, _pose(pre, zc, d))[1]

    def _segment(task):
        if state["done"]:
            return arm_tag, []
        d, dist, face, zc = _live()
        if dist <= goal_tol and state["overshoot_left"] <= 1e-9:
            state["done"] = True
            return arm_tag, []
        box_z = float(np.asarray(_unwrap(obj).get_pose().p, dtype=np.float64).reshape(3)[2])
        if box_z < table_z - 0.02:  # fell off the table: nothing left to push
            state["done"] = True
            return arm_tag, []
        if state["dist"] is not None and dist > state["dist"] + 0.005:
            state["grew"] += 1
            if state["grew"] >= 2:  # moving away from the goal twice: box lost
                state["done"] = True
                return arm_tag, []
        else:
            state["grew"] = 0
        state["dist"] = dist
        if state["line0"] is not None:
            # lane guard: a box deflected off its start->goal line is LOST —
            # chasing it sideways plowed franka's hand into both bystanders
            # (diag job 22636635 seed 4); stop and let the outcome stand
            line = goal - state["line0"]
            n = float(np.linalg.norm(line))
            if n > 1e-9:
                c = np.asarray(_unwrap(obj).get_pose().p, dtype=np.float64).reshape(3)[:2] - state["line0"]
                off = abs(float(c[0] * line[1] - c[1] * line[0]) / n)
                if off > float(lane_tol):
                    state["done"] = True
                    return arm_tag, []
        pre = face - d * float(approach_gap)
        travel = dist + float(approach_gap)
        last = travel <= float(seg_len)
        length = travel + state["overshoot_left"] if last else float(seg_len)
        if last:
            state["overshoot_left"] = 0.0
            state["done"] = True
        actions = []
        ee = np.asarray(task.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)
        if float(np.linalg.norm(ee[:2] - pre)) > float(realign_tol):
            d_old = state["d"] if state["d"] is not None else d
            actions += task.move_to_pose(arm_tag, _pose(ee[:2] - d_old * 0.02, zc, d_old))[1]
            actions += task.move_to_pose(arm_tag, _pose(pre, zc, d))[1]
        actions += task.move_to_pose(arm_tag, _pose(pre + d * length, zc, d))[1]
        state["d"] = d
        return arm_tag, actions

    def _retreat(task):
        # back along the INITIAL push direction (toward the arm's own side):
        # after an overshoot the re-aimed heading points back at the goal, and
        # retreating along it went further across the midline, out of reach
        # (diag job 22636290 ur5 seed 0: plan failure at x = 0.235)
        d = state["d0"] if state["d0"] is not None else np.array([1.0, 0.0])
        return task.move_by_displacement(arm_tag, x=float(-d[0] * 0.04), y=float(-d[1] * 0.04), z=0.06)

    return [
        task.close_gripper(arm_tag),
        _Deferred(arm_tag, _prepush),
        *[RiskyStep(_Deferred(arm_tag, _segment)) for _ in range(int(max_segments))],
        _Deferred(arm_tag, _retreat),
    ]


# ---------------------------------------------------------------------------
# safe push along a route (T3 nominal_mode="safe", Yuxuan 2026-10-27): new
# functions only — push_to_goal and _push_quat above stay the careless push
# ---------------------------------------------------------------------------


def _ee_yaw(quat):
    """Yaw psi of a top-down EE quaternion (wxyz) about world z, from TOP_DOWN_QUAT: the finger axis (EE y) points
    along world (-cos psi, -sin psi)."""
    import numpy as np

    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(-(1.0 - 2.0 * (x * x + z * z)), -2.0 * (x * y - w * z)))


def _yaw_quat(psi):
    """TOP_DOWN_QUAT yawed by psi about world z (wxyz): _push_quat's composition."""
    import numpy as np

    w1, x1, y1, z1 = float(np.cos(psi / 2.0)), 0.0, 0.0, float(np.sin(psi / 2.0))
    w2, x2, y2, z2 = TOP_DOWN_QUAT
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _wrap_pi(a):
    """``a`` wrapped into [-pi, pi)."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _push_yaws(push_dir_xy, live_yaw, hand="wide"):
    """The two yaws 180 deg apart that push alike along ``push_dir_xy`` (the closed gripper is symmetric; the finger
    axis across the push, ``hand`` "narrow": along it): (the one nearest ``live_yaw``, the other). A tie (exactly
    90 deg either way) keeps _push_quat's fold first."""
    import numpy as np

    d = np.asarray(push_dir_xy, dtype=np.float64).reshape(2)
    psi = float(np.arctan2(d[0], -d[1]) - np.pi)   # R_z(psi) @ (-1, 0) == (-d_y, d_x), as in _push_quat
    if hand == "narrow":
        psi += np.pi / 2.0
    psi = (psi + np.pi / 2.0) % np.pi - np.pi / 2.0
    if abs(_wrap_pi(psi + np.pi - live_yaw)) < abs(_wrap_pi(psi - live_yaw)) - 1e-9:
        psi += np.pi
    return psi, psi + np.pi


def _push_quat_near(push_dir_xy, ee_quat, hand="wide"):
    """_push_quat's gripper pose (the finger axis across the push; ``hand`` "narrow": along it) with, of the two
    yaws 180 deg apart that push alike (the closed gripper is symmetric), the one nearest the live EE yaw.
    _push_quat folds the yaw into [-90, 90) deg, so its target jumps 180 deg whenever a re-aimed push crosses pure
    +-x (careless seed 6400: plan #9 "large joint travel 3.23 rad" on a 10 cm segment, the episode failed). A tie
    (exactly 90 deg either way) keeps _push_quat's fold."""
    return [float(v) for v in _yaw_quat(_push_yaws(push_dir_xy, _ee_yaw(ee_quat), hand)[0])]


# The wrist_3 limit (T3 safe mode). ur5's last joint turns a top-down hand about the vertical: yawing the hand by
# dpsi turns wrist_3 by -dpsi, and the shoulder pan's change adds to it (the review's fit on the careless T3 log
# runs/lang/t6_1023/tilt.log, seeds 6400 / 6401: wrist_3 ~= pan - psi + 1.58 within 0.15 rad, pan ~= the EE's bearing
# about the arm base + const). The URDF keeps wrist_3 within +-pi: a yaw that needs more makes cuRobo turn the
# wrist the other way, nearly a full turn (or flip to another IK branch).
WRIST_LIMIT = math.pi
WRIST_MARGIN = 0.3        # rad kept off the limit: the per-move prediction leaves out the pan's change in the move
JOINT_TRAVEL_MAX = 2.0    # rad, any arm joint, one executed move (the controller's BRANCH_FLIP_TRAVEL)


def arm_qpos(task, arm_tag):
    """The arm's live joint positions (RoboTwin robot.get_<arm>_arm_real_jointState without its gripper value;
    ur5: pan, lift, elbow, wrist_1, wrist_2, wrist_3), None without a robot (the CPU tests' fakes)."""
    import numpy as np

    fn = getattr(getattr(task, "robot", None), f"get_{arm_tag}_arm_real_jointState", None)
    if fn is None:
        return None
    try:
        q = np.asarray(fn(), dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001  (diagnostics never break the push)
        return None
    return q[:-1].copy() if q.size > 1 else None


def _arm_base_xy(task, arm_tag):
    """World xy of the arm's base (RoboTwin robot.<arm>_entity_origion_pose), None without a robot."""
    import numpy as np

    pose = getattr(getattr(task, "robot", None), f"{arm_tag}_entity_origion_pose", None)
    p = getattr(pose, "p", None)
    return None if p is None else np.asarray(p, dtype=np.float64).reshape(-1)[:2].copy()


def pick_yaw(push_dir_xy, live_yaw, w3, lifted, hand="wide", band=WRIST_LIMIT - WRIST_MARGIN):
    """push_via's yaw for one move: (yaw, None), or (None, why) when no yaw keeps wrist_3 within +-``band``. Of the
    two yaws that push alike, the one nearest ``live_yaw`` if its predicted wrist_3 w3' = w3 - wrap(yaw - live_yaw)
    stays within the band, else (only when ``lifted``: a lifted corner, the 180 deg switch) the other one. ``w3``
    None (no joint readings): the nearest. Example: live yaw 0, a push needing yaw -0.5 or +2.64, w3 = 2.5: the
    nearest gives w3' = 3.0 > 2.84 (pi - 0.3); lifted, the other gives -0.14 and is taken; not lifted, none."""
    near, other = _push_yaws(push_dir_xy, live_yaw, hand)
    if w3 is None:
        return near, None
    pred = []
    for k, psi in enumerate((near, other)):
        if k and not lifted:
            break
        w = float(w3) - _wrap_pi(psi - live_yaw)
        pred.append(round(w, 3))
        if abs(w) <= band:
            return psi, None
    return None, {"w3": round(float(w3), 3), "pred": pred, "lifted": bool(lifted)}


def wrist_track(w3, live_yaw, ee_xy, path, first, base_xy=None, hand="wide"):
    """Predicted wrist_3 along a push: the live wrist_3 ``w3``, EE yaw ``live_yaw`` and EE xy ``ee_xy``, then
    ``path`` [(EE xy, push direction), ...] with, at the first entry, the yaw ``first`` (0: nearest the live yaw,
    1: the other one) and after it the yaw nearest the previous one (push_via between lifted corners). Each move
    turns wrist_3 by the pan's change (the EE's bearing change about ``base_xy``; none when None) minus the yaw's.
    Returns the predicted wrist_3 after each entry."""
    import numpy as np

    w, psi, p = float(w3), float(live_yaw), np.asarray(ee_xy, dtype=np.float64).reshape(2)

    def bearing(xy):
        return math.atan2(float(xy[1]) - float(base_xy[1]), float(xy[0]) - float(base_xy[0]))

    out = []
    for i, (xy, d) in enumerate(path):
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        new = _push_yaws(d, psi, hand)[int(first) if i == 0 else 0]
        if base_xy is not None:
            w += _wrap_pi(bearing(xy) - bearing(p))
        w -= _wrap_pi(new - psi)
        psi, p = new, xy
        out.append(w)
    return out


def push_via(task, arm_tag, obj, via_xy, guard=None, replan=None, on_fail=None, log=None, seg_len=0.06,
             via_tol=0.012, goal_tol=0.005, goal_done_tol=0.02, lane_tol=0.02, approach_gap=0.015,
             descend_gap=0.03, backoff=0.03, realign_backoff=0.02, realign_tol=0.01, lift_turn_deg=30.0,
             corner_lift=0.02, hand="wide", contact=None, tip_len=None, tip_clear=0.008, height_offset=0.0,
             height_frac=0.6, approach_lift=0.10, retreat_lift=0.08, max_segments=30, max_replans=4,
             max_box_tilt_deg=45.0, wrist_limit=WRIST_LIMIT, wrist_margin=WRIST_MARGIN, travel_max=JOINT_TRAVEL_MAX):
    """Closed-loop table-top push of ``obj`` along the via points ``via_xy`` (the goal last): the T3 safe mode's
    executor (nominal_mode="safe", Yuxuan 2026-10-27; push_to_goal stays the careless push).

    One leg per via point. Every push segment is re-aimed at the leg's via point from the LIVE box, at most
    ``seg_len`` and never past the via point (no overshoot); a via point counts as reached within ``via_tol`` or
    once the box passed it. Before each segment the hand is put behind the box for the new heading:

      d     = unit(via - box centre)                    (re-aimed every segment)
      touch = centre - d * (OBB depth along d + contact)  (the finger front face on the box face)
      push  = touch + d * min(seg_len, distance to the via point)

    - a corner (the heading turns more than ``lift_turn_deg``): back off ``backoff`` along the old heading, lift the
      fingertips ``corner_lift`` above the box top, move across to ``descend_gap`` behind the new face and come down
      (push_to_goal's approach from above): the hand never swings around the box at push height;
    - a smaller turn, or the hand off its push line by more than ``realign_tol``: back off ``realign_backoff`` and
      re-align at push height (push_to_goal's re-alignment).

    Gripper yaw, of the two yaws 180 deg apart that push alike (wrist_3 limit, the arm's last joint: ur5 +-pi). With
    the arm's joints readable (arm_qpos), wrist_3 after a move is predicted as w3' = w3 - wrap(psi' - psi_live):
    - the first approach (from above, the hand far from everything) takes the yaw whose predicted wrist_3 along the
      whole route (wrist_track: the pan's change from the arm base's bearing, the yaw continuous after it) stays
      within +-(``wrist_limit`` - ``wrist_margin``), the one nearest the live yaw if both do, the smaller peak if
      neither does;
    - every later move keeps the yaw nearest the live one (no 180 deg flip when the heading crosses +-x) while its
      w3' stays in that band; only at a lifted corner it may take the other one (the 180 deg switch: the hand turns
      in place, lifted, before it moves across); no yaw in the band stops the push ("wrist_limit") before the move;
    - a turn larger than ``travel_max`` - ``wrist_margin`` (the first approach, a switch) is split into equal parts,
      one move each, so no move asks a joint for more than push_zone's ``travel_max`` check allows.
    Without joint readings (fakes) the yaw nearest the live one, always. Push height: push_to_goal's (``tip_len``:
    fingertip bottoms ``tip_clear`` above the table; else the TCP model at ``height_frac`` of the CoM height).

    Safety, the safe mode never falls back to an unsafe motion: ``guard(ee_poses, box_from, box_to, box_yaw)`` ->
    (ok, why) checks every step before it runs (the EE poses it commands, the live one first, and the box slide of
    the push). A guard failure, or the box more than ``lane_tol`` off its leg line, asks ``replan(centre, yaw)``
    for new via points from the live box; with none (or more than ``max_replans``) the push stops: the remaining
    steps are empty and ``on_fail(reason)`` runs (push_zone marks a plan failure and writes
    episode_info["safe_fail"]). Reasons: "guard", "lane", "box_fell", "box_tipped" (tilt > ``max_box_tilt_deg``),
    "box_lost" (moving away from its via point twice), "replans", "wrist_limit", "segments" (out of segments before
    the goal). A planner failure (the task's plan_success turned False: RoboTwin's move() then does nothing) stops
    the push too, without on_fail (the runner counts it as a plan failure; log "fail" = "plan").

    ``log`` (a dict, filled in place): the steps taken, the re-plans, the largest yaw change between consecutive
    pushes, the yaw switches, the largest EE miss (live EE vs the last commanded pose; ``ee_miss_flag`` above 2 cm),
    the box's largest tilt, the failure reason; "max_joint_travel" is push_zone's (it executes the moves).

    Returns the skill list: close gripper -> approach from ``approach_lift`` above, ``descend_gap`` behind the box
    -> ``max_segments`` x (re-position, RiskyStep push segment) -> retreat (back ``approach_gap``, up
    ``retreat_lift``). The first two entries pair with the idle arm's close / tuck, as push_to_goal's do. Execute
    through ``execute_steps`` or the zipped ``realize_step`` pairs (push_zone).
    """
    import numpy as np

    table_z = TABLE_TOP_Z + float(getattr(task, "table_z_bias", 0.0))
    contact = float(contact) if contact is not None else (0.015 if hand == "wide" else 0.0227)  # WSG finger faces
    band = float(wrist_limit) - float(wrist_margin)        # |predicted wrist_3| kept within
    turn_step = float(travel_max) - float(wrist_margin)    # largest yaw change per move
    log = {} if log is None else log
    log.update(steps=[], replans=[], yaw_jump_max_deg=0.0, ee_miss_max=0.0, ee_miss_flag=False, box_tilt_max_deg=0.0,
               fail=None, wrist_switches=0, wrist_first=None, max_joint_travel=0.0)
    state = {"via": [np.asarray(v, dtype=np.float64).reshape(2) for v in via_xy], "leg": 0, "start": None,
             "d": None, "done": False, "dist": None, "grew": 0, "replans": 0, "skip": False, "last_cmd": None,
             "last_yaw": None, "up": None}

    def _fail(reason):
        state["done"] = True
        if log["fail"] is None:
            log["fail"] = str(reason)
            if on_fail is not None:
                on_fail(str(reason))

    def _planner_failed():
        # a cuRobo failure in an earlier move: RoboTwin's move() ignores everything after it, so stop here (no
        # on_fail: the runner counts the plan failure, push_zone records plan_fail_step)
        if getattr(task, "plan_success", True) is False:
            state["done"] = True
            log["fail"] = log["fail"] or "plan"
            return True
        return False

    def _live():
        pose = _unwrap(obj).get_pose()
        p = np.asarray(pose.p, dtype=np.float64).reshape(3)
        rot = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)[:3, :3]
        half = _obb_half_world(obj)
        if half is None:
            half = np.array([0.03, 0.03, 0.03], dtype=np.float64)
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        if state["up"] is None:   # the box's up axis, latched at the start (prim boxes z, RoboTwin meshes y)
            state["up"] = int(np.argmax(np.abs(rot[2, :])))
        tilt = float(np.degrees(np.arccos(np.clip(abs(rot[2, state["up"]]), 0.0, 1.0))))
        top = float(p[2] + np.abs(rot[2, :]) @ half)
        return p, rot, half, yaw, tilt, top

    def _depth(rot, half, d):
        return float(np.abs(rot.T @ np.array([d[0], d[1], 0.0])) @ half)

    def _z_push(center_z):
        if tip_len is not None:
            zc = table_z + float(tip_clear) + float(tip_len) - APPROACH_TO_TCP + float(height_offset)
        else:
            zc = max(table_z + float(height_frac) * (center_z - table_z) + float(height_offset), table_z + 0.01)
        return zc + APPROACH_TO_TCP

    def _pose(xy, z, quat):
        return [float(xy[0]), float(xy[1]), float(z)] + [float(v) for v in quat]

    def _ee():
        return np.asarray(task.get_arm_pose(arm_tag), dtype=np.float64).reshape(7)

    def _xy(v):
        return [round(float(v[0]), 4), round(float(v[1]), 4)]

    def _guard(poses, c0, c1, yaw):
        if guard is None:
            return True, None
        res = guard(poses, c0, c1, yaw)
        if isinstance(res, (tuple, list)):
            return bool(res[0]), (res[1] if len(res) > 1 else None)
        return bool(res), None

    def _yaw(d, ee, lifted):
        """(yaw, None) or (None, why) for push direction d from the live EE (pick_yaw on the live wrist_3)."""
        live = _ee_yaw(ee[3:])
        q = arm_qpos(task, arm_tag)
        psi, why = pick_yaw(d, live, None if q is None else float(q[-1]), lifted, hand=hand, band=band)
        if psi is not None and abs(_wrap_pi(psi - live)) > math.pi / 2.0 + 1e-9:
            log["wrist_switches"] += 1
        return psi, why

    def _turns(psi_from, psi_to):
        """The yaws a turn psi_from -> psi_to (the short way of the full circle) passes in parts of at most
        turn_step, the last one psi_to."""
        dpsi = _wrap_pi(psi_to - psi_from)
        n = max(1, int(np.ceil(abs(dpsi) / max(turn_step, 1e-3) - 1e-9)))
        return [psi_from + dpsi * (j + 1) / n for j in range(n - 1)] + [psi_to]

    def _replan(c, yaw, why):
        """True: new via points from the live box (the next step re-positions for them); False: the push stops."""
        if replan is None:
            _fail(why)
            return False
        if state["replans"] >= int(max_replans):
            _fail("replans")
            return False
        state["replans"] += 1
        new = replan(c, yaw)
        log["replans"].append({"at": _xy(c), "why": why, "ok": bool(new)})
        if not new:
            _fail(why)
            return False
        state["via"] = [np.asarray(v, dtype=np.float64).reshape(2) for v in new]
        state.update(leg=0, start=np.asarray(c, dtype=np.float64).copy(), dist=None, grew=0)
        return True

    def _target(c):
        """The current via point, past the reached ones; (via point, is the goal)."""
        via = state["via"]
        while state["leg"] < len(via) - 1:
            tgt = via[state["leg"]]
            line = tgt - state["start"]
            n = float(np.linalg.norm(line))
            passed = n > 1e-9 and float((c - tgt) @ line) / n >= 0.0
            if float(np.linalg.norm(c - tgt)) > float(via_tol) and not passed:
                break
            state["leg"] += 1
            state["start"] = c.copy()
            state["dist"], state["grew"] = None, 0
        return via[state["leg"]], state["leg"] == len(via) - 1

    def _off_lane(c, tgt):
        line = tgt - state["start"]
        n = float(np.linalg.norm(line))
        if n < 1e-9:
            return 0.0
        rel = c - state["start"]
        return abs(float(rel[0] * line[1] - rel[1] * line[0])) / n

    def _miss(ee):
        # the live EE vs the last commanded pose: more than 2 cm off flags the demo (logged, not a failure)
        if state["last_cmd"] is not None:
            log["ee_miss_max"] = max(log["ee_miss_max"], round(float(np.linalg.norm(ee[:3] - state["last_cmd"])), 4))
            log["ee_miss_flag"] = bool(log["ee_miss_max"] > 0.02)

    def _route_path(c, rot, half):
        """(EE xy, push direction) of every planned push segment, the box moved along the via points."""
        out, c = [], np.asarray(c, dtype=np.float64).copy()
        for b in state["via"]:
            for _ in range(200):
                r = b - c
                n = float(np.linalg.norm(r))
                if n < 1e-6:
                    break
                d = r / n
                out.append((c - d * (_depth(rot, half, d) + contact), d))
                c = c + d * min(float(seg_len), n)
        return out

    def _first_yaw(d, ee, c, rot, half):
        """The first approach's yaw (see the docstring): (yaw, None) or (None, why)."""
        live = _ee_yaw(ee[3:])
        cands = _push_yaws(d, live, hand)
        q = arm_qpos(task, arm_tag)
        if q is None:
            return cands[0], None
        w3 = float(q[-1])
        path = _route_path(c, rot, half)
        base = _arm_base_xy(task, arm_tag)
        peaks = []
        for k in (0, 1):
            first_ok = abs(w3 - _wrap_pi(cands[k] - live)) <= band
            track = wrist_track(w3, live, ee[:2], path, k, base_xy=base, hand=hand)
            peaks.append((first_ok, max([abs(w) for w in track] or [abs(w3)])))
        ok = [k for k in (0, 1) if peaks[k][0] and peaks[k][1] <= band]
        if ok:
            k = ok[0]
        else:
            fine = [k for k in (0, 1) if peaks[k][0]]
            if not fine:
                return None, {"w3": round(w3, 3), "first": True}
            k = min(fine, key=lambda j: peaks[j][1])
        log["wrist_first"] = {"w3": round(w3, 3), "pick": "near" if k == 0 else "other",
                              "peaks": [round(peaks[0][1], 3), round(peaks[1][1], 3)]}
        return cands[k], None

    def _prepush(task):
        # approach from ABOVE, descend_gap behind the box face (push_to_goal's approach): a direct path to a pose
        # beside the box can sweep through it
        if _planner_failed():
            return arm_tag, []
        p, rot, half, yaw, _, _ = _live()
        c = p[:2]
        state["start"] = c.copy()
        tgt = state["via"][0]
        dist = float(np.linalg.norm(tgt - c))
        d = (tgt - c) / dist if dist > 1e-9 else np.array([1.0, 0.0])
        ee = _ee()
        psi, why = _first_yaw(d, ee, c, rot, half)
        if psi is None:
            log["steps"].append({"kind": "approach", "c": _xy(c), "wrist": why})
            _fail("wrist_limit")
            return arm_tag, []
        quat = _yaw_quat(psi)
        far = c - d * (_depth(rot, half, d) + contact + float(descend_gap))
        z_push = _z_push(float(p[2]))
        # the turn from the live yaw happens above, split when it is large (the far yaw from the home pose)
        poses = [_pose(far, z_push + float(approach_lift), _yaw_quat(a)) for a in _turns(_ee_yaw(ee[3:]), psi)]
        poses.append(_pose(far, z_push, quat))
        ok, why = _guard([ee.tolist()] + poses, c, c, yaw)
        if not ok:
            log["steps"].append({"kind": "approach", "c": _xy(c), "guard": why})
            _fail("guard")
            return arm_tag, []
        state["d"] = d
        state["last_cmd"] = np.asarray(poses[-1][:3])
        state["last_yaw"] = psi
        log["steps"].append({"kind": "approach", "c": _xy(c), "n_turns": len(poses) - 1,
                             "d_deg": round(float(np.degrees(np.arctan2(d[1], d[0]))), 1)})
        actions = []
        for pose in poses:
            actions += task.move_to_pose(arm_tag, pose)[1]
        return arm_tag, actions

    def _reposition(task):
        if state["done"] or _planner_failed():
            return arm_tag, []
        p, rot, half, yaw, tilt, top = _live()
        c = p[:2]
        log["box_tilt_max_deg"] = max(log["box_tilt_max_deg"], round(tilt, 1))
        if p[2] < table_z - 0.02:   # fell off the table
            _fail("box_fell")
            return arm_tag, []
        if tilt > float(max_box_tilt_deg):
            _fail("box_tipped")
            return arm_tag, []
        tgt, last = _target(c)
        dist = float(np.linalg.norm(tgt - c))
        d_old = state["d"]
        if last and (dist <= float(goal_tol) or (dist <= float(goal_done_tol) and d_old is not None
                                                 and float((c - tgt) @ d_old) >= 0.0)):
            state["done"] = True   # at the goal (or just past it): no turning back around the box
            return arm_tag, []
        if state["dist"] is not None and dist > state["dist"] + 0.005:
            state["grew"] += 1
            if state["grew"] >= 2:   # moving away from its via point twice: box lost
                _fail("box_lost")
                return arm_tag, []
        else:
            state["grew"] = 0
        state["dist"] = dist
        if _off_lane(c, tgt) > float(lane_tol):
            if not _replan(c, yaw, "lane"):
                return arm_tag, []
            tgt, last = _target(c)
            dist = float(np.linalg.norm(tgt - c))
        d = (tgt - c) / max(dist, 1e-9)
        d_old = d if d_old is None else d_old
        ee = _ee()
        _miss(ee)
        turn = float(np.degrees(np.arccos(np.clip(float(d_old @ d), -1.0, 1.0))))
        lifted = turn > float(lift_turn_deg)
        psi, why = _yaw(d, ee, lifted)
        entry = {"kind": "lift" if lifted else "none", "leg": int(state["leg"]), "c": _xy(c),
                 "turn_deg": round(turn, 1)}
        if psi is None:
            entry["wrist"] = why
            log["steps"].append(entry)
            _fail("wrist_limit")
            return arm_tag, []
        quat = _yaw_quat(psi)
        touch = c - d * (_depth(rot, half, d) + contact)
        z_push = _z_push(float(p[2]))
        rel = touch - ee[:2]
        lateral = abs(float(rel[0] * d[1] - rel[1] * d[0]))
        behind = float(rel @ d)
        live_yaw = _ee_yaw(ee[3:])
        dyaw = abs(_wrap_pi(psi - live_yaw))
        in_place = (lateral <= float(realign_tol) and -0.005 <= behind <= float(descend_gap) + float(realign_tol)
                    and np.degrees(dyaw) <= 10.0 and abs(float(ee[2]) - z_push) <= 0.01)
        poses = []
        if lifted:
            back = ee[:2] - d_old * float(backoff)
            fingers = float(tip_len) if tip_len is not None else APPROACH_TO_TCP
            z_lift = max(z_push, top + float(corner_lift) + fingers)
            far = touch - d * float(descend_gap)
            poses = [_pose(back, ee[2], ee[3:]), _pose(back, z_lift, ee[3:])]
            if dyaw > math.pi / 2.0 + 1e-9:
                # the 180 deg switch: turn in place, lifted (in parts when large), then move across
                poses += [_pose(back, z_lift, _yaw_quat(a)) for a in _turns(live_yaw, psi)]
            poses += [_pose(far, z_lift, quat), _pose(far, z_push, quat)]
        elif not in_place:
            back = ee[:2] - d_old * float(realign_backoff)
            poses = [_pose(back, ee[2], ee[3:]), _pose(touch - d * float(approach_gap), z_push, quat)]
            entry["kind"] = "realign"
        move = min(float(seg_len), dist)
        ok, why = _guard([ee.tolist()] + poses + [_pose(touch + d * move, z_push, quat)], c, c + d * move, yaw)
        if not ok:
            entry["guard"] = why
            log["steps"].append(entry)
            state["skip"] = True   # the push below waits: the next step re-positions for the new route
            _replan(c, yaw, "guard")
            return arm_tag, []
        log["steps"].append(entry)
        actions = []
        for pose in poses:
            actions += task.move_to_pose(arm_tag, pose)[1]
        if poses:
            state["last_cmd"] = np.asarray(poses[-1][:3])
        return arm_tag, actions

    def _push(task):
        if state["done"] or _planner_failed():
            return arm_tag, []
        if state["skip"]:
            state["skip"] = False
            return arm_tag, []
        p, rot, half, yaw, _, _ = _live()
        c = p[:2]
        tgt = state["via"][state["leg"]]
        dist = float(np.linalg.norm(tgt - c))
        if dist < 1e-6:
            return arm_tag, []
        d = (tgt - c) / dist
        ee = _ee()
        _miss(ee)
        psi, why = _yaw(d, ee, False)
        if psi is None:
            log["steps"].append({"kind": "push", "leg": int(state["leg"]), "c": _xy(c), "wrist": why})
            _fail("wrist_limit")
            return arm_tag, []
        quat = _yaw_quat(psi)
        move = min(float(seg_len), dist)
        target = _pose(c - d * (_depth(rot, half, d) + contact) + d * move, _z_push(float(p[2])), quat)
        ok, why = _guard([ee.tolist(), target], c, c + d * move, yaw)
        if not ok:
            log["steps"].append({"kind": "push", "leg": int(state["leg"]), "c": _xy(c), "guard": why})
            _replan(c, yaw, "guard")   # a new route waits for the next re-position step
            return arm_tag, []
        if state["last_yaw"] is not None:
            jump = abs(_wrap_pi(psi - state["last_yaw"]))
            log["yaw_jump_max_deg"] = max(log["yaw_jump_max_deg"], round(float(np.degrees(jump)), 1))
        state["last_yaw"] = psi
        state["d"] = d
        state["last_cmd"] = np.asarray(target[:3])
        log["steps"].append({"kind": "push", "leg": int(state["leg"]), "c": _xy(c), "move": round(move, 4),
                             "d_deg": round(float(np.degrees(np.arctan2(d[1], d[0]))), 1),
                             "yaw_deg": round(float(np.degrees(_wrap_pi(psi))), 1)})
        return arm_tag, task.move_to_pose(arm_tag, target)[1]

    def _retreat(task):
        if _planner_failed() or log["fail"] is not None:
            return arm_tag, []
        if not state["done"]:
            _fail("segments")
            return arm_tag, []
        p, _, _, yaw, _, _ = _live()
        ee = _ee()
        d = state["d"] if state["d"] is not None else np.array([1.0, 0.0])
        back = ee[:2] - d * float(approach_gap)
        poses = [_pose(back, ee[2], ee[3:]), _pose(back, float(ee[2]) + float(retreat_lift), ee[3:])]
        if not _guard([ee.tolist()] + poses, p[:2], p[:2], yaw)[0]:
            poses = [_pose(ee[:2], float(ee[2]) + float(retreat_lift), ee[3:])]   # straight up only
            if not _guard([ee.tolist()] + poses, p[:2], p[:2], yaw)[0]:
                return arm_tag, []   # the box is home: stay rather than move into something
        actions = []
        for pose in poses:
            actions += task.move_to_pose(arm_tag, pose)[1]
        return arm_tag, actions

    steps = [task.close_gripper(arm_tag), _Deferred(arm_tag, _prepush)]
    for _ in range(int(max_segments)):
        steps += [_Deferred(arm_tag, _reposition), RiskyStep(_Deferred(arm_tag, _push))]
    steps.append(_Deferred(arm_tag, _retreat))
    return steps


# ---------------------------------------------------------------------------
# granular_pour (T2) helpers: bowl geometry + bead grid + bead spawner
# ---------------------------------------------------------------------------
# Geometry source of truth: the collision mesh pieces of the asset, measured
# at RUNTIME (never the json extents — see probes/p2.py, whose pure-numpy
# measurement this mirrors; feasibility.md P2 used it to pass the four
# 002_bowl ids). The glb pieces sit in per-node local frames, so the
# scene-graph node transforms are baked in before scaling, matching what
# sapien cooks. All returned quantities live in the asset FILE frame
# (glb convention: y = up, origin at the bowl's exterior bottom).

POUR_BAND_H = 0.0025  # m, interior wall-radius band height (taper profile)


def _collision_verts(modelname, model_id, obj_root=None):
    """All collision-piece vertices in the file frame, scaled to metres."""
    import json
    from pathlib import Path

    import numpy as np
    import trimesh

    root = Path(obj_root) if obj_root is not None else Path("assets/objects")
    modeldir = root / str(modelname)
    collision = modeldir / "collision" / f"base{int(model_id)}.glb"
    if not collision.exists():
        collision = modeldir / "collision" / f"textured{int(model_id)}.obj"
    if not collision.exists():
        collision = modeldir / f"base{int(model_id)}.glb"
    if not collision.exists():
        collision = modeldir / f"textured{int(model_id)}.obj"
    with open(modeldir / f"model_data{int(model_id)}.json", "r", encoding="utf-8") as fh:
        scale = np.asarray(json.load(fh)["scale"], dtype=np.float64)
    loaded = trimesh.load(str(collision), force="scene", process=False)
    verts = []
    if hasattr(loaded, "graph") and hasattr(loaded, "geometry"):
        for node in loaded.graph.nodes_geometry:
            T, geom_name = loaded.graph[node]
            g = loaded.geometry[geom_name]
            v = getattr(g, "vertices", None)
            if v is None or len(v) < 4:
                continue
            verts.append(np.asarray(v, dtype=np.float64) @ T[:3, :3].T + T[:3, 3])
    else:
        v = getattr(loaded, "vertices", None)
        if v is not None and len(v) >= 4:
            verts.append(np.asarray(v, dtype=np.float64))
    if not verts:
        raise ValueError(f"no collision geometry in {collision}")
    return np.concatenate(verts, axis=0) * scale


def _bowl_geometry_from_verts(verts):
    """Container geometry estimates in the file frame (y = up).

    - rim_y / bottom_y: collision-AABB top / bottom along the up axis.
    - outer_radius: half the smaller horizontal AABB extent.
    - inner_radius: minimum horizontal radius over vertices in the top 15%
      band — the inner edge of the rim ring. This is the ``rim_radius``
      container regions are built with.
    - floor_y: highest vertex below half height within 0.4 x inner radius of
      the axis — the interior floor surface (bead spawn reference).
    - bands: [[y_lo, min_rho], ...] — inner wall radius per POUR_BAND_H-tall
      band, floor to rim (the bowls TAPER toward the bottom, so the spawn
      grid sizes each layer by the wall radius over the bead's extent).
    - half_extents / obb_center_local: the full AABB (for ItemSpec).
    """
    import numpy as np

    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    lo, hi = v.min(0), v.max(0)
    rim_y, bottom_y = float(hi[1]), float(lo[1])
    outer_radius = 0.5 * float(min(hi[0] - lo[0], hi[2] - lo[2]))
    rho = np.linalg.norm(v[:, [0, 2]], axis=1)
    top_band = v[v[:, 1] > rim_y - 0.15 * (rim_y - bottom_y)]
    inner_radius = float(np.linalg.norm(top_band[:, [0, 2]], axis=1).min()) if len(top_band) else outer_radius
    central = v[(rho < 0.4 * inner_radius) & (v[:, 1] < 0.5 * rim_y)]
    floor_y = float(central[:, 1].max()) if len(central) else bottom_y
    bands = []
    y_lo = floor_y + 0.5 * POUR_BAND_H
    while y_lo < rim_y - POUR_BAND_H:
        band = (v[:, 1] >= y_lo) & (v[:, 1] < y_lo + POUR_BAND_H)
        if band.any():
            bands.append([float(y_lo), float(rho[band].min())])
        y_lo += POUR_BAND_H
    if not bands:
        bands = [[floor_y, inner_radius]]
    return {
        "rim_y": rim_y,
        "bottom_y": bottom_y,
        "outer_radius": outer_radius,
        "inner_radius": inner_radius,
        "floor_y": floor_y,
        "bands": bands,
        "half_extents": (0.5 * (hi - lo)).tolist(),
        "obb_center_local": (0.5 * (hi + lo)).tolist(),
    }


def bowl_geometry(modelname, model_id, obj_root=None, _cache={}):
    """Measured collision-mesh geometry for a bowl asset (cached per id).

    Returns the ``_bowl_geometry_from_verts`` dict. Measured values for
    002_bowl (assets checkout 2026-10-01, also recorded in the T2 task card):
    id 1 rim_y 0.0489 / inner_r 0.0512, id 2 rim_y 0.0507 / inner_r 0.0558,
    id 3 rim_y 0.0521 / inner_r 0.0618, id 5 rim_y 0.0690 / inner_r 0.0599.
    """
    key = (str(modelname), int(model_id), str(obj_root))
    if key not in _cache:
        _cache[key] = _bowl_geometry_from_verts(_collision_verts(modelname, model_id, obj_root))
    return _cache[key]


def _wall_radius_interval(bands, y_lo, y_hi):
    """Min inner wall radius over [y_lo, y_hi] (band lookup with clamping)."""
    lo0, hi0 = bands[0][0], bands[-1][0] + POUR_BAND_H
    y_lo = max(float(y_lo), lo0)
    y_hi = min(max(float(y_hi), y_lo), hi0)
    radii = [mr for lo, mr in bands if lo + POUR_BAND_H > y_lo and lo < y_hi]
    if not radii:  # interval entirely between two bands (sub-band gap)
        mid = 0.5 * (y_lo + y_hi)
        radii = [min(bands, key=lambda b: abs(b[0] + 0.5 * POUR_BAND_H - mid))[1]]
    return float(min(radii))


def bead_grid_positions(n, r, inner_radius, floor_y, bands=None, spacing_factor=1.05, rng=None):
    """[n, 3] bead centres in the bowl up-frame (y = up, origin at exterior bottom).

    Hexagonal layers (row spacing s*sqrt(3)/2, alternate rows offset s/2),
    first layer one bead radius above ``floor_y``, layer pitch = s. Cells
    keep the bead fully inside ``inner_radius - 1 mm``; with ``bands`` (the
    taper profile from ``bowl_geometry``) each layer is instead sized by the
    wall radius over the bead's vertical extent — the bowls taper toward the
    bottom (bowl 5: 4.2 cm at the floor vs 6.0 cm at the rim) and a flat
    rim-sized grid spawns the lower layers INTO the wall (the P2
    spawn-ejection failure). Layers stack above the rim when the bowl cannot
    hold ``n`` beads — the overflow then spills during settle (by design in
    P2's r = 8 mm cells). At the T2 default 100 x 6 mm the top grid layer
    sits just below the rim plane (~1 cm of bead-center clearance) — near
    the rim on purpose (easier transport spill); rest containment itself was
    validated at the P2 default 60 x 6 mm.
    ``rng`` (anything with .uniform, e.g. the seeded global numpy RNG) adds a
    +/-0.2 mm horizontal jitter per bead.
    """
    import math

    import numpy as np

    if n <= 0:
        raise ValueError("n must be positive")
    if inner_radius <= r:
        raise ValueError(f"inner_radius {inner_radius} too small for bead r {r}")
    s = 2.0 * r * float(spacing_factor)
    keep_r = inner_radius - r - 0.001
    dy = s * math.sqrt(3.0) / 2.0
    wide = keep_r
    if bands is not None:
        wide = max(keep_r, max(mr for _, mr in bands) - r - 0.001)
    jmax = int(math.ceil(wide / dy)) + 1
    imax = int(math.ceil(wide / s)) + 1
    pts = []
    layer = 0
    while len(pts) < n:
        y = floor_y + r + 5e-4 + layer * s
        keep = keep_r
        if bands is not None:
            keep = max(0.0, _wall_radius_interval(bands, y - r, y + r) - r - 0.001)
        added = 0
        for j in range(-jmax, jmax + 1):
            if len(pts) >= n:
                break
            z = j * dy
            if abs(z) > keep:
                continue
            x_off = (s / 2.0) if (j % 2) else 0.0
            for i in range(-imax, imax + 1):
                x = i * s + x_off
                if x * x + z * z <= keep * keep:
                    pts.append([float(x), float(y), float(z)])
                    added += 1
        if added == 0:
            raise ValueError(f"no grid cell fits inner_radius {inner_radius} with bead r {r}")
        layer += 1
    out = np.asarray(pts[:n], dtype=np.float64)
    if rng is not None:
        jitter = rng.uniform(-2e-4, 2e-4, size=(int(n), 3))
        jitter[:, 1] = 0.0  # keep the layer height exact
        out = out + jitter
    return out


def bowl_bead_capacity(geom, r, phi=0.58):
    """Beads of radius r a random pack (volume fraction phi) holds up to the
    rim: interior volume from the measured wall-radius bands (band height
    POUR_BAND_H) / bead volume."""
    import numpy as np

    b = np.asarray(geom["bands"], dtype=np.float64)
    m = (b[:, 0] >= float(geom["floor_y"])) & (b[:, 0] + POUR_BAND_H <= float(geom["rim_y"]))
    vol = float(np.sum(np.pi * np.clip(b[m, 1] - 0.0005, 0.0, None) ** 2 * POUR_BAND_H))
    return int(round(phi * vol / (4.0 / 3.0 * np.pi * float(r) ** 3)))


def mug_geometry(model_id, modelname="039_mug", obj_root=None):
    """Geometry of a handled mug (039_mug) in a BODY-CENTRED file frame
    (y up, x / z horizontal; the asset's handle points to file -z and its
    body axis sits ``axis_offset_z`` from the file origin).

    Returns the bowl_geometry-style dict of the BODY (handle vertices
    excluded — they would corrupt the rim / wall radii) with all positions
    shifted so the body axis is the origin, plus:
      - axis_offset_z: file z of the body axis (shift the mesh by -this)
      - handle_point: [x, y, z] centre of the handle's OUTER bar at mid
        height (the grasp point), body-centred frame
      - handle_dir: [0, 0, -1] (body axis -> handle, body-centred frame)
      - half_extents / obb_center_local: AABB of body + handle.
    Measured on the collision mesh (_collision_verts)."""
    import numpy as np

    v = np.asarray(_collision_verts(modelname, model_id, obj_root), dtype=np.float64)
    r_body = float(np.abs(v[:, 0]).max())  # the handle is narrow in x
    z_c = float(v[:, 2].max()) - r_body
    v = v - np.array([0.0, 0.0, z_c])
    body = v[v[:, 2] > -r_body - 0.003]
    handle = v[v[:, 2] < -r_body - 0.003]
    g = dict(_bowl_geometry_from_verts(body))
    lo, hi = v.min(0), v.max(0)
    g["half_extents"] = (0.5 * (hi - lo)).tolist()
    g["obb_center_local"] = (0.5 * (hi + lo)).tolist()
    g["axis_offset_z"] = z_c
    if len(handle):
        z_out = float(handle[:, 2].min())
        arc = handle[handle[:, 2] < z_out + 0.010]  # outer bar of the C-loop
        y_mid = 0.5 * (float(arc[:, 1].min()) + float(arc[:, 1].max()))
        near = arc[np.abs(arc[:, 1] - y_mid) < 0.01]
        if not len(near):
            near = arc
        g["handle_point"] = [float(near[:, 0].mean()), y_mid, 0.5 * (float(near[:, 2].min()) + float(near[:, 2].max()))]
        g["handle_width"] = float(near[:, 0].max() - near[:, 0].min())
        g["handle_y_range"] = [float(arc[:, 1].min()), float(arc[:, 1].max())]  # outer bar, bottom -> top
    else:
        g["handle_point"] = None
    g["handle_dir"] = [0.0, 0.0, -1.0]
    return g


def bowl_lattice_capacity(geom, r, n_max=2000):
    """Beads of radius r that fit FULLY below the rim: hex-lattice cells
    (bead_grid_positions, wall-band aware) whose centre is <= rim_y - r.
    Exact for thin layers — the volume estimate (bowl_bead_capacity) counts
    a 2.5 cm interior 2-3x too high (a random pack of that many beads half
    spilled during the settle, diag d49)."""
    import numpy as np

    loc = bead_grid_positions(int(n_max), float(r), float(geom["inner_radius"]), float(geom["floor_y"]),
                              bands=geom.get("bands"), rng=None)
    return int((loc[:, 1] <= float(geom["rim_y"]) - float(r)).sum())


def bead_random_positions(n, r, geom, rng, gap=5e-4, column=0.10, max_tries=200000):
    """[n, 3] RANDOM non-overlapping bead centres in the bowl up-frame (y up,
    origin at the exterior bottom): random sequential addition inside the
    taper-aware interior (``geom`` = bowl_geometry dict) and, above the rim,
    a loose column of the rim's inner radius up to ``column`` above it. The
    beads settle into a RANDOM pack — a hex lattice (bead_grid_positions)
    settles into a crystal that holds far past the angle of repose (P2
    finding F3: pour start ~85 deg), so a full lattice bowl barely spills.
    ``rng``: anything with .uniform (the seeded global numpy RNG)."""
    import math

    import numpy as np

    inner = float(geom["inner_radius"])
    floor_y, rim_y = float(geom["floor_y"]), float(geom["rim_y"])
    bands = geom.get("bands")
    d_min2 = (2.0 * r + gap) ** 2
    cell = 2.0 * r + gap
    grid: dict = {}
    pts = []

    def wall(y):
        if bands is None or y - r >= rim_y:
            return inner
        return _wall_radius_interval(bands, y - r, min(y + r, rim_y))

    y_lo = floor_y + r + 5e-4
    y_hi = min(rim_y, y_lo + 0.02)
    fails = 0
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        y = float(rng.uniform(y_lo, y_hi))
        rad = max(0.0, wall(y) - r - 0.001)
        rho = rad * math.sqrt(float(rng.uniform(0.0, 1.0)))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        p = (rho * math.cos(th), y, rho * math.sin(th))
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)), int(math.floor(p[2] / cell)))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < d_min2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
        if not ok:
            fails += 1
            if fails > 400:  # this slab is saturated: grow the sampling column
                y_hi = min(y_hi + 0.004, rim_y + column)
                fails = 0
            continue
        fails = 0
        grid.setdefault(key, []).append(p)
        pts.append(p)
    if len(pts) < n:
        raise ValueError(f"bead_random_positions: placed {len(pts)} / {n}")
    return np.asarray(pts, dtype=np.float64)


def spawn_beads(scene, positions_world, radius, masses, color=(0.85, 0.15, 0.15), name_prefix="bead",
                sleep_threshold=None):
    """Spawn the bead spheres as raw entities; returns the entity list.

    Same raw-Entity idiom as ``zone_mat`` above (no ActorBuilder, render body
    only when the scene actually has a render system — headless CPU scenes
    take the collision-only path). ``masses`` is one explicit kg value per
    bead, written through PhysX exactly the way dump_bin_bigbin does it
    (``entity.find_component_by_type(PhysxRigidDynamicComponent).mass = m``).
    Beads are named ``<name_prefix>_<i>``; granular_pour keeps that name set
    to skip beads in check_stable (they may jitter).
    """
    import sapien

    pts = positions_world
    beads = []
    for i, p in enumerate(pts):
        entity = sapien.Entity()
        entity.set_name(f"{name_prefix}_{i}")
        material = getattr(scene, "default_physical_material", None)
        if material is None:
            material = scene.create_physical_material(0.5, 0.5, 0.0)
        rigid = sapien.physx.PhysxRigidDynamicComponent()
        rigid.attach(sapien.physx.PhysxCollisionShapeSphere(radius=float(radius), material=material))
        if sleep_threshold is not None:
            # PhysX default 0.005 (mass-normalized KE): bodies slower than ~0.1 m/s
            # fall asleep, so a slowly tilting pile can stay frozen as one block
            rigid.set_sleep_threshold(float(sleep_threshold))
        entity.add_component(rigid)
        try:  # query, not construction: get_system raises when absent
            scene.get_system("render")
            has_render = True
        except RuntimeError:
            has_render = False
        if has_render:
            visual = sapien.render.RenderMaterial(base_color=[float(color[0]), float(color[1]), float(color[2]), 1.0])
            render = sapien.render.RenderBodyComponent()
            render.attach(sapien.render.RenderShapeSphere(float(radius), visual))
            entity.add_component(render)
        entity.set_pose(sapien.Pose([float(p[0]), float(p[1]), float(p[2])], [1, 0, 0, 0]))
        scene.add_entity(entity)
        entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = float(masses[i])
        beads.append(entity)
    return beads


# ---------------------------------------------------------------------------
# T2 acceptance-fix helpers (p4-accept shard 2): container-region radius, zone
# keepaway predicate, pour-height floor. Pure numpy / no sapien.
# ---------------------------------------------------------------------------


def container_region_radius(geom, pad=0.002):
    """Registration ``rim_radius`` for a bowl's container region.

    The region polygon (a 16-gon) is the container's FOOTPRINT column, not the
    rim opening: 002_bowl's interior belly is up to 5.3 mm wider than the rim
    opening (bowl 3: 0.0671 vs 0.0618 m), so resting beads press against the
    belly wall OUTSIDE a rim-radius polygon, and the T-softmin over 60 beads
    pulls the joint particles_inside margin negative even when every bead is
    physically inside (p4-accept t=0 failures, softmin -1.6..-5.1 mm). The
    outer AABB radius + ``pad`` contains bead centres with 1.8-2.3 cm of
    margin while a bead on the table just clear of the bowl's outer wall
    still reads outside. Semantics: a bead is "inside" while it is anywhere
    in the column above the bowl's footprint (the PARTICLES_INSIDE margin is
    2D, margins.py m_particles_inside).
    """
    return float(geom["outer_radius"]) + float(pad)


def rect_support_radius(size_xy, dir_xy):
    """Half-extent of an axis-aligned rect (full ``size_xy``) along ``dir_xy``.

    Exact support function of the mat footprint rectangle; used for zone
    keepaways (the centre-distance + half-x proxy left < 5 mm edge gaps on
    ~30% of draws — a resting bead then reads inside the zone polygon at
    t=0, the p4-accept particles_outside failure).
    """
    import numpy as np

    d = np.asarray(dir_xy, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return 0.5 * float(max(size_xy[0], size_xy[1]))
    d = d / n
    return 0.5 * (abs(float(d[0])) * float(size_xy[0]) + abs(float(d[1])) * float(size_xy[1]))


def zone_clear_ok(zxy, size_xy, bowls, gap):
    """True when the mat at ``zxy`` keeps a ``gap``-metre edge clearance.

    ``bowls``: iterable of (center_xy, radius) — the exclusion discs (bowl
    outer radius for resting-bead reach). The mat's exact support radius in
    each bowl's direction (rect_support_radius) plus the disc radius plus
    ``gap`` bounds the centre distance.
    """
    import numpy as np

    z = np.asarray(zxy, dtype=np.float64).reshape(2)
    for center, radius in bowls:
        c = np.asarray(center, dtype=np.float64).reshape(2)
        d = c - z
        if float(np.linalg.norm(d)) < rect_support_radius(size_xy, -d) + float(radius) + float(gap):
            return False
    return True


def min_pour_height(src_geom, clearance=0.015):
    """Minimum effective pour height (rim pivot above the target rim plane).

    The pivot pins the source bowl's rim CENTRE above the target rim. Tilting
    to theta dips the source rim ring's lowest point to pivot - r_out*sin(theta);
    at theta = 90 deg that is pivot - r_out, hanging INSIDE the target mouth —
    two 002_bowl rims of near-equal radius cannot interlink, so with
    pour_height < r_out the bowls collide at ~80-95 deg of tilt and the source
    bowl is knocked out of the grasp (p4-accept ur5 plan #12: a 0.51 m
    unreachable tilt target, episode abort). r_out + ``clearance`` keeps the
    ring's lowest point ``clearance`` above the target rim plane at the worst
    tilt angle. 0.0677 + 0.015 = 0.083 for the largest bowl — inside the
    yaml knob range [0.05, 0.20].
    """
    return float(src_geom["outer_radius"]) + float(clearance)
