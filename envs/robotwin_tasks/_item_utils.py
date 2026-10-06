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
