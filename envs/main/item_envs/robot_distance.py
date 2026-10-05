"""Multi-item robot distances and contact forces (PLAN v3 §2.2).

Generalizes CEHJ's ``main/envs/distance.py`` from the single ``env.obstacle``
to any list of ItemSpecs WITHOUT modifying it: the collision-sphere, OBB,
held-detection, and contact-force helpers are imported and reused verbatim.

Semantics: an item held by a gripper counts as part of the robot body for
every OTHER item's distance (its OBB joins the payload set); its own distance
is +inf while it is held. Forces follow distance.py's
``_robot_obstacle_contact_force`` pattern per item: max over contact points of
|impulse| / dt for contacts between the item and anything except the table,
skipping contacts internal to the item itself (e.g. bead-bead inside one
particle_set).

Held detection is CEHJ's ``detect_held_by_arm`` (the contact/latch/near
machinery over the host task's named actors), supplemented by a direct
contact scan over the item list (``_contact_held_item_indices``) so items of
tasks that are not in CEHJ's TASK_SPECS (T2/T3) are still detected when a
closed gripper touches them. One consequence vs CEHJ's single-obstacle
semantics: a closed gripper touching the queried item marks it held (+inf),
where distance_info would still report a small d_system for env.obstacle.
That is the intended v3 rule, not a bug.

Wrapper-pinning assumption: ALL entity matching in this module is keyed on
``entity.per_scene_id`` — the C++-assigned int that stays valid across
pybind11 wrappers. Keying on ``id()`` is broken here because sapien hands out
temporary python wrappers that are garbage-collected: CEHJ's own
``_entity_side_lookup`` / ``_detect_held_from_contacts`` key on wrapper
``id()``, so contact-time entities never match them and the held signal is
lost (observed on both ur5-wsg and franka-panda; runner.py's held detection
hit the same bug and fixed it the same way). CEHJ's id()-based
``_entity_ids`` is therefore used only inside CEHJ's own
``detect_held_by_arm`` base path, never in this module. The wrapper identity
that CEHJ's base path DOES retain (the latched actor in
``env._cehj_held_latch``) is safe because item/task actors are pinned alive
for the whole episode — item wrappers live in ``task.item_meta`` and task
actors are task attributes, so their wrappers are never garbage-collected
mid-episode; only per-call temporaries (``get_links()``, contact bodies)
are. The robot side map is cached on ``env._item_side_psids``; link psids
are fixed for the scene's lifetime, so the cache pins per_scene_id ints,
not wrappers.
"""

from __future__ import annotations

import numpy as np

from ._bootstrap import ensure_main_path

ensure_main_path()

from . import item_shapes
from .items import ItemSpec, entity_of

__all__ = ["robot_item_distances", "robot_item_forces", "HeldTracker", "DISTANCE_MODE"]

# "shape" (default, 2026-10-05): items and the held payload are their PhysX
# collision shapes, exact signed distances (item_shapes.py). "obb": the
# previous box model — every item its OBB, the payload 8 box corners
# (CEHJ distance_info parity; a held bowl touching a bottle read +0.8 cm and
# read -0.6 cm ~5 cm away, user report 2026-10-05).
DISTANCE_MODE = __import__("os").environ.get("ITEM_DISTANCE", "shape")


def _cehj():
    """CEHJ distance module + collision-sphere loader (lazy: needs main.envs)."""
    from main.envs import distance as cehj_distance
    from main.envs.obstacle import load_collision_spheres

    return cehj_distance, load_collision_spheres


_SPHERES_DIR = __import__("pathlib").Path(__file__).resolve().parent / "configs" / "collision_spheres"
_SPHERE_CACHE: dict = {}


def _item_collision_spheres(embodiment: str, cehj_loader) -> dict:
    """Collision spheres for the item distance: our own file when one exists
    for the embodiment (configs/collision_spheres/<emb>.yml — ur5-wsg: the
    RoboTwin file models a Robotiq gripper, the WSG had no spheres), else
    CEHJ's loader (RoboTwin's file)."""
    if embodiment not in _SPHERE_CACHE:
        path = _SPHERES_DIR / f"{embodiment}.yml"
        if path.exists():
            import yaml

            _SPHERE_CACHE[embodiment] = yaml.safe_load(path.read_text())["collision_spheres"]
        else:
            _SPHERE_CACHE[embodiment] = cehj_loader(embodiment)
    return _SPHERE_CACHE[embodiment]


def _as_specs(items) -> list:
    specs = list(items.values()) if hasattr(items, "values") else list(items)
    return specs


def _spec_obb(spec: ItemSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(origin, rot, half) of the item OBB in the world frame.

    Same construction as distance.py ``_obb_from_actor``: origin = pose.p +
    rot @ center_local, with center/half from the ItemSpec.
    """
    ent = entity_of(spec)
    pose = ent.get_pose()
    rot = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)[:3, :3]
    origin = np.asarray(pose.p, dtype=np.float64).reshape(3) + rot @ spec.obb_center_local
    return origin, rot, spec.half_extents


def _item_entity_psids(spec: ItemSpec) -> set[int]:
    """per_scene_ids of every entity the item owns (entity + particles)."""
    entities = [entity_of(spec)]
    if spec.kind == "particle_set":
        entities += [entity_of(p) for p in (spec.particles or [])]
    return {int(e.per_scene_id) for e in entities}


def _robot_side_psids(env) -> dict[int, str]:
    """per_scene_id -> "left"/"right" for every robot link entity (cached on env).

    Same construction as HeldTracker.contact_held_sides' side map. Keyed on
    per_scene_id
    because distance.py's ``_entity_side_lookup`` keys on id() of temporary
    link wrappers that are garbage-collected before contact time.
    """
    cache = getattr(env, "_item_side_psids", None)
    if cache is None:
        cache = {}
        for side, art in (("left", env.robot.left_entity), ("right", env.robot.right_entity)):
            for link in art.get_links():
                ent = getattr(link, "entity", None)
                if ent is not None:
                    cache[int(ent.per_scene_id)] = side
        env._item_side_psids = cache
    return cache


PINCH_MIN_LINKS = 2  # distinct gripper links touching an item for it to count as held


def _closed_gripper_contacts(env, key_by_psid: dict, entity_by_key: dict, cehj) -> dict:
    """side -> key of the item a CLOSED gripper is holding this physics step.

    A held item is PINCHED: touched by at least ``PINCH_MIN_LINKS`` distinct
    gripper links (both fingers squeeze it). Among several pinched items the
    one nearest that arm's TCP wins. Single-link contacts never start a hold
    — "first contact wins" (the previous rule) latched a can brushing the
    outside of a finger instead of the bread pinched between them, and later
    the basket the carried bread touched (diag jobs 22636048 / 22636122,
    selective_avoid/place_bread_basket/franka seeds 1, 3): the touched item
    then read avoid = +inf while the gripper was on it, and the real payload
    dropped out of the distance. HeldTracker keeps an EXISTING latch while
    the arm stays closed and the item near (CEHJ ``_hold_metrics``).
    """
    side_psids = _robot_side_psids(env)
    touch: dict[str, dict] = {}  # side -> key -> set of gripper link psids
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:
        return {}  # transient (e.g. before the first step), distance.py's guard
    for contact in contacts:
        bodies = contact.bodies
        for i in range(2):
            key = key_by_psid.get(int(bodies[1 - i].entity.per_scene_id))
            if key is None:
                continue
            grip_ent = bodies[i].entity
            if not cehj._is_gripper_entity(env, grip_ent):
                continue
            side = side_psids.get(int(grip_ent.per_scene_id))
            if side is None:
                continue
            touch.setdefault(side, {}).setdefault(key, set()).add(int(grip_ent.per_scene_id))
    out: dict = {}
    for side, cands in touch.items():
        if not cehj._arm_is_holding(env, side):
            continue
        cands = {k: links for k, links in cands.items() if len(links) >= PINCH_MIN_LINKS}
        if not cands:
            continue
        if len(cands) == 1:
            out[side] = next(iter(cands))
            continue

        def _rank(key, _side=side, _cands=cands):
            try:
                d = float(cehj._hold_metrics(env, entity_by_key[key], _side)["origin_dist_tcp"])
            except Exception:
                d = float("inf")
            return (-len(_cands[key]), d if d == d else float("inf"), str(key))

        out[side] = min(cands, key=_rank)
    return out


def _contact_held_item_indices(env, specs: list, cehj) -> set[int]:
    """Indices of items HELD by a closed gripper this physics step (one per
    arm, chosen by ``_closed_gripper_contacts``: most gripper links in
    contact, then nearest to the TCP). Item side matched by psid, gripper
    side by name + psid side map, arm must read closed."""
    spec_by_psid: dict[int, int] = {}
    for i, spec in enumerate(specs):
        for psid in _item_entity_psids(spec):
            spec_by_psid.setdefault(psid, i)
    if not spec_by_psid:
        return set()
    entity_by_key = {i: spec.entity for i, spec in enumerate(specs)}
    return set(_closed_gripper_contacts(env, spec_by_psid, entity_by_key, cehj).values())


class HeldTracker:
    """Per-arm held-item latch (runner._held_indices' core, shared with
    margin_hook.ItemMarginHFn — Phase 5).

    CEHJ latch semantics: a closed-gripper contact STARTS the latch; it
    holds while the arm stays closed and the item remains near the
    TCP/fingers (``_hold_metrics``), and releases otherwise. Raises on
    failure — callers guard (the runner warns once into
    meta.diag["held_error"]).
    """

    def __init__(self, items):
        self._items = _as_specs(items)
        self._item_psids = {
            int(entity_of(s).per_scene_id): s.item_id for s in self._items
        }
        self._latch: dict[str, str | None] = {"left": None, "right": None}

    @property
    def latched(self) -> dict[str, str | None]:
        """side -> item_id currently latched (post held_indices), None if free."""
        return dict(self._latch)

    def contact_held_sides(self, env, cehj) -> dict[str, str]:
        """side -> item_id held by a CLOSED gripper this step (same selection
        as ``_contact_held_item_indices``: most gripper links in contact,
        then nearest to the TCP)."""
        entity_by_key = {s.item_id: s.entity for s in self._items}
        return _closed_gripper_contacts(env, self._item_psids, entity_by_key, cehj)

    def held_indices(self, env) -> np.ndarray:
        """[2] item index held by left/right arm, -1 if none (contact latch)."""
        held = np.array([-1, -1], dtype=np.int64)
        if not self._item_psids:
            return held
        cehj, _load_spheres = _cehj()
        found = self.contact_held_sides(env, cehj)
        index = {s.item_id: k for k, s in enumerate(self._items)}
        by_id = {s.item_id: s for s in self._items}
        for side, slot in (("left", 0), ("right", 1)):
            item_id = found.get(side)
            if item_id is None:
                prev = self._latch.get(side)
                if prev is not None and cehj._arm_is_holding(env, side):
                    if cehj._hold_metrics(env, by_id[prev].entity, side)["near"]:
                        item_id = prev
            self._latch[side] = item_id
            if item_id is not None:
                held[slot] = index[item_id]
        return held


def _spec_obb_distance(cehj, held_obb: tuple, origin, rot, half) -> float:
    """OBB-to-OBB signed distance between a held item's spec OBB and a query OBB.

    Replicates distance.py ``_actor_obb_distance``'s 16-corner min exactly
    (same point_obb_signed_distance calls, same loop order), so a held item
    contributes bit-identically to a held task actor queried through
    ``_actor_obb_distance`` whenever the spec OBB was built from that actor's
    ``_obb_from_actor`` values.
    """
    h_o, h_r, h_h = held_obb
    d_min = float("inf")
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * h_h[0], sy * h_h[1], sz * h_h[2]])
                d_min = min(d_min, cehj.point_obb_signed_distance(h_o + h_r @ local, origin, rot, half))
                local_b = np.array([sx * half[0], sy * half[1], sz * half[2]])
                d_min = min(d_min, cehj.point_obb_signed_distance(origin + rot @ local_b, h_o, h_r, h_h))
    return d_min


def _held_info(env, specs: list, cehj) -> tuple[set[int], list[tuple]]:
    """(held item indices, payload refs) for the current scene state.

    Base: CEHJ ``detect_held_by_arm`` (latch semantics over task-named
    actors). A detected actor whose psid belongs to a listed item marks that
    item held and contributes the item's OWN spec OBB as payload
    (``("spec", idx)`` — numerically identical to ``_actor_obb_distance`` on
    the actor when the spec was built from it, so d_system parity is kept); a
    detected actor outside the item list contributes ``("actor", actor)``
    and is distance-checked through CEHJ's ``_actor_obb_distance`` verbatim.
    Supplement: the psid-keyed contact scan adds any further item touched by
    a closed gripper. Dedup is entirely by per_scene_id — no id() identity.
    """
    spec_by_psid: dict[int, int] = {}
    for i, spec in enumerate(specs):
        for psid in _item_entity_psids(spec):
            spec_by_psid.setdefault(psid, i)
    held_idx: set[int] = set()
    payloads: list[tuple] = []
    seen_psids: set[int] = set()

    detected = cehj.detect_held_by_arm(env)
    for side in ("left", "right"):
        got = detected.get(side)
        if not got:
            continue
        actor = got[0]
        try:
            psids = {int(entity_of(actor).per_scene_id)}
        except Exception:
            psids = set()  # unknown wrapper shape: keep the CEHJ actor path
        idx = spec_by_psid.get(next(iter(psids), None))
        if idx is not None:
            held_idx.add(idx)
            payload = ("spec", idx)
        else:
            payload = ("actor", actor)
        if not psids or seen_psids.isdisjoint(psids):
            seen_psids |= psids
            payloads.append(payload)

    for idx in _contact_held_item_indices(env, specs, cehj):
        if idx in held_idx:
            continue
        held_idx.add(idx)
        psids = _item_entity_psids(specs[idx])
        if seen_psids.isdisjoint(psids):
            seen_psids |= psids
            payloads.append(("spec", idx))
    return held_idx, payloads


_RULE_ROLES = ("candidate", "bystander")  # items the rules protect: held = touched, never exempt
HELD_CONTACT_DEPTH = 0.005  # m: a held rule item's distance is at most -this (it is in the gripper)


def _arm_spheres_world(env, spheres, cehj) -> tuple[np.ndarray, np.ndarray]:
    """(centers [N,3], radii [N]) of both arms' collision spheres, world frame
    (the same sphere set as ``cehj._arm_sphere_distance``)."""
    centers, radii = [], []
    for side in ("left", "right"):
        entity = env.robot.left_entity if side == "left" else env.robot.right_entity
        for c, r, _name in cehj.spheres_with_names(entity, cehj._arm_collision_spheres(env, side, spheres)):
            centers.append(c)
            radii.append(r)
    return np.asarray(centers, dtype=np.float64).reshape(-1, 3), np.asarray(radii, dtype=np.float64)


def _shape(obj, spec: ItemSpec | None = None):
    """ItemShape of an item spec (OBB box when it has no collision shape) or a
    task actor; None for particle sets (their distance stays the OBB one)."""
    if spec is not None:
        if spec.kind == "particle_set":
            return None
        return item_shapes.shape_of(spec, fallback_obb=(spec.obb_center_local, spec.half_extents))
    return item_shapes.shape_of(obj)


def robot_item_distances(env, items) -> np.ndarray:
    """[K] min signed distance from robot links + held payload to each item (m).

    Positive = gap, negative = penetration depth (same convention as CEHJ's
    h). DISTANCE_MODE "shape": arm collision spheres and the held payload's
    collision shapes vs the item's collision shapes, exact (item_shapes);
    "obb": sphere-to-OBB and 16-corner OBB-to-OBB. The task's own held object (role
    task_object / support) gets +inf; a held candidate / bystander keeps its
    real (contact, <= 0) distance. Every held item counts as robot payload
    for the OTHER items. items: iterable of ItemSpec (or mapping id ->
    ItemSpec).
    """
    cehj, load_collision_spheres = _cehj()
    specs = _as_specs(items)
    out = np.full(len(specs), np.inf, dtype=np.float64)
    if not specs:
        return out
    spheres = _item_collision_spheres(env.embodiment, load_collision_spheres)
    held_idx, payloads = _held_info(env, specs, cehj)
    use_shapes = DISTANCE_MODE == "shape"
    if use_shapes:
        centers, radii = _arm_spheres_world(env, spheres, cehj)
    for i, spec in enumerate(specs):
        if i in held_idx and str(getattr(spec, "role", "")) not in _RULE_ROLES:
            continue  # the task's own held object is part of the robot body: own distance +inf
        # a held CANDIDATE / bystander was grabbed by the gripper — e.g. the
        # open fingers closing on the pick object caught a T1 candidate 7 cm
        # from it (audit 2026-10-04): its avoid margin must read the contact
        # (<= 0), not +inf ("safe" while the robot holds it)
        origin, rot, half = _spec_obb(spec)
        # exact shapes only where a rule reads the distance (avoid protects
        # candidates / bystanders); the task's own objects (pick object,
        # support) keep the cheap box value — a diagnostic no rule reads
        shape = _shape(None, spec) if use_shapes and str(getattr(spec, "role", "")) in _RULE_ROLES else None
        if shape is not None:
            best = item_shapes.spheres_min_distance(centers, radii, shape)[0]
        else:
            best = min(
                cehj._arm_sphere_distance(env, "left", origin, rot, half, spheres)[0],
                cehj._arm_sphere_distance(env, "right", origin, rot, half, spheres)[0],
            )
        for kind, ref in payloads:
            if kind == "spec" and ref == i:
                continue  # never its own payload
            try:
                p_shape = None
                if shape is not None:
                    p_shape = _shape(None, specs[ref]) if kind == "spec" else _shape(ref)
                if p_shape is not None:
                    d = item_shapes.shape_distance(p_shape, shape)
                elif kind == "spec":
                    d = _spec_obb_distance(cehj, _spec_obb(specs[ref]), origin, rot, half)
                else:
                    d = cehj._actor_obb_distance(ref, origin, rot, half)
            except (TypeError, AttributeError):
                continue
            if d < best:
                best = d
        if i in held_idx:
            # held = in contact by definition; the collision-sphere model of
            # the fingers does not reach an object clamped between them (a
            # grasped cup read +3.2 cm, sim test): count it as inside the grasp
            best = min(best, -HELD_CONTACT_DEPTH)
        out[i] = float(best)
    return out


def robot_item_forces(env, items) -> np.ndarray:
    """[K] contact force (N) on each item, this physics step.

    distance.py ``_robot_obstacle_contact_force`` generalized to K items: max
    over the item's contact points of |impulse| / dt, for contacts with
    anything except the table; contacts between two bodies of the SAME item
    (particle-particle) are skipped, contacts between two DIFFERENT items
    count for both. Matching is by ``entity.per_scene_id`` (recon.md Q7).
    """
    specs = _as_specs(items)
    out = np.zeros(len(specs), dtype=np.float64)
    if not specs:
        return out
    owner: dict[int, int] = {}
    for i, spec in enumerate(specs):
        entities = [entity_of(spec)]
        if spec.kind == "particle_set":
            entities += [entity_of(p) for p in (spec.particles or [])]
        for ent in entities:
            owner[int(ent.per_scene_id)] = i
    scene = env.task.scene
    try:
        contacts = scene.get_contacts()
    except Exception:
        return out
    dt = float(getattr(scene, "timestep", 1.0 / 250.0))
    for contact in contacts:
        bodies = contact.bodies
        ents = [bodies[0].entity, bodies[1].entity]
        idx = [owner.get(int(e.per_scene_id)) for e in ents]
        affected: set[int] = set()
        for a in range(2):
            i, j = idx[a], idx[1 - a]
            if i is None or i == j:
                continue  # no item on this side, or same-item internal contact
            if j is None and "table" in (getattr(ents[1 - a], "name", "") or "").lower():
                continue  # the item resting on the table is not a robot contact
            affected.add(i)
        if not affected:
            continue
        f_max = 0.0
        for p in contact.points:
            f = float(np.linalg.norm(p.impulse)) / dt
            if f > f_max:
                f_max = f
        for i in affected:
            if f_max > out[i]:
                out[i] = f_max
    return out
