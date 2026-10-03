"""Non-sim tests for T3 push_zone (PLAN v3 §7 T3).

Two contracts, no SAPIEN:

1. ``configs/tasks/T3_push_zone.yaml`` + objects.yaml + splits.yaml against a
   synthetic push_zone scene (two push candidates, two bystanders, two zones):
   the rule table has one ``inside`` and one ``outside``-per-zone rule per push
   candidate and one ``upright`` + one ``stay`` rule per bystander, rule_ids
   are stable across enumerations, and phrases never contain digits.

2. Source-grep contract: the module constants in
   ``robotwin_tasks/push_zone.py`` (parsed with ast.literal_eval off single-
   line assignments) mirror the task yaml — the task file cannot import
   ``main.*``, so the yaml is the CEHJ-side copy of the same static config.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from main.item_envs.items import ItemSpec
from main.item_envs.regions import Region
from main.item_envs.rules import enumerate_rules, resolve_splits

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "main" / "item_envs" / "configs"
TASK_SRC = ROOT / "robotwin_tasks" / "push_zone.py"

REQUIRED_KEYS = [
    "task", "robotwin_task", "embodiments", "n_push", "push_objects", "push", "goal",
    "elements", "risk", "rules", "embodiment_overrides", "risk_collect",
]

EXPECTED_RULE_IDS = [
    "inside|push_0|table_top",
    "inside|push_1|table_top",
    "outside|push_0|zone_blue",
    "outside|push_0|zone_red",
    "outside|push_1|zone_blue",
    "outside|push_1|zone_red",
    "upright|bystander_0|-",
    "upright|bystander_1|-",
    "stay|bystander_0|-",
    "stay|bystander_1|-",
]


def _load(name):
    with open(CONFIGS / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _task_cfg():
    return _load("tasks/T3_push_zone.yaml")


def _const(name: str):
    """ast.literal_eval of a single-line module assignment in push_zone.py."""
    src = TASK_SRC.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", src, re.M)
    assert m, f"constant {name} not found as a single-line assignment in {TASK_SRC.name}"
    rhs = re.sub(r"\s+#.*$", "", m.group(1))  # strip trailing comment
    return ast.literal_eval(rhs)


def _norm(v):
    """Tuples <-> lists and ints -> floats for yaml/python comparison."""
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in v.items()}
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return v


def mock_push_scene():
    """push_0 (red prim box) + push_1 (tea box) candidates, two bystanders,
    a candidate outside the push group (decoy), table_top + two zones."""
    items = {
        "push_0": ItemSpec(item_id="push_0", entity=object(), category="box", color="red", role="candidate", group="push"),
        "push_1": ItemSpec(item_id="push_1", entity=object(), category="tea_box", color=None, role="candidate", group="push"),
        "bystander_0": ItemSpec(item_id="bystander_0", entity=object(), category="bottle", color=None, role="bystander"),
        "bystander_1": ItemSpec(item_id="bystander_1", entity=object(), category="cylinder", color="yellow", role="bystander"),
        "decoy_0": ItemSpec(item_id="decoy_0", entity=object(), category="box", color="green", role="candidate"),
    }
    table_poly = np.array([[-0.6, -0.35], [0.6, -0.35], [0.6, 0.35], [-0.6, 0.35]])
    zone_poly = np.array([[-0.07, -0.06], [0.07, -0.06], [0.07, 0.06], [-0.07, 0.06]])
    regions = {
        "table_top": Region(
            region_id="table_top", kind="surface", frame_fn=lambda: np.eye(4),
            polygon=table_poly, z=0.74, z_band=0.01, nameable=True,
        ),
        "zone_red": Region(
            region_id="zone_red", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=zone_poly, z=0.0, z_band=0.01, nameable=True, color="red",
        ),
        "zone_blue": Region(
            region_id="zone_blue", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=zone_poly, z=0.0, z_band=0.01, nameable=True, color="blue",
        ),
    }
    return items, regions


def _rules():
    items, regions = mock_push_scene()
    return enumerate_rules(
        items,
        regions,
        _task_cfg(),
        objects_cfg=_load("objects.yaml"),
        splits_cfg=resolve_splits(_load("splits.yaml"), "push_zone"),
    )


# --------------------------------------------------------------------- yaml ----


def test_t3_yaml_required_keys():
    cfg = _task_cfg()
    for key in REQUIRED_KEYS:
        assert key in cfg, f"missing key {key!r} in T3_push_zone.yaml"
    assert cfg["task"] == "push_zone"
    assert cfg["robotwin_task"] == "push_zone"
    assert cfg["embodiments"] == ["ur5-wsg", "franka-panda"]
    assert cfg["risk_collect"] is None
    risk = cfg["risk"]
    assert risk["time_scale"] == {"dist": "choice", "values": [0.5, 0.75, 1.0, 1.15, 1.3]}
    assert risk["safe_time_scale"] == 0.6
    assert risk["task"]["overshoot"] == {"dist": "uniform", "range": [0.0, 0.08]}
    assert risk["task"]["lateral_offset"] == {"dist": "uniform", "range": [-0.02, 0.02]}
    # play_once's knob defaults sit at the low end of the sampled ranges
    assert risk["task"]["overshoot"]["range"][0] <= 0.0 <= risk["task"]["overshoot"]["range"][1]
    assert risk["task"]["lateral_offset"]["range"][0] <= 0.0 <= risk["task"]["lateral_offset"]["range"][1]
    # the four plan rules entries, in order
    entries = cfg["rules"]
    assert [e["consequence"] for e in entries] == ["inside", "outside", "upright", "stay"]
    assert entries[0]["items"] == {"role": "candidate", "group": "push"}
    assert entries[0]["regions"] == ["table_top"]
    assert entries[1]["items"] == {"role": "candidate", "group": "push"}
    assert entries[1]["regions"] == {"kind": "zone"}
    assert entries[2]["items"] == {"role": "bystander"}
    assert entries[2]["theta_max_deg"] == 45
    assert entries[3]["items"] == {"role": "bystander"}
    assert entries[3]["d_tol"] == 0.03
    assert entries[3]["theta_tol_deg"] == 20


# ------------------------------------------------------------------- rules ----


def test_push_rule_table_order_and_ids():
    assert [r.rule_id for r in _rules()] == EXPECTED_RULE_IDS


def test_push_rule_ids_stable():
    first = [r.rule_id for r in _rules()]
    second = [r.rule_id for r in _rules()]
    assert first == second


def test_push_rule_semantics():
    rules = _rules()
    by_id = {r.rule_id: r for r in rules}
    for rid, rule in by_id.items():
        consequence, item_part, region_part = rid.split("|")
        assert rule.consequence.value == consequence
        assert item_part != "decoy_0"  # the non-push-group candidate gets no rules
        assert rule.phrases, f"{rid} rendered no phrases"
        for phrase in rule.phrases:
            assert not any(c.isdigit() for c in phrase), phrase
        assert rule.split in ("train", "heldout_item", "heldout_region", "heldout_phrase")
    # inside/outside only on push candidates; upright/stay only on bystanders
    for rid in EXPECTED_RULE_IDS:
        consequence, item_part, _ = rid.split("|")
        if consequence in ("inside", "outside"):
            assert item_part in ("push_0", "push_1")
        else:
            assert item_part in ("bystander_0", "bystander_1")
    assert by_id["upright|bystander_0|-"].params["theta_max"] == pytest.approx(np.deg2rad(45))
    stay = by_id["stay|bystander_1|-"]
    assert stay.params["d_tol"] == pytest.approx(0.03)
    assert stay.params["theta_tol"] == pytest.approx(np.deg2rad(20))
    # colored items name their color in the naming metadata
    assert by_id["inside|push_0|table_top"].params["color"] == "red"
    assert by_id["upright|bystander_1|-"].params["color"] == "yellow"
    # splits.yaml: tea_box is a heldout item category for push_zone
    assert by_id["inside|push_1|table_top"].split == "heldout_item"
    assert by_id["inside|push_0|table_top"].split == "train"


# ----------------------------------------------------------------- contract ----


def test_module_constants_mirror_yaml():
    cfg = _task_cfg()
    assert _norm(_const("N_PUSH")) == _norm(cfg["n_push"])
    assert _const("OVERSHOOT_MAX") == pytest.approx(cfg["risk"]["task"]["overshoot"]["range"][1])
    assert _norm(_const("PUSH_OBJECT_POOL")) == _norm(cfg["push_objects"]["pool"])
    assert _norm(_const("PRIM_SIDE_XY")) == _norm(cfg["push_objects"]["side"])
    assert _norm(_const("PRIM_HEIGHT")) == _norm(cfg["push_objects"]["height"])
    assert _norm(_const("START_ABS_X")) == _norm(cfg["push_objects"]["start"]["abs_x"])
    assert _norm(_const("START_Y")) == _norm(cfg["push_objects"]["start"]["y"])
    mesh_ids = _const("MESH_MODEL_IDS")
    assert _norm({k: mesh_ids[k] for k in cfg["push_objects"]["ids"]}) == _norm(cfg["push_objects"]["ids"])
    assert _norm(_const("PUSH_SEG_LEN")) == _norm(cfg["push"]["seg_len"])
    assert _norm(_const("PUSH_LEN")) == _norm(cfg["push"]["len"])
    assert _norm(_const("PATH_LEN_PER_EXTRA")) == _norm(cfg["push"]["len_per_extra"])
    assert _norm(_const("GOAL_ABS_X")) == _norm(cfg["goal"]["abs_x"])
    assert _norm(_const("GOAL_Y")) == _norm(cfg["goal"]["y"])
    assert _norm(_const("GOAL_PAD")) == _norm(cfg["goal"]["pad"])
    el = cfg["elements"]
    assert _norm(_const("MODES")) == _norm(el["modes"])
    assert _norm(_const("N_ELEMENTS")) == _norm(el["n"])
    assert _norm(_const("N_ON_PATH")) == _norm(el["on_path"])
    assert _norm(_const("ELEM_GAP")) == _norm(el["gap"])
    assert _norm(_const("ZONE_SIZE")) == _norm(el["zone_size"])
    assert _norm(_const("ON_PATH_LAT")) == _norm(el["on_path_lat"])
    assert _norm(_const("OFF_PATH_OFFSET")) == _norm(el["off_path_offset"])
    assert _norm(_const("OBSTACLE_POOL")) == _norm(el["obstacle_pool"])
    assert _norm({k: mesh_ids[k] for k in el["obstacle_ids"]}) == _norm(el["obstacle_ids"])
    assert _norm(_const("CYLINDER_RADIUS")) == _norm(el["cylinder"]["radius"])
    assert _norm(_const("CYLINDER_HEIGHT")) == _norm(el["cylinder"]["height"])
    for emb in ("ur5-wsg", "franka-panda"):
        assert _norm(_const("PREPUSH_DIST")) == _norm(cfg["embodiment_overrides"][emb]["prepush_dist"])
        assert _norm(_const("PUSH_HEIGHT_OFFSET")) == _norm(cfg["embodiment_overrides"][emb]["push_height_offset"])
    lat_range = cfg["risk"]["task"]["lateral_offset"]["range"]
    assert _const("LAT_OFFSET_MAX") == pytest.approx(max(abs(v) for v in lat_range))


def test_color_palette_matches_objects_yaml():
    objects_cfg = _load("objects.yaml")
    assert _norm(_const("TRAIN_COLORS")) == _norm(objects_cfg["colors"]["train"])
    for name in _const("TRAIN_COLORS"):
        assert _norm(_const("COLOR_RGB")[name]) == _norm(objects_cfg["color_rgb"][name])


# ------------------------------------------------------- offline layout replay ----
#
# p4-accept regression (GPU shard 2): seeds 5/9 exhausted bystander placement,
# seed 2's 001_bottle bystander tipped in check_stable. The layout path is
# pure numpy — replayable off-GPU by stubbing sapien + RoboTwin's utils,
# loading the REAL task module, and reproducing the exact pre-load global-RNG
# stream:
#   np.random.seed(seed)                     (_base_task.py:58)
#   np.random.uniform(-0.0, 0.0)             (_base_task.py:111 table_z_bias)
#   np.random.randn(3); np.random.uniform()  (camera/camera.py:98-100 head-cam jitter)
# The pre-load stream prefix is unchanged by the cross-table redesign; the
# load_actors draw order it feeds is documented in the task module docstring.
# (The old design was additionally verified against the p4-accept plan lines —
# seed 0 pre-poses/distances to 1.5 mm — and the seed 5/9 RuntimeErrors.)

ROBOTWIN = ROOT.parent.parent / "RoboTwin"
ROBOTWIN_OBJECTS = ROBOTWIN / "assets" / "objects"


class _FakePose:
    def __init__(self, p, q=None):
        self.p = np.asarray(p, dtype=np.float64).reshape(3)
        self.q = list(q) if q is not None else [1, 0, 0, 0]

    def to_transformation_matrix(self):
        w, x, y, z = self.q
        m = np.eye(4)
        m[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        m[:3, 3] = self.p
        return m


class _FakeEntity:
    def __init__(self):
        self._pose = _FakePose([0, 0, 0])

    def set_name(self, n):
        pass

    def set_pose(self, p):
        self._pose = p

    def get_pose(self):
        return self._pose

    def add_component(self, c):
        pass

    def find_component_by_type(self, t):
        return None


class _FakeActor:  # RoboTwin Actor wrapper: .config = model_data json dict
    def __init__(self, entity, config):
        self.actor = entity
        self.config = config

    def get_pose(self):
        return self.actor.get_pose()


class _FakeScene:
    default_physical_material = object()

    def get_system(self, name):
        raise RuntimeError("no render")

    def add_entity(self, e):
        pass


def _model_data(modelname, model_id):
    import json

    jf = ROBOTWIN_OBJECTS / str(modelname) / (
        f"model_data{model_id}.json" if model_id is not None else "model_data.json"
    )
    with open(jf, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_task_module():
    """Import robotwin_tasks/push_zone.py with sapien + RoboTwin utils stubbed."""
    import importlib.util
    import sys
    import types

    sapien = types.ModuleType("sapien")
    sapien.Pose = _FakePose
    sapien.Entity = _FakeEntity
    physx = types.ModuleType("sapien.physx")
    physx.PhysxRigidStaticComponent = type("C", (), {"attach": lambda self, s: None})
    physx.PhysxRigidDynamicComponent = type("D", (), {})
    physx.PhysxCollisionShapeBox = type("S", (), {"__init__": lambda self, **kw: None})
    render = types.ModuleType("sapien.render")
    render.RenderMaterial = type("M", (), {"__init__": lambda self, **kw: None})
    render.RenderBodyComponent = type("B", (), {"attach": lambda self, s: None})
    render.RenderShapeBox = type("R", (), {"__init__": lambda self, *a: None})
    sapien.physx = physx
    sapien.render = render

    def create_box(scene, pose, half_size, color=None, name=None, **kw):
        ent = _FakeEntity()
        ent.set_pose(pose)
        cfg = {"extents": list(half_size), "scale": list(half_size), "center": [0, 0, 0]}
        return _FakeActor(ent, cfg)

    def create_cylinder(scene, pose, radius, half_length, color=None, name=None, **kw):
        ent = _FakeEntity()
        ent.set_pose(pose)
        cfg = {"extents": [radius, radius, half_length], "scale": [radius, radius, half_length], "center": [0, 0, 0]}
        return _FakeActor(ent, cfg)

    def create_actor(scene, pose, modelname, convex=False, is_static=False, model_id=0, **kw):
        ent = _FakeEntity()
        ent.set_pose(pose)
        return _FakeActor(ent, _model_data(modelname, model_id))

    utils = types.ModuleType("pz_replay.utils")
    utils.np = np
    utils.create_box = create_box
    utils.create_cylinder = create_cylinder
    utils.create_actor = create_actor
    base = types.ModuleType("pz_replay._base_task")
    base.Base_Task = type("Base_Task", (), {})
    pkg = types.ModuleType("pz_replay")
    pkg.__path__ = [str(TASK_SRC.parent)]

    saved = {k: sys.modules.get(k) for k in ("sapien", "sapien.physx", "sapien.render")}
    sys.modules.update(
        {
            "sapien": sapien,
            "sapien.physx": physx,
            "sapien.render": render,
            "pz_replay": pkg,
            "pz_replay.utils": utils,
            "pz_replay._base_task": base,
        }
    )
    mods = {}
    for name, fname in (("pz_replay._item_utils", "_item_utils.py"), ("pz_replay.push_zone", "push_zone.py")):
        spec = importlib.util.spec_from_file_location(name, TASK_SRC.parent / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod

    def _restore():
        for key, old in saved.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old

    # the stubs must STAY installed while the replay runs: _item_utils.zone_mat
    # does a function-level `import sapien`, and the real sapien imports on this
    # node (GPU-less) and would cook real PhysX shapes against the fake scene
    return mods["pz_replay.push_zone"], _restore


def _replay_layout(pz, seed):
    task = pz.push_zone()
    task.scene = _FakeScene()
    task.table_z_bias = 0.0
    np.random.seed(seed)
    np.random.uniform(-0.0, 0.0)  # _base_task.py:111
    np.random.randn(3)  # camera.py:98
    np.random.uniform(0.0, 0.0)  # camera.py:100
    task.load_actors()
    return task


def _interior_zone_clearances(task):
    """(corridor_k, t, edge clearance) per zone/corridor pair with an interior foot.

    For each (zone, corridor) pair where the zone center's perpendicular foot
    lands strictly inside the corridor segment: the foot parameter t, and the
    distance from the zone center to the segment minus the corridor
    half-width minus the zone's support radius along the corridor normal
    (sim-test mirror)."""
    out = []
    for zone in task.zones:
        c = np.asarray(zone["xy"], dtype=np.float64).reshape(2)
        hx, hy = zone["size_xy"][0] / 2.0, zone["size_xy"][1] / 2.0
        for k, corr in enumerate(task.layout_info["corridors"]):
            a = np.asarray(corr["a"], dtype=np.float64)
            b = np.asarray(corr["b"], dtype=np.float64)
            ab = b - a
            t = float(((c - a) @ ab) / float(ab @ ab))
            if not (0.01 < t < 0.99):
                continue
            d = ab / float(np.linalg.norm(ab))
            perp = np.array([-d[1], d[0]])
            z_half = float(abs(perp[0]) * hx + abs(perp[1]) * hy)
            dist = float(np.linalg.norm(c - (a + t * ab)))
            out.append((k, t, dist - float(corr["half"]) - z_half))
    return out


@pytest.mark.skipif(not ROBOTWIN_OBJECTS.is_dir(), reason="RoboTwin assets not available")
def test_layout_replay_seeds():
    """300 replayed layouts: one cross-midline push (straight or diagonal),
    reachable gripper path, mode-consistent elements, 1-3 on the path (the
    line crosses them), the rest off, every pair >= ELEM_GAP apart."""
    pz, restore = _load_task_module()
    try:
        seen = {"sides": set(), "modes": set(), "n_on": set()}
        diagonal = 0
        for seed in range(300):
            task = _replay_layout(pz, seed)  # raises RuntimeError on a bad layout
            assert len(task.push_objects) == 1
            rec = task.push_objects[0]
            side = rec["arm"]
            assert side == task.push_side == task.layout_info["push_side"]
            seen["sides"].add(side)
            sgn = -1.0 if side == "left" else 1.0
            p, g = rec["start_xy"], rec["goal_xy"]
            assert sgn * p[0] > 0 and sgn * g[0] < 0, (seed, "does not cross the midline")
            assert pz.PUSH_LEN[0] - 1e-9 <= float(np.linalg.norm(g - p)) <= pz.PUSH_LEN[1] + 1e-9
            diagonal += abs(float(g[1] - p[1])) > 0.10
            # the GRIPPER path (one contact depth behind the box, out to the
            # longest overshoot) stays inside the shared reach
            d = (g - p) / np.linalg.norm(g - p)
            for t in np.linspace(0.0, 1.0, 21):
                q = p + t * (g + d * pz.OVERSHOOT_MAX - p) - d * rec["half_dir"]
                if q[0] * sgn > 0.0:
                    assert q[1] <= pz.OWN_SIDE_Y_MAX + 1e-9, (seed, q)
                else:
                    assert abs(q[0]) <= pz.REACH_X_MAX[0] + pz.REACH_X_MAX[1] * q[1] + 1e-9, (seed, q)

            li = task.layout_info
            seen["modes"].add(li["mode"])
            kinds = {e["kind"] for e in li["elements"]}
            if li["mode"] == "zone":
                assert kinds == {"zone"}
            elif li["mode"] == "obstacle":
                assert kinds == {"obstacle"}
            else:
                assert kinds == {"zone", "obstacle"}
            assert 2 <= len(li["elements"]) + li["elements_dropped"] <= 4
            a = np.asarray(li["corridors"][0]["a"]); b = np.asarray(li["corridors"][0]["b"])
            n_on = 0
            boxes = []
            for e in li["elements"]:
                c = np.asarray(e["xy"]); hx, hy = e["half"]
                box = (c[0] - hx, c[1] - hy, c[0] + hx, c[1] + hy)
                crossed = pz._rect_seg_dist(box, a, b) == 0.0
                assert crossed == e["on_path"], (seed, e)
                n_on += e["on_path"]
                boxes.append(box)
            assert 1 <= n_on <= 3, (seed, n_on)
            seen["n_on"].add(n_on)
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    A, B = boxes[i], boxes[j]
                    gap = max(B[0] - A[2], A[0] - B[2], B[1] - A[3], A[1] - B[3])
                    assert gap >= pz.ELEM_GAP - 1e-6, (seed, i, j, gap)
            # flat mats stay visible: no zone in an excluded (table-colored) color
            zone_ids = [e["id"] for e in li["elements"] if e["kind"] == "zone"]
            assert not any(zid.split("_", 1)[1] in pz.ZONE_EXCLUDE_COLORS for zid in zone_ids), (seed, zone_ids)
            assert len(set(zone_ids)) == len(zone_ids), (seed, zone_ids)
            # the primary rule names an on-path element
            on_ids = {e["id"] for e in li["elements"] if e["on_path"]}
            rid = task.primary_rule_id.split("|")
            assert (rid[2] if rid[0] == "outside" else rid[1]) in on_ids
        assert seen["sides"] == {"left", "right"}
        assert seen["modes"] == {"zone", "obstacle", "mix"}
        assert seen["n_on"] == {1, 2, 3}
        assert diagonal > 30  # straight AND diagonal pushes
    finally:
        restore()


def _mesh_verts(modelname, model_id):
    """Collision-mesh vertices, node transforms baked (sapien's cook), scaled."""
    trimesh = pytest.importorskip("trimesh")

    modeldir = ROBOTWIN_OBJECTS / str(modelname)
    coll = modeldir / "collision" / f"base{model_id}.glb"
    if not coll.exists():
        coll = modeldir / "collision" / f"textured{model_id}.obj"
    if not coll.exists():
        coll = modeldir / f"base{model_id}.glb"
    if not coll.exists():
        coll = modeldir / f"textured{model_id}.obj"
    md = _model_data(modelname, model_id)
    scale = np.asarray(md["scale"], dtype=np.float64).reshape(-1)
    scale = np.full(3, scale[0]) if scale.size == 1 else np.resize(scale, 3)
    loaded = trimesh.load(str(coll), force="scene", process=False)
    dumped = loaded.dump()
    meshes = dumped if isinstance(dumped, (list, tuple)) else [dumped]
    return np.vstack([np.asarray(m.vertices, dtype=np.float64) for m in meshes]) * scale


@pytest.mark.skipif(not ROBOTWIN_OBJECTS.is_dir(), reason="RoboTwin assets not available")
def test_pool_meshes_bottom_origin_and_pads():
    """z-placement + pad contracts for every pool (asset, id) — the T1-class bug
    is a centered origin spawning embedded; the pads must cover the true OBB."""
    pz, restore = _load_task_module()
    try:
        seen = set()
        for asset, ids in pz.MESH_MODEL_IDS.items():
            for model_id in ids:
                assert (asset, model_id) not in seen
                seen.add((asset, model_id))
                v = _mesh_verts(asset, model_id)
                lo, hi = v.min(axis=0), v.max(axis=0)
                # Y-up bottom origin: spawning at z + 0.001 embeds at most ~2 mm
                assert lo[1] > -0.005, (asset, model_id, lo[1])
                assert hi[1] - lo[1] > 0.01
                # json extents match the true mesh (the registered OBB is honest)
                md = _model_data(asset, model_id)
                scale = np.asarray(md["scale"], dtype=np.float64).reshape(-1)
                scale = np.full(3, scale[0]) if scale.size == 1 else np.resize(scale, 3)
                jext = np.abs(np.asarray(md["extents"], dtype=np.float64)) * scale
                assert np.all(np.abs(jext - (hi - lo)) < 0.01), (asset, model_id, jext, hi - lo)
                # placement pad covers the true horizontal half extents
                pad = pz.MESH_PAD_HALF[asset]
                assert pad >= max(hi[0] - lo[0], hi[2] - lo[2]) / 2.0 + 0.002, (asset, model_id, pad)
                # mesh bystanders stay squat (the punt-ring 19-26 cm bottles tipped
                # in check_stable — keep tall meshes out of the bystander pool)
                if asset in pz.OBSTACLE_POOL:
                    assert hi[1] - lo[1] <= 0.12, (asset, model_id, hi[1] - lo[1])
    finally:
        restore()
