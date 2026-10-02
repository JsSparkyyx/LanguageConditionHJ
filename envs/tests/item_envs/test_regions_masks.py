"""Non-sim tests for regions constructors and the masks token-grid core.

Region constructors are exercised with duck-typed mock actors exposing
get_pose(); the masks pooling core is pure numpy. No sapien imports here.
"""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs import masks, regions
from main.item_envs.geometry import pose_to_mat
from main.item_envs.items import ItemSpec


# ------------------------------------------------------------------- mocks


class MockPose:
    def __init__(self, p=(0, 0, 0), q=(1, 0, 0, 0)):
        self.p = np.asarray(p, dtype=np.float64).reshape(3)
        self.q = np.asarray(q, dtype=np.float64).reshape(4)

    def to_transformation_matrix(self):
        return pose_to_mat(self.p, self.q)


class MockActor:
    """Duck-typed sapien entity: get_pose() + per_scene_id (+ optional attrs)."""

    def __init__(self, p=(0, 0, 0), q=(1, 0, 0, 0), per_scene_id=1, **attrs):
        self._pose = MockPose(p, q)
        self.per_scene_id = per_scene_id
        for key, val in attrs.items():
            setattr(self, key, val)

    def get_pose(self):
        return self._pose

    def set_pose(self, p, q=(1, 0, 0, 0)):
        self._pose = MockPose(p, q)


class MockTask:
    def __init__(self, xy_bias=(0.0, 0.0), z_bias=0.0, table=None):
        self.table_xy_bias = list(xy_bias)
        self.table_z_bias = float(z_bias)
        self.table = table


def signed_area(poly) -> float:
    """0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i); positive iff CCW."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# --------------------------------------------------------------- table_region


def test_table_region_geometry():
    table = MockActor(per_scene_id=7)
    task = MockTask(xy_bias=(0.05, -0.02), z_bias=-0.03, table=table)
    region = regions.table_region(task)
    assert region.region_id == "table_top"
    assert region.kind == "surface"
    assert region.nameable is True
    assert region.z_band == pytest.approx(0.01)
    assert region.z == pytest.approx(0.74 - 0.03)  # 0.74 + table_z_bias
    assert region.entity is table
    poly = region.polygon
    assert signed_area(poly) > 0  # CCW
    xs, ys = poly[:, 0], poly[:, 1]
    assert (xs.max() - xs.min()) == pytest.approx(1.2)
    assert (ys.max() - ys.min()) == pytest.approx(0.7)
    assert xs.mean() == pytest.approx(0.0) and ys.mean() == pytest.approx(0.0)
    T = region.world_T()
    assert T[:3, 3] == pytest.approx([0.05, -0.02, 0.0])
    assert np.allclose(T[:3, :3], np.eye(3))


def test_table_region_defaults():
    region = regions.table_region(MockTask())
    assert region.z == pytest.approx(0.74)
    assert np.allclose(region.world_T()[:3, 3], [0, 0, 0])


# ---------------------------------------------------------------- zone_region


def test_zone_region_footprint_and_follow():
    mat = MockActor(p=(0.1, 0.2, 0.74), per_scene_id=3, size_xy=(0.12, 0.08))
    region = regions.zone_region(mat, "red")
    assert region.region_id == "zone_red"
    assert region.kind == "zone"
    assert region.nameable is True
    assert region.color == "red"
    assert region.entity is mat
    poly = region.polygon
    assert signed_area(poly) > 0
    xs, ys = poly[:, 0], poly[:, 1]
    assert (xs.max() - xs.min()) == pytest.approx(0.12)
    assert (ys.max() - ys.min()) == pytest.approx(0.08)
    # frame follows the mat every tick
    assert np.allclose(region.world_T()[:3, 3], [0.1, 0.2, 0.74])
    mat.set_pose((-0.3, 0.0, 0.75))
    assert np.allclose(region.world_T()[:3, 3], [-0.3, 0.0, 0.75])


def test_zone_region_explicit_id_and_no_color():
    mat = MockActor(size_xy=(0.1, 0.1))
    assert regions.zone_region(mat, None).region_id == "zone"
    assert regions.zone_region(mat, "blue", region_id="zone_custom").region_id == "zone_custom"


# ---------------------------------------------------------- container_region


def test_container_region_up_frame_and_follow():
    # actor rotated so its local +y (RoboTwin glb up convention) points along
    # world +x: quat wxyz for -90 deg about world z
    q = (np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4))
    bowl = MockActor(p=(0.2, 0.1, 0.8), q=q, per_scene_id=5)
    region = regions.container_region(bowl, 0.055, "bowl_src_interior")
    assert region.region_id == "bowl_src_interior"
    assert region.kind == "container"
    assert region.nameable is True
    assert region.entity is bowl
    poly = region.polygon
    assert poly.shape == (16, 2)
    assert signed_area(poly) > 0
    assert np.allclose(np.linalg.norm(poly, axis=1), 0.055)
    # region +z in world == actor rotation @ up_local ([0,1,0] default)
    world_T = region.world_T()
    up_world = world_T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    R_actor = pose_to_mat(np.zeros(3), q)[:3, :3]
    assert np.allclose(up_world, R_actor @ np.array([0.0, 1.0, 0.0]), atol=1e-12)
    # frame follows the actor
    bowl.set_pose((0.0, 0.0, 0.9), q)
    assert np.allclose(region.world_T()[:3, 3], [0.0, 0.0, 0.9])


def test_container_region_custom_up_and_rim_height():
    actor = MockActor(p=(0, 0, 0.5))  # identity rotation
    region = regions.container_region(actor, 0.04, "target_interior", up_axis_local=[0, 0, 1], rim_height=0.06)
    T = region.world_T()
    assert np.allclose(T[:3, :3] @ np.array([0, 0, 1.0]), [0, 0, 1.0])
    assert np.allclose(T[:3, 3], [0, 0, 0.56])  # rim plane origin along the up axis
    assert region.z == pytest.approx(0.0)  # rim plane is region z = 0


# ---------------------------------------------------------------- token grid


def test_token_grid_layout():
    scales, n_tok = masks.token_grid()
    assert scales == [(16, 16, 20), (32, 8, 10)]  # (stride, rows, cols)
    assert n_tok == 320 + 80 == 400


def test_token_index_order():
    # stride-16 scale first, row-major, then stride-32 after its 320 tokens
    assert masks.token_index(16, 0, 0) == 0
    assert masks.token_index(16, 2, 4) == 2 * 20 + 4
    assert masks.token_index(16, 15, 19) == 319
    assert masks.token_index(32, 0, 0) == 320
    assert masks.token_index(32, 1, 2) == 320 + 1 * 10 + 2
    assert masks.token_index(32, 7, 9) == 399
    with pytest.raises(IndexError):
        masks.token_index(16, 16, 0)
    with pytest.raises(ValueError):
        masks.token_index(8, 0, 0)


def test_pool_mask_full_and_empty():
    full = np.ones((256, 320), dtype=bool)
    out = masks.pool_mask_to_tokens(full)
    assert out.shape == (400,)
    assert out.dtype == np.uint8
    assert np.all(out == 255)
    assert np.all(masks.pool_mask_to_tokens(np.zeros((256, 320), dtype=bool)) == 0)


def test_pool_mask_single_stride16_cell():
    m = np.zeros((256, 320), dtype=bool)
    m[32:48, 64:80] = True  # exactly cell (row=2, col=4) of the stride-16 scale
    out = masks.pool_mask_to_tokens(m)
    assert out[masks.token_index(16, 2, 4)] == 255
    assert np.all(out[:320][np.arange(320) != masks.token_index(16, 2, 4)] == 0)
    # the patch covers a quarter of stride-32 cell (1, 2)
    assert out[masks.token_index(32, 1, 2)] == pytest.approx(64, abs=1)
    assert out[320:].sum() == out[masks.token_index(32, 1, 2)]


def test_pool_mask_half_cell():
    m = np.zeros((256, 320), dtype=bool)
    m[16:24, 0:16] = True  # half of stride-16 cell (1, 0): 8 x 16 of 16 x 16
    out = masks.pool_mask_to_tokens(m)
    assert out[masks.token_index(16, 1, 0)] == pytest.approx(128, abs=1)
    # the same patch covers an eighth of stride-32 cell (0, 0)
    assert out[masks.token_index(32, 0, 0)] == pytest.approx(32, abs=1)


def test_pool_mask_single_stride32_cell():
    m = np.zeros((256, 320), dtype=bool)
    m[32:64, 64:96] = True  # exactly stride-32 cell (1, 2)
    out = masks.pool_mask_to_tokens(m)
    assert out[masks.token_index(32, 1, 2)] == 255
    # and the four stride-16 cells it contains
    for r in (2, 3):
        for c in (4, 5):
            assert out[masks.token_index(16, r, c)] == 255


def test_pool_mask_rejects_bad_shape():
    with pytest.raises(ValueError):
        masks.pool_mask_to_tokens(np.zeros((240, 320), dtype=bool))


# -------------------------------------------------------- segmentation resize


def test_resize_segmentation_nearest_ids_exact():
    rng = np.random.RandomState(0)
    ids = rng.randint(0, 300, size=(240, 320)).astype(np.int64)
    out = masks.resize_segmentation_nearest(ids)
    assert out.shape == (256, 320)
    assert out.dtype == ids.dtype  # dtype preserved (cv2 silently retypes)
    # documented NEAREST convention: src = floor(dst * src/dst), integer-exact
    rows = (np.arange(256, dtype=np.int64) * 240) // 256
    assert np.array_equal(out, ids[rows])
    # ids are preserved exactly (no interpolation artefacts)
    assert set(np.unique(out)) <= set(np.unique(ids))
    # every source row survives at least once
    assert len(set(rows.tolist())) == 240


def test_resize_segmentation_nearest_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.RandomState(1)
    ids = rng.randint(0, 300, size=(240, 320)).astype(np.int32)
    out = masks.resize_segmentation_nearest(ids)
    ref = cv2.resize(ids, (320, 256), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(out, ref.astype(out.dtype))


def test_resize_segmentation_nearest_noop():
    ids = np.arange(256 * 320).reshape(256, 320)
    out = masks.resize_segmentation_nearest(ids)
    assert np.array_equal(out, ids)


# ------------------------------------------------------------------- helpers


def test_seg_id_like_entity_id_collection():
    # _item_entity_ids unions the entity with its particles (particle_set)
    parts = [MockActor(per_scene_id=10 + i) for i in range(3)]
    spec = ItemSpec(item_id="beads", entity=parts[0], category="beads", kind="particle_set", particles=parts)
    assert sorted(masks._item_entity_ids(spec).tolist()) == [10, 11, 12]
    rigid = ItemSpec(item_id="cup", entity=MockActor(per_scene_id=42), category="cup")
    assert masks._item_entity_ids(rigid).tolist() == [42]


class _MockCameras:
    def __init__(self):
        self.left_camera = object()
        self.right_camera = object()
        self.static_camera_list = ["head0", "head1"]
        self.head_camera_id = 1


class _MockCamTask:
    def __init__(self):
        self.cameras = _MockCameras()


def test_resolve_cameras_default_order():
    task = _MockCamTask()
    cams = masks.resolve_cameras(task, None)
    assert cams == [task.cameras.left_camera, task.cameras.right_camera, "head1"]


def test_resolve_cameras_names_and_passthrough():
    task = _MockCamTask()
    assert masks.resolve_cameras(task, ["head"]) == ["head1"]
    assert masks.resolve_cameras(task, ["left_wrist"]) == [task.cameras.left_camera]
    sentinel = object()
    assert masks.resolve_cameras(task, [sentinel]) == [sentinel]


# ------------------------------------------------- item_masks with mock cams


class MockCamera:
    """Duck-typed sapien camera returning a painted 320x240 segmentation."""

    def __init__(self, seg_ids: np.ndarray):
        self._seg = np.zeros((240, 320, 4), dtype=np.float32)
        self._seg[..., 1] = seg_ids
        self.pictures_taken = 0

    def take_picture(self):
        self.pictures_taken += 1

    def get_picture(self, name):
        assert name == "Segmentation"
        return self._seg

    def get_intrinsic_matrix(self):
        return np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]])

    def get_extrinsic_matrix(self):
        return np.eye(4)


class _MockMaskTask:
    def __init__(self):
        self.render_updates = 0

    def _update_render(self):
        self.render_updates += 1


def test_item_masks_items_and_nameable_regions():
    """End-to-end without sapien: one item + one nameable region, one camera.

    Exercises the region-target branch (the GPU-run NameError regression):
    painted seg ids must land in the documented token cells, and visibility
    must be nonzero where the item renders.
    """
    seg = np.zeros((240, 320), dtype=np.int64)
    seg[30:60, 40:88] = 11  # the item
    seg[100:164, 96:192] = 12  # the region entity
    cam = MockCamera(seg)
    task = _MockMaskTask()
    item = ItemSpec(
        item_id="obj_0",
        entity=MockActor(p=(0, 0, 1.0), per_scene_id=11),
        category="box",
        half_extents=[0.03, 0.03, 0.03],
    )
    region = regions.Region(
        region_id="zone_red",
        kind="zone",
        frame_fn=lambda: np.eye(4),
        polygon=[[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]],
        z=0.0,
        z_band=0.01,
        nameable=True,
        color="red",
        entity=MockActor(per_scene_id=12),
    )
    visibility, token_masks = masks.item_masks(task, [item], [region], [cam])
    assert task.render_updates == 1  # _update_render ran before the pictures
    assert visibility.shape == (1, 1) and visibility.dtype == np.float32
    assert token_masks.shape == (2, 1, 400) and token_masks.dtype == np.uint8
    # the item renders -> visible fraction in (0, 1]
    assert 0.0 < float(visibility[0, 0]) <= 1.0
    # item patch fully covers stride-16 cell (2, 3); region patch covers (8, 9)
    assert token_masks[0, 0, masks.token_index(16, 2, 3)] == 255
    assert token_masks[1, 0, masks.token_index(16, 8, 9)] == 255
    assert token_masks[0, 0].sum() > 0 and token_masks[1, 0].sum() > 0


def test_item_masks_skips_unmaskable_regions():
    """nameable=False or entity=None regions get no token-mask row."""
    seg = np.zeros((240, 320), dtype=np.int64)
    cam = MockCamera(seg)
    item = ItemSpec(
        item_id="obj_0",
        entity=MockActor(p=(0, 0, 1.0), per_scene_id=11),
        category="box",
        half_extents=[0.03, 0.03, 0.03],
    )
    hidden = regions.Region(
        region_id="internal",
        kind="zone",
        frame_fn=lambda: np.eye(4),
        polygon=[[0, 0], [1, 0], [1, 1]],
        z=0.0,
        z_band=0.01,
        nameable=False,
        entity=MockActor(per_scene_id=12),
    )
    no_entity = regions.Region(
        region_id="ghost",
        kind="zone",
        frame_fn=lambda: np.eye(4),
        polygon=[[0, 0], [1, 0], [1, 1]],
        z=0.0,
        z_band=0.01,
        nameable=True,
        entity=None,
    )
    visibility, token_masks = masks.item_masks(_MockMaskTask(), [item], [hidden, no_entity], [cam])
    assert visibility.shape == (1, 1)
    assert token_masks.shape == (1, 1, 400)  # only the item row
