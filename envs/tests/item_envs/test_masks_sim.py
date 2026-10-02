"""Sim tests for masks.item_masks (PLAN v3 §2.4) on a full RoboTwin Env.

Asserts the entity -> segmentation-id mapping (recon Q8 PENDING-GPU item: the
raw actor-level Segmentation channel 1 contains entity.per_scene_id where the
item renders), that a known item is visible in at least one camera at t=0, and
that its token mask is nonzero in that camera. Runs on the GPU node.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sapien")

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()

from main.item_envs import masks
from main.item_envs.items import entity_of
from main.item_envs.regions import table_region

from .sim_utils import make_env, obstacle_spec

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_item_masks_visibility_and_token_grid(embodiment):
    env = make_env(embodiment)
    task = env.task
    spec = obstacle_spec(env)
    ent = entity_of(spec.entity)
    ent_id = int(ent.per_scene_id)

    # entity -> seg-id mapping covers the obstacle (recon Q8)
    seg_map = masks._seg_id_map(task)
    assert ent_id in seg_map

    regions = [table_region(task)]
    nameable = [r for r in regions if r.nameable and r.entity is not None]
    assert len(nameable) == 1  # task.table must resolve for the table mask

    visibility, token_masks = masks.item_masks(task, [spec], regions, None)
    _, n_tok = masks.token_grid()
    assert visibility.shape == (1, 3)
    assert visibility.dtype == np.float32
    assert token_masks.shape == (1 + len(nameable), 3, n_tok)
    assert token_masks.dtype == np.uint8

    # the obstacle is visible in at least one camera at t = 0
    assert float(visibility[0].max()) > 0.0
    cam = int(np.argmax(visibility[0]))
    # and its token mask is nonzero in that camera
    assert int(token_masks[0, cam].max()) > 0

    # raw actor-level seg channel contains the obstacle's per_scene_id where it
    # renders (closes recon Q8's PENDING-GPU runtime confirmation)
    task._update_render()
    head = task.cameras.static_camera_list[task.cameras.head_camera_id]
    head.take_picture()
    seg = head.get_picture("Segmentation")
    assert (seg[..., 1] == ent_id).any()
