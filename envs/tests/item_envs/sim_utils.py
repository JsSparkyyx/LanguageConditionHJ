"""Shared helpers for the sim-marked item_envs tests (GPU node only).

These build a full RoboTwin Env through CEHJ's make_training_env — the same
path collection uses (cuRobo controller install, on-path obstacle, cuRobo
world refresh). They cannot run on the login node (no GPU/Vulkan).
"""

from __future__ import annotations

import numpy as np

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()

from main.item_envs.items import ItemSpec, entity_of


def make_env(embodiment: str, seed: int = 0):
    """place_empty_cup + the single CrossSafe obstacle, as make_training_env does."""
    from main.train.collect import make_training_env
    from main.train.config import FrozenConfig

    cfg = FrozenConfig(
        task="place_empty_cup",
        embodiment=embodiment,
        obstacle_mode="on_path",
        obstacle_model="068_boxdrink",
    )
    return make_training_env(cfg, seed)


def actor_spec(env, actor, item_id: str, category: str, half=None, role: str = "candidate") -> ItemSpec:
    """ItemSpec for any scene actor with OBB parity to distance.py's _obb_from_actor.

    center_local/half are recovered from the same values _obb_from_actor
    returns, so robot_item_distances reproduces _actor_obb_distance /
    distance_info exactly for this actor. ``half`` is the optional override
    passed through to _obb_from_actor (env.obstacle_half for the obstacle).
    """
    from main.envs import distance as cehj_distance

    origin, rot, half_out = cehj_distance._obb_from_actor(actor, half)
    pose = entity_of(actor).get_pose()
    center_local = rot.T @ (origin - np.asarray(pose.p, dtype=np.float64).reshape(3))
    return ItemSpec(
        item_id=item_id,
        entity=actor,
        category=category,
        half_extents=np.asarray(half_out, dtype=np.float64),
        obb_center_local=center_local,
        role=role,
    )


def obstacle_spec(env) -> ItemSpec:
    """ItemSpec for env.obstacle with OBB parity to distance_info's d_system."""
    return actor_spec(env, env.obstacle, "obstacle", "drink_box", half=getattr(env, "obstacle_half", None))
