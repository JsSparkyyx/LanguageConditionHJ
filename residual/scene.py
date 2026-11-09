"""Scenes of the don't-touch task: build one, find which extra objects are in the planner's way, name one rule.

In the planner's way (``main/lang/scene_roles._t1_reasons``, fixed per seed): on-path and near-path objects, and
off-path objects within 10 cm of a second carry corridor. Every other extra object is out of the way. The spawner
puts 2 objects on or near the path whenever it can, so only some scenes have an out-of-the-way object.

The named rule is one extra object's ``avoid`` (don't touch: an arm sphere or the held object within 1 cm) or
``stay`` (don't move: 3 cm, or tilt, 20 deg counting as the full 3 cm) rule. Other objects are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ensure_paths

ensure_paths()

RULE_KINDS = ("avoid", "stay")


@dataclass
class Named:
    item_id: str
    kind: str            # "avoid" | "stay"
    in_way: bool
    reason: str          # on_path / near_path / second_corridor / out_of_way
    rule_index: int      # index into runner.rules
    item_index: int      # index into runner.items
    category: str = ""


def build_runner(cfg, seed: int, out_root: str):
    """ItemEpisodeRunner with Env, objects, physics and rules built, no controller (render off)."""
    from main.item_envs.runner import ItemEpisodeRunner

    r = ItemEpisodeRunner(cfg.task, embodiment=cfg.embodiment, seed=int(seed), render="none", out_root=out_root,
                          max_steps=int(cfg.max_physics_steps))
    try:
        r.setup(build_controller=False)
    except BaseException:
        r.close()
        raise
    r.env._item_runner = r   # scene_roles reads the runner through the env
    return r


def candidate_roles(runner) -> dict[str, str]:
    """{item_id: reason} for every extra object: on_path / near_path / second_corridor, or out_of_way."""
    from main.lang.scene_roles import _t1_reasons

    reasons = _t1_reasons(runner.env, None)
    return {s.item_id: reasons.get(s.item_id, "out_of_way") for s in runner.items if s.role == "candidate"}


def _rule_index(runner, kind: str, item_id: str) -> int | None:
    rid = f"{kind}|{item_id}|-"
    for j, rule in enumerate(runner.rules):
        if rule.rule_id == rid:
            return j
    return None


def named_options(runner) -> dict[bool, list[str]]:
    """{True: in-the-way item ids, False: out-of-the-way item ids}, each with both an avoid and a stay rule."""
    roles = candidate_roles(runner)
    out = {True: [], False: []}
    for iid, why in roles.items():
        if all(_rule_index(runner, k, iid) is not None for k in RULE_KINDS):
            out[why != "out_of_way"].append(iid)
    return out


_REASON_ORDER = {"on_path": 0, "near_path": 1, "second_corridor": 2, "out_of_way": 3}


def make_named(runner, item_id: str, kind: str) -> Named:
    roles = candidate_roles(runner)
    if item_id not in roles:
        raise ValueError(f"{item_id!r} is not an extra object of this scene ({sorted(roles)})")
    j = _rule_index(runner, kind, item_id)
    if j is None:
        raise ValueError(f"no {kind} rule for {item_id!r}")
    k = next(i for i, s in enumerate(runner.items) if s.item_id == item_id)
    why = roles[item_id]
    return Named(item_id=item_id, kind=kind, in_way=why != "out_of_way", reason=why, rule_index=j, item_index=k,
                 category=str(runner.items[k].category))


def pick_named(runner, in_way: bool, kind: str, rng: np.random.Generator | None = None) -> Named | None:
    """One object of the wanted side (random with ``rng``; the most in-the-way one, then the first, without)."""
    opts = named_options(runner)[bool(in_way)]
    if not opts:
        return None
    if rng is None:
        roles = candidate_roles(runner)
        item_id = sorted(opts, key=lambda i: (_REASON_ORDER.get(roles[i], 9), i))[0]
    else:
        item_id = opts[int(rng.integers(len(opts)))]
    return make_named(runner, item_id, kind)


def scan_record(runner) -> dict:
    """What a seed's scene offers (for the evaluation scene list and the logs)."""
    roles = candidate_roles(runner)
    return {
        "host": runner.host,
        "roles": roles,
        "categories": {s.item_id: str(s.category) for s in runner.items if s.role == "candidate"},
        "n_candidates": len(roles),
        "time_scale": float(runner.risk_params.time_scale),
    }
