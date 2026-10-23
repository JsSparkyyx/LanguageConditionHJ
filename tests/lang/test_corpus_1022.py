"""The hand-picked T1 corpus of 2026-10-22 (Yuxuan): training models and colours in the training pool, the held-out
models and colours only in the "heldout" pool, a category drawn uniformly (book has two assets), and every rule
naming a held-out model in the heldout_item split."""

from collections import Counter

import numpy as np

import main.lang  # noqa: F401  (puts envs/main on main.__path__)
from main.item_envs import physics, rules, spawn
from main.item_envs.rules import RuleSpec
from main.lang import legacy_pool


def _pool(name):
    try:
        legacy_pool.install_spawn_pool(name)
        return spawn.t1_pool_entries(physics.load_objects_cfg())
    finally:
        legacy_pool.install_spawn_pool("current")


def test_training_pool_never_draws_a_heldout_model():
    cfg = physics.load_objects_cfg()
    assert cfg["colors"] == {"train": ["red", "green", "blue"], "heldout": ["purple", "orange"], "fixed": ["white"]}
    pool = _pool("current")
    assert sorted({e["category"] for e in pool}) == ["apple", "bottle", "box", "cylinder", "milk_box", "wine_glass"]
    for e in pool:
        assert not set(e.get("ids") or []) & set(e.get("heldout_ids") or []), e["asset"]
        assert not e.get("heldout") and "colors" not in e
    # 3 training + 1 held-out model per object type (apple 1 + 1)
    per_cat = Counter()
    held = Counter()
    for e in pool:
        if not e["asset"].startswith("prim_"):
            per_cat[e["category"]] += len(e["ids"])
            held[e["category"]] += len(e.get("heldout_ids") or [])
    assert per_cat == {"wine_glass": 3, "milk_box": 3, "apple": 1, "bottle": 3}
    assert set(held.values()) == {1} and set(held) == set(per_cat)


def test_heldout_pool_draws_only_heldout_models_and_colours():
    pool = _pool("heldout")
    assert legacy_pool.installed_pool() == "current"
    assert len(pool) == 6 and all(e["heldout"] for e in pool)
    rng = np.random.RandomState(0)
    for _ in range(300):
        d = spawn._draw_one(pool, rng, ["red", "green", "blue"])
        assert d.heldout
        if d.asset.startswith("prim_"):
            assert d.color in ("purple", "orange")
        else:
            e = next(x for x in pool if x["asset"] == d.asset)
            assert [d.model_id] == [i for i in e["heldout_ids"] if i == d.model_id]


def test_category_is_drawn_uniformly():
    pool = _pool("current")
    # a category with two assets (as the book had, 043_book + 092_notebook): uniform over categories
    two = pool + [{**pool[0], "asset": pool[0]["asset"] + "_b"}]
    rng = np.random.RandomState(1)
    n = 18000
    cats = Counter(spawn._draw_one(two, rng, ["red", "green", "blue"]).category for _ in range(n))
    assert len(cats) == 6
    for c, k in cats.items():   # 1/6 each (asset-uniform would give the doubled category 2/7)
        assert abs(k / n - 1 / 6) < 0.015, (c, k / n)
    # a pool with one asset per category keeps the single original draw (old pools replay the same objects)
    a, b = np.random.RandomState(5), np.random.RandomState(5)
    assert spawn._pick_entry(pool, a) is pool[int(b.randint(len(pool)))]


def test_heldout_model_rule_is_heldout_item():
    import yaml

    split = rules.resolve_splits(yaml.safe_load((main.lang.REPO / "envs/main/item_envs/configs/splits.yaml").read_text()),
                                 "selective_avoid")
    base = dict(consequence=rules.Consequence.AVOID, item_ids=("obj_0",), region_ids=(), phrases=["don't touch the kettle"])
    plain = RuleSpec(rule_id="r0", params={"categories": ("kettle",), "colors": (None,)}, **base)
    held = RuleSpec(rule_id="r1", params={"categories": ("kettle",), "colors": (None,), "heldout_model": True}, **base)
    assert rules.assign_split(plain, split) == "train"
    assert rules.assign_split(held, split) == "heldout_item"
    assert rules.assign_split(RuleSpec(rule_id="r2", params={"categories": ("box",), "colors": ("purple",)}, **base),
                              {"heldout_colors": ["purple", "orange"]}) == "heldout_item"


def test_every_pool_model_fits_the_on_path_limit():
    """2026-10-22: the models wider than the on-path size limit are shrunk (objects.yaml scale_by_id), so every T1
    model, training and held-out, can stand on the path."""
    import yaml

    t1 = yaml.safe_load((main.lang.REPO / "envs/main/item_envs/configs/tasks/T1_selective_avoid.yaml").read_text())
    cap = float(t1["objects"]["on_path_max_r_xy"])
    assert spawn.model_scale("088_wineglass", 0) == 0.87 and spawn.model_scale("088_wineglass", 1) == 1.0
    for pool in ("current", "heldout"):
        for e in _pool(pool):
            if e["asset"].startswith("prim_"):
                continue
            for mid in e["ids"]:
                g = spawn._geometry_of(spawn._ObjectDraw(asset=e["asset"], category=e["category"], model_id=int(mid)))
                assert float(np.max(g.world_half[:2])) <= cap, (pool, e["asset"], mid, g.world_half)
