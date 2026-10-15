"""The T1 candidate pool before 2026-10-12 (V6, qwen_impl.md 4).

7c2476d (2026-10-12) took 068_boxdrink id 0 and 086_woodenblock ids 1 / 2 out of the T1 pool and renamed
068_boxdrink's category (drink_box -> bottle), so today's code draws other T1 objects for the seeds the vanilla
runs were collected and trained with. objects.yaml keeps the old draws as ``legacy_ids`` / ``legacy_category``;
``install_spawn_pool("pre_1012")`` rebuilds the old pool from them (every entry gets ``ids = sorted(ids +
legacy_ids)`` and ``category = legacy_category or category``; the pool order is objects.yaml's ``t1_pool``, which
did not change), so the same seed places the same objects again. "current" restores today's pool. Names still
come from today's vocabulary.

spawn.py looks ``t1_pool_entries`` up as a module global at each layout (spawn.py:1145), so wrapping the global
switches every later episode of the process; the wrap is idempotent.
"""

from __future__ import annotations

# the bank hash every pre-10-12 shard records: bank_hash() then covered paraphrases.yaml + splits.yaml only (the
# same at ae5b02c and 7c2476d^; objects.yaml did not change between them)
PRE_1012_BANK_HASH = "693f7ef522ed"
POOLS = ("current", "pre_1012")

_ORIGINAL = None


def _spawn():
    import main.lang  # noqa: F401  (puts envs/main on main.__path__)
    from main.item_envs import spawn

    return spawn


def legacy_entries(entries: list[dict]) -> list[dict]:
    """Today's pool entries -> the pre-10-12 ones (copies; entries without legacy fields are unchanged)."""
    out = []
    for e in entries:
        e = dict(e)
        legacy = list(e.get("legacy_ids") or [])
        if legacy:
            e["ids"] = sorted({int(i) for i in list(e.get("ids") or []) + legacy})
        if e.get("legacy_category"):
            e["category"] = e["legacy_category"]
        out.append(e)
    return out


def install_spawn_pool(name: str) -> None:
    """Switch the process's T1 pool to ``name`` ("current" or "pre_1012")."""
    global _ORIGINAL
    if name not in POOLS:
        raise ValueError(f"spawn_pool {name!r}: expected one of {POOLS}")
    spawn = _spawn()
    if _ORIGINAL is None:
        _ORIGINAL = spawn.t1_pool_entries
    if name == "current":
        spawn.t1_pool_entries = _ORIGINAL
        return
    original = _ORIGINAL

    def t1_pool_entries_pre_1012(objects_cfg: dict) -> list[dict]:
        return legacy_entries(original(objects_cfg))

    t1_pool_entries_pre_1012.spawn_pool = "pre_1012"
    spawn.t1_pool_entries = t1_pool_entries_pre_1012


def installed_pool() -> str:
    return getattr(_spawn().t1_pool_entries, "spawn_pool", "current")
