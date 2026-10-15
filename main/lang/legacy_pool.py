"""The T1 candidate pool and the vocabulary before 2026-10-12 (V6, qwen_impl.md 4).

7c2476d (2026-10-12) took 068_boxdrink id 0 and 086_woodenblock ids 1 / 2 out of the T1 pool and renamed
068_boxdrink's category (drink_box -> bottle), so today's code draws other T1 objects for the seeds the vanilla
runs were collected and trained with. objects.yaml keeps the old draws as ``legacy_ids`` / ``legacy_category``;
``install_spawn_pool("pre_1012")`` rebuilds the old pool from them (every entry gets ``ids = sorted(ids +
legacy_ids)`` and ``category = legacy_category or category``; the pool order is objects.yaml's ``t1_pool``, which
did not change), so the same seed places the same objects again. "current" restores today's pool. Names still
come from today's vocabulary.

spawn.py looks ``t1_pool_entries`` up as a module global at each layout (spawn.py:1145), so wrapping the global
switches every later episode of the process; the wrap is idempotent.

``install_vocab("pre_1012")`` (2026-10-15, the V6 pre-run review) goes further: the whole process uses objects.yaml
and paraphrases.yaml as they were before 7c2476d (copies in vocab_pre_1012/, byte-identical to the files of both
vanilla runs' worktrees), so V6 sees the vanilla runs' sentences, sampleable rules and held-out words, not only
their scenes. The 10-12 change of objects.yaml touched names only (every mass and friction range, colour flag and
colour list is unchanged; the old file holds the old T1 pool itself), so the old file is installed everywhere it is
read: physics.load_objects_cfg (physics, the T1 pool, the rule table's names and "ambiguous" flags), the naming
caches of the language module (_OBJECTS, _BANK: object words, sentences, sampleable flags, held-out words) and
rules._HELDOUT_WORDS. RelabelSampler re-names every stored record at load (language.refresh_record, wrapped here to
re-name the records' region words too), so shards collected with today's names train on the old ones. Install before anything names an object (collect.run_episodes,
LangTrainer, the evaluators).
"""

from __future__ import annotations

from pathlib import Path

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
    if name == "current" and _ORIGINAL is None:
        return   # never switched in this process: nothing to restore (and no spawn import)
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


VOCABS = ("current", "pre_1012")
VOCAB_DIR = Path(__file__).resolve().parent / "vocab_pre_1012"
_VOCAB = {"name": "current", "load_objects_cfg": None, "refresh_record": None}


def vocab_files(name: str) -> tuple[Path, Path]:
    """(objects.yaml, paraphrases.yaml) of a vocabulary."""
    if name == "current":
        root = Path(__file__).resolve().parents[2] / "envs" / "main" / "item_envs"
        return root / "configs" / "objects.yaml", root / "language" / "paraphrases.yaml"
    if name == "pre_1012":
        return VOCAB_DIR / "objects.yaml", VOCAB_DIR / "paraphrases.yaml"
    raise ValueError(f"vocab {name!r}: expected one of {VOCABS}")


def install_vocab(name: str) -> None:
    """Switch the process's objects.yaml / paraphrase bank to ``name`` ("current" or "pre_1012"); idempotent."""
    if name not in VOCABS:
        raise ValueError(f"vocab {name!r}: expected one of {VOCABS}")
    if name == _VOCAB["name"]:
        return
    import yaml

    import main.lang  # noqa: F401  (puts envs/main on main.__path__)
    from main.item_envs import language, physics, rules

    if _VOCAB["load_objects_cfg"] is None:
        _VOCAB["load_objects_cfg"] = physics.load_objects_cfg
        _VOCAB["refresh_record"] = language.refresh_record
    original = _VOCAB["load_objects_cfg"]
    if name == "current":
        physics.load_objects_cfg = original
        language.refresh_record = _VOCAB["refresh_record"]
        language._OBJECTS = language._BANK = None   # reloaded from today's files on first use
        rules._HELDOUT_WORDS = None
    else:
        objects_p, bank_p = vocab_files(name)
        objects = yaml.safe_load(objects_p.read_text())
        bank = yaml.safe_load(bank_p.read_text())

        def load_objects_cfg(path: str | None = None) -> dict:
            return original(path) if path else objects   # one shared dict, as the lru_cache'd original gives

        load_objects_cfg.vocab = name
        physics.load_objects_cfg = load_objects_cfg
        language._OBJECTS, language._BANK = objects, bank
        refresh = _VOCAB["refresh_record"]

        def refresh_record(record: dict, model_ids: dict | None = None, objects_cfg: dict | None = None,
                           bank: dict | None = None) -> dict:
            """language.refresh_record re-names objects and bead words only; a record also stores its regions'
            words (episode_record, from objects.yaml regions): re-name them first, then the original re-names the
            rest and recomputes every rule's sampleable flag (T6 bins were 'bin / box / tub' before 10-12)."""
            cfg = objects_cfg if objects_cfg is not None else language._objects_cfg()
            regions_cfg = cfg.get("regions", {}) or {}
            for r in record.get("regions") or []:
                key, entry = language._region_key(r["region_id"], regions_cfg)
                r["kind"], r["names"] = key, language._words(entry) or [key.replace("_", " ")]
            return refresh(record, model_ids, objects_cfg, bank)

        refresh_record.vocab = name
        language.refresh_record = refresh_record
        rules._HELDOUT_WORDS = {str(w) for w in bank.get("heldout_words", []) or []}
    _VOCAB["name"] = name


def installed_vocab() -> str:
    return _VOCAB["name"]


def checkpoint_vocab(lang_config: dict) -> str:
    """The vocabulary a saved config trained with: its vocab field, else "pre_1012" for configs saved before the
    10-12 renaming (their bank hash is PRE_1012_BANK_HASH: both vanilla runs), else "current". For the evaluators;
    LangConfig.from_dict keeps "current" as the default, since a default there would move the collect hash of every
    saved shard config."""
    d = lang_config or {}
    if d.get("vocab"):
        return str(d["vocab"])
    return "pre_1012" if d.get("bank_hash") == PRE_1012_BANK_HASH else "current"
