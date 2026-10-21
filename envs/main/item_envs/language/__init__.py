"""Relabeling language side (docs/item_envs/relabeling.md).

- ``episode_record(runner)``: the per-episode ``episodes/{episode_id}.json``
  in the relabeling format — objects / regions with the words a sentence
  needs, one rule entry per column of the per-tick ``h_rules`` array.
- ``describe(obj, scene)``: referent phrases that name ONE object of the
  scene ("the cup" is invalid with two cups: the color / size form is
  required); ``describe_region`` the same for zones, bins, plates, the table.
- ``sentences(rule, scene, split)``: every (template x referent phrase)
  sentence of a rule; ``sample_sentence`` draws one.

The scene is the episode record (a plain dict), so training code needs no
simulator. Sentences are never stored per tick: they are generated at sample
time from the rule and the paraphrase bank (paraphrases.yaml).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import yaml

__all__ = [
    "SPEC_CONSEQUENCE", "load_bank", "episode_record", "describe", "describe_region",
    "materials", "sentences", "sample_sentence", "sampleable", "refresh_record", "load_model_ids",
]
NAMES_VERSION = 3   # 2026-10-12: names checked against renders, per-model names (objects.yaml assets); 3 (2026-10-20): plain region names, regions re-named at load

# rules.Consequence value -> relabeling consequence name (the bank's keys)
SPEC_CONSEQUENCE = {
    "avoid": "avoid",
    "upright": "upright",
    "stay": "stay",
    "outside": "zone_exclude",
    "inside": "zone_include",
    "particles_inside": "contain_particles",
    "particles_outside": "exclude_particles",
}
SPLITS = ("train", "heldout_template", "heldout_word")

_BANK = None
_HERE = Path(__file__).resolve().parent


def load_bank(path: str | Path | None = None) -> dict:
    """The paraphrase bank (cached for the default path)."""
    global _BANK
    if path is not None:
        return yaml.safe_load(Path(path).read_text())
    if _BANK is None:
        _BANK = yaml.safe_load((_HERE / "paraphrases.yaml").read_text())
    return _BANK


_OBJECTS = None


def _objects_cfg() -> dict:
    """objects.yaml (cached: refresh_record runs once per stored record)."""
    global _OBJECTS
    if _OBJECTS is None:
        _OBJECTS = yaml.safe_load((_HERE.parent / "configs" / "objects.yaml").read_text())
    return _OBJECTS


# ---------------------------------------------------------------- record ----


def _words(entry: dict) -> list[str]:
    return [str(w) for w in list(entry.get("names", []) or []) + list(entry.get("synonyms", []) or [])]


def _cat_words(cats: dict, key: str) -> list[str]:
    return _words(cats.get(str(key), {}) or {}) or [str(key).replace("_", " ")]


def model_words(cfg: dict, asset: str, model_id: int) -> tuple[str, list[str], bool]:
    """(category key, words, nameable) of one asset model: the asset's category, unless objects.yaml overrides it
    for that model (category_by_id, names_by_id); unnamed_ids are never the subject of a sentence."""
    cats, a = cfg.get("categories", {}) or {}, (cfg.get("assets", {}) or {}).get(str(asset)) or {}
    mid = int(model_id)
    key = str((a.get("category_by_id") or {}).get(mid, a.get("category", asset)))
    over = (a.get("names_by_id") or {}).get(mid)
    words = _words(over) if over else _cat_words(cats, key)
    return key, words, mid not in {int(i) for i in (a.get("unnamed_ids") or [])}


def unknown_model_words(cfg: dict, category_key: str) -> tuple[str, list[str], bool]:
    """A pool object recorded without its model id (episodes collected before 2026-10-12, no recovered id): the
    words every model it may be share. Nameable only if all those models are nameable and of one category; with no
    shared word it keeps its category's words (so other objects' phrases stay unique) and is never named."""
    cats, assets = cfg.get("categories", {}) or {}, cfg.get("assets", {}) or {}
    # the models a record of that key can be: the key its episodes recorded (legacy_category when the asset's key
    # changed since, e.g. 068_boxdrink was "drink_box" for every model)
    cands = [(n, a) for n, a in assets.items() if a.get("legacy_category", a.get("category")) == category_key]
    if not any(a.get("names_by_id") or a.get("category_by_id") or a.get("unnamed_ids") for _, a in cands):
        return category_key, _cat_words(cats, category_key), True
    keys, shared, ok = set(), None, True
    for n, a in cands:
        for mid in list(a.get("ids") or [0]) + list(a.get("legacy_ids") or []):
            k, w, named = model_words(cfg, n, mid)
            keys.add(k); ok &= named
            shared = set(w) if shared is None else shared & set(w)
    words = [w for w in _cat_words(cats, category_key) if w in (shared or set())]
    if not words:
        return category_key, _cat_words(cats, category_key), False
    return category_key, words, ok and len(keys) == 1


def _name_object(o: dict, cfg: dict, task, asset=None, model_id=None) -> dict:
    """Fill an object entry's names in place (category, synonyms, nameable, asset, model_id, category_key)."""
    cats = cfg.get("categories", {}) or {}
    if "category_key" not in o:   # a hand-built scene (tests, record_from_meta fixtures): nothing to re-name from
        o.setdefault("nameable", True)
        return o
    asset = asset if asset is not None else o.get("asset")
    model_id = model_id if model_id is not None else o.get("model_id")
    if asset is not None and model_id is not None and asset in (cfg.get("assets", {}) or {}):
        key, words, ok = model_words(cfg, asset, model_id)
    elif task == "selective_avoid" and o.get("role") == "candidate":
        key, words, ok = unknown_model_words(cfg, o["category_key"])
    else:
        key, words, ok = o["category_key"], _cat_words(cats, o["category_key"]), True
    o.update({"category_key": key, "category": words[0], "synonyms": words[1:], "nameable": bool(ok),
              "asset": asset, "model_id": None if model_id is None else int(model_id)})
    return o


def _region_key(region_id: str, regions_cfg: dict) -> tuple[str, dict]:
    tokens = region_id.split("_")
    for end in range(len(tokens), 0, -1):
        key = "_".join(tokens[:end])
        if key in regions_cfg:
            return key, regions_cfg[key] or {}
    return region_id, {}


def episode_record(runner, episode_id: int = 0, objects_cfg: dict | None = None) -> dict:
    """The relabeling episode record of an ItemEpisodeRunner (or anything with
    task_name / embodiment / items / regions / rules / active_rule_id)."""
    from ..margins import h_unit

    cfg = objects_cfg if objects_cfg is not None else _objects_cfg()
    cats = cfg.get("categories", {}) or {}
    regions_cfg = cfg.get("regions", {}) or {}
    # T1 pool objects: the asset and model id from the spawn report, so names can follow the model (2026-10-12)
    drawn = {p.get("item_id"): (p.get("asset"), p.get("model_id"))
             for p in ((getattr(runner, "_spawn_report", None) or {}).get("placements") or [])}
    objects = []
    for k, s in enumerate(runner.items):
        o = {"id": k, "item_id": s.item_id, "category_key": str(s.category), "color": (s.color or None),
             "size": getattr(s, "size", None), "role": s.role, "kind": s.kind}
        asset, mid = drawn.get(s.item_id, (None, None))
        objects.append(_name_object(o, cfg, getattr(runner, "task_name", None), asset, mid))
    regions = []
    for k, r in enumerate(runner.regions):
        key, entry = _region_key(r.region_id, regions_cfg)
        words = _words(entry) or [key.replace("_", " ")]
        regions.append({
            "id": k, "region_id": r.region_id, "kind": key, "names": words,
            "color": (getattr(r, "color", None) or None), "role": getattr(r, "role", None),
            "nameable": bool(getattr(r, "nameable", True)),
        })
    obj_index = {o["item_id"]: o["id"] for o in objects}
    reg_index = {r["region_id"]: r["id"] for r in regions}
    rules = []
    active = -1
    for j, rule in enumerate(runner.rules):
        cons = rule.consequence.value if hasattr(rule.consequence, "value") else str(rule.consequence)
        name = SPEC_CONSEQUENCE.get(cons, cons)
        p = dict(rule.params or {})
        particles = cons.startswith("particles")
        is_set = len(rule.item_ids) > 1 or (cons in ("inside", "outside") and len(rule.region_ids) > 1)
        subj_ids = [obj_index[i] for i in rule.item_ids]
        if particles:
            subject = {"particles": subj_ids[0]}
        else:
            subject = {"object": subj_ids[0]} if len(subj_ids) == 1 else {"objects": subj_ids}
        ref_ids = [reg_index[i] for i in rule.region_ids]
        reference = None if not ref_ids else ({"region": ref_ids[0]} if len(ref_ids) == 1 else {"regions": ref_ids})
        params = {"h_unit": h_unit(rule)}
        if "margin" in p:
            params["margin_m"] = float(p["margin"])
        if "theta_max" in p:
            params["tilt_max_deg"] = round(float(np.rad2deg(p["theta_max"])), 3)
        if "d_tol" in p:
            params["disp_max_m"] = float(p["d_tol"])
        if "theta_tol" in p:
            params["tilt_max_deg"] = round(float(np.rad2deg(p["theta_tol"])), 3)
        if particles:
            params["material"] = materials(objects[subj_ids[0]])
        entry = {
            "idx": j, "rule_id": rule.rule_id, "consequence": name, "subject": subject,
            "reference": reference, "params": params, "set": bool(is_set),
            "split": rule.split, "ambiguous": bool(getattr(rule, "ambiguous", False)),
        }
        rules.append(entry)
        if rule.rule_id == getattr(runner, "active_rule_id", None):
            active = j
    record = {
        "episode_id": int(episode_id),
        "task": getattr(runner, "task_name", None),
        "host_task": getattr(runner, "host", None),
        "embodiment": getattr(runner, "embodiment", None),
        "seed": getattr(runner, "seed", None),
        "active_rule": active,
        "active_instruction": (runner.rules[active].phrases[0] if active >= 0 and runner.rules[active].phrases else None),
        "objects": objects,
        "regions": regions,
        "rules": rules,
        "names_version": NAMES_VERSION,
    }
    for entry in rules:
        entry["sampleable"] = sampleable(entry, record)
    return record


# --------------------------------------------------------------- describe ----


def _heldout_words(bank: dict) -> set:
    return {str(w) for w in bank.get("heldout_words", []) or []}


def _obj_words(o: dict) -> list[str]:
    return [o["category"]] + list(o.get("synonyms") or [])


def _answers(o: dict, noun: str, color, size) -> bool:
    """Would someone looking at object o accept the phrase (noun, color, size)?
    A bare head noun fits any of o's words with that head ("the box" fits a
    coffee box, "the bottle" a pill bottle); "the coffee box" still does not
    fit a plain box (D1, Yuxuan 2026-10-09)."""
    if not any(w == noun or w.endswith(" " + noun) for w in _obj_words(o)):
        return False
    if color is not None and o.get("color") != color:
        return False
    if size is not None and o.get("size") != size:
        return False
    return True


def _forms(noun: str, color, size) -> list[tuple]:
    out = [(None, None)]
    if color:
        out.append((color, None))
    if size:
        out.append((None, size))
    if color and size:
        out.append((color, size))
    return out


def _phrase(noun: str, color, size) -> str:
    return "the " + " ".join(w for w in (size, color, noun) if w)


def _noun_ok(noun: str, words: str, heldout: set) -> bool:
    return (noun in heldout) if words == "heldout" else (noun not in heldout)


def describe(obj: dict, scene: dict, words: str = "train", bank: dict | None = None) -> list[str]:
    """Referent phrases for one object that no OTHER object of the scene
    answers to: "the {color} {noun}", "the {noun}", "the {size} {noun}",
    "the {size} {color} {noun}", the noun from the category word and its
    synonyms. words="train" never uses a held-out word, words="heldout" only
    held-out ones. Empty when nothing names the object uniquely."""
    if obj.get("nameable") is False:   # e.g. a model the names check excluded (objects.yaml unnamed_ids)
        return []
    bank = bank or load_bank()
    heldout = _heldout_words(bank)
    others = [o for o in scene["objects"] if o["id"] != obj["id"] and o.get("kind") != "particle_set"]
    out = []
    for noun in _obj_words(obj):
        if not _noun_ok(noun, words, heldout):
            continue
        for color, size in _forms(noun, obj.get("color"), obj.get("size")):
            if any(_answers(o, noun, color, size) for o in others):
                continue
            p = _phrase(noun, color, size)
            if p not in out:
                out.append(p)
    return out


def describe_region(region: dict, scene: dict, words: str = "train", bank: dict | None = None) -> list[str]:
    """Phrases for a region ("the red zone", "the green bin", "the table")
    that no other region — and no object — of the scene answers to."""
    bank = bank or load_bank()
    heldout = _heldout_words(bank)
    others = [r for r in scene["regions"] if r["id"] != region["id"]]
    objs = [o for o in scene["objects"] if o.get("kind") != "particle_set"]
    out = []
    for noun in region.get("names") or []:
        if not _noun_ok(noun, words, heldout):
            continue
        for color, _ in _forms(noun, region.get("color"), None):
            if any(noun in (r.get("names") or []) and (color is None or r.get("color") == color) for r in others):
                continue
            if any(_answers(o, noun, color, None) for o in objs):
                continue
            p = _phrase(noun, color, None)
            if p not in out:
                out.append(p)
    return out


def materials(particles_obj: dict, words: str = "train", bank: dict | None = None) -> list[str]:
    """Bare particle words ("rice", "beads", ...) for a particle set."""
    heldout = _heldout_words(bank or load_bank())
    return [w for w in _obj_words(particles_obj) if _noun_ok(w, words, heldout)]


# -------------------------------------------------------------- sentences ----


def _fillers(rule: dict, scene: dict, words: str, bank: dict) -> dict | None:
    """Placeholder -> phrase choices for a rule; None if a referent has no
    unique phrase (or the rule has no single-referent form)."""
    if rule.get("set"):
        return None
    subj, ref = rule["subject"], rule.get("reference")
    out = {}
    if "particles" in subj:
        out["material"] = materials(scene["objects"][subj["particles"]], words, bank)
    elif "object" in subj:
        out["obj"] = describe(scene["objects"][subj["object"]], scene, words, bank)
    else:
        return None
    if rule["consequence"] == "contain_particles":
        ref = None  # the containers are not named ("don't spill the rice")
    if ref is not None:
        if "region" not in ref:
            return None
        out["region"] = describe_region(scene["regions"][ref["region"]], scene, words, bank)
    if any(len(v) == 0 for v in out.values()):
        return None
    return out


def sentences(rule: dict, scene: dict, split: str = "train", bank: dict | None = None) -> list[str]:
    """Every sentence of a rule in a split: "train" = train templates x train
    words; "heldout_template" = held-out templates x train words;
    "heldout_word" = train templates x held-out words."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    bank = bank or load_bank()
    entry = bank.get(rule["consequence"])
    if not entry:
        return []
    templates = entry["heldout"] if split == "heldout_template" else entry["train"]
    words = "heldout" if split == "heldout_word" else "train"
    fill = _fillers(rule, scene, words, bank)
    if fill is None:
        return []
    out = []
    for t in templates:
        keys = [k for k in ("obj", "region", "material") if "{" + k + "}" in t]
        if any(k not in fill for k in keys):
            continue
        for combo in itertools.product(*(fill[k] for k in keys)):
            s = t.format(**dict(zip(keys, combo)))
            if s not in out:
                out.append(s)
    return out


def sample_sentence(rule: dict, scene: dict, rng: np.random.Generator, split: str = "train",
                    bank: dict | None = None) -> str:
    """One sentence: a uniform template, then uniform referent phrases."""
    bank = bank or load_bank()
    entry = bank[rule["consequence"]]
    templates = list(entry["heldout"] if split == "heldout_template" else entry["train"])
    fill = _fillers(rule, scene, "heldout" if split == "heldout_word" else "train", bank)
    if fill is None:
        raise ValueError(f"rule {rule.get('rule_id')} has no {split} sentence in this scene")
    rng.shuffle(templates)
    for t in templates:
        keys = [k for k in ("obj", "region", "material") if "{" + k + "}" in t]
        if all(k in fill for k in keys):
            return t.format(**{k: fill[k][int(rng.integers(len(fill[k])))] for k in keys})
    raise ValueError(f"rule {rule.get('rule_id')}: no template fits its referents")


def load_model_ids(root) -> dict:
    """{episode_id: {item_id: (asset, model_id)}} recovered for a data root's T1 episodes recorded without model ids
    (scripts/lang/recover_model_ids.py writes <shard>/item_buffer/model_ids.json; root = the shard or its
    item_buffer/); {} when absent."""
    import json

    p = Path(root) / "model_ids.json"
    if not p.exists():
        p = Path(root) / "item_buffer" / "model_ids.json"   # a shard root instead of its item_buffer/
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    # primitives (prim_box, prim_tall_cylinder) have no model id: None, named by their category
    return {int(e): {i: (v["asset"], None if v.get("model_id") is None else int(v["model_id"])) for i, v in items.items()}
            for e, items in d["episodes"].items()}


def refresh_record(record: dict, model_ids: dict | None = None, objects_cfg: dict | None = None,
                   bank: dict | None = None) -> dict:
    """Re-name a stored episode record with the current objects.yaml / paraphrase bank (in place): an episode stores
    the words it was collected with, so a names fix reaches old data only through this. model_ids: {item_id:
    (asset, model_id)} for T1 objects the record lacks them for (load_model_ids). Recomputes the bead words of particle
    rules and every rule's sampleable flag."""
    cfg = objects_cfg if objects_cfg is not None else _objects_cfg()
    model_ids = model_ids or {}
    # the regions' words too (episode_record stores them from objects.yaml regions): a record collected before a
    # region rename (2026-10-20: zone / marked area -> mat, bin / box -> bin, plate / dish / saucer -> plate) keeps
    # its old words otherwise
    regions_cfg = cfg.get("regions", {}) or {}
    for r in record.get("regions") or []:
        key, entry = _region_key(r["region_id"], regions_cfg)
        r["kind"], r["names"] = key, _words(entry) or [key.replace("_", " ")]
    for o in record["objects"]:
        asset, mid = model_ids.get(o["item_id"], (o.get("asset"), o.get("model_id")))
        _name_object(o, cfg, record.get("task"), asset, mid)
    for entry in record["rules"]:
        subj = entry.get("subject") or {}
        if "particles" in subj:
            entry.setdefault("params", {})["material"] = materials(record["objects"][subj["particles"]], bank=bank)
        entry["sampleable"] = sampleable(entry, record, bank)
    record["names_version"] = NAMES_VERSION
    return record


def sampleable(rule: dict, scene: dict, bank: dict | None = None) -> bool:
    """A rule the training sampler may draw: a training-split, single-referent
    rule with at least one training sentence in this scene."""
    if rule.get("split", "train") != "train" or rule.get("ambiguous") or rule.get("set"):
        return False
    return bool(sentences(rule, scene, "train", bank))


def record_from_meta(meta: dict, episode_id: int = 0, objects_cfg: dict | None = None) -> dict:
    """The episode record of a saved runner meta.json (probe / calibration
    runs: logger's item / region / rule dicts), e.g. to build a text cache
    from runs on disk."""
    from types import SimpleNamespace

    from ..rules import RuleSpec

    items = [SimpleNamespace(item_id=d["item_id"], category=d["category"], color=d.get("color"),
                             size=d.get("size"), role=d.get("role"), kind=d.get("kind", "rigid"))
             for d in meta["items"]]
    regions = [SimpleNamespace(region_id=d["region_id"], color=d.get("color"), role=d.get("role"),
                               nameable=d.get("nameable", True)) for d in meta["regions"]]
    rules = [RuleSpec(rule_id=d["rule_id"], consequence=d["consequence"], item_ids=d["item_ids"],
                      region_ids=d["region_ids"], params=d["params"], phrases=d.get("phrases", []),
                      split=d.get("split", "train")) for d in meta["rules"]]
    for r, d in zip(rules, meta["rules"]):
        r.ambiguous = bool(d.get("ambiguous", False))
    runner = SimpleNamespace(task_name=meta.get("task"), host=meta.get("host_task"), embodiment=meta.get("embodiment"),
                             seed=meta.get("seed"), items=items, regions=regions, rules=rules,
                             active_rule_id=meta.get("active_rule_id") or (meta.get("diag") or {}).get("primary_rule"))
    return episode_record(runner, episode_id, objects_cfg)
