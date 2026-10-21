"""RuleSpec, candidate rule enumeration, phrase generation, split assignment.

Phrase templates live here (PLAN v3 §6), not in a YAML. Task-YAML rule entries
(PLAN v3 §9) look like:

    {consequence: avoid, items: {role: candidate}, margin: 0.0}
    {consequence: avoid, items: {role: candidate, same_category_min: 2}, set_rule: true, margin: 0.0}
    {consequence: avoid, items: {role: candidate}, set_rule: all, margin: 0.0}
    {consequence: inside, items: {role: candidate, group: push}, regions: [table_top]}
    {consequence: outside, items: {role: candidate, group: push}, regions: {kind: zone}}
    {consequence: upright, items: {role: bystander}, theta_max_deg: 45}
    {consequence: stay, items: {role: bystander}, d_tol: 0.03, theta_tol_deg: 20}
    {consequence: outside, items: {role: candidate, group: push}, regions: {kind: zone}, region_set: pairs}
    {consequence: stay, items: {role: bystander}, set_rule: pairs, d_tol: 0.03, theta_tol_deg: 20}

Item selectors match key/value pairs against item attributes; the directive
same_category_min keeps only items whose category has at least that many
members. set_rule: true emits one rule per qualifying category (item_ids = all
members), set_rule: all a single rule over every matched item, set_rule: pairs one rule
per 2-combination of matched items (margin = min over the pair).
region_set: pairs (inside / outside) emits one rule per 2-combination of the
matched regions ("keep the box out of the red zone and the blue zone",
outside margin = min over the regions, inside = max). Region
selectors are a list of region ids or a dict matched against region
attributes (e.g. {kind: zone} expands to every zone region).
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Consequence(str, Enum):
    AVOID = "avoid"  # robot links and held payload keep distance >= margin from the item
    INSIDE = "inside"  # item footprint stays inside a region (on its surface for kind == "surface")
    OUTSIDE = "outside"  # item footprint stays outside a region
    UPRIGHT = "upright"  # tilt(item) <= theta_max
    STAY = "stay"  # item stays within d_tol (and theta_tol) of its pose at rule activation
    PARTICLES_INSIDE = "particles_inside"  # every particle stays in the column above one of the regions
    PARTICLES_OUTSIDE = "particles_outside"  # no particle enters the column above any of the regions


@dataclass
class RuleSpec:
    rule_id: str  # stable string, e.g. "avoid|obj_2|-", "outside|box_0|zone_red"
    consequence: Consequence
    item_ids: tuple[str, ...]  # usually one item. Several items = set rule ("avoid every cup"), margin = min
    region_ids: tuple[str, ...]  # empty for AVOID, UPRIGHT, STAY
    params: dict  # {"margin": m} | {"theta_max": rad} | {"d_tol": m, "theta_tol": rad} | {"T": m}
    phrases: list[str] = field(default_factory=list)  # training phrases (never from a held-out template)
    # evaluation-only phrases from the held-out templates (splits.yaml
    # heldout_phrase_templates): the rule stays in its split, only these
    # wordings are kept out of training (user decision 2026-10-05; before, one
    # held-out phrase among three held out the WHOLE rule — 55 % of T6's)
    heldout_phrases: list[str] = field(default_factory=list)
    split: str = "train"  # "train" | "heldout_item" | "heldout_region" | "heldout_phrase" (only when every template is held out)
    ambiguous: bool = False  # single-item rule whose item no name tells apart in this scene (never a conditioning rule)

    def __post_init__(self):
        self.consequence = Consequence(self.consequence)
        self.item_ids = tuple(self.item_ids)
        self.region_ids = tuple(self.region_ids)


# Phrase templates per consequence (PLAN v3 §6). Never contain digits.
TEMPLATES: dict[Consequence, list[str]] = {
    # the paraphrase bank's sentences in {item} / {region} form (language/paraphrases.yaml, 2026-10-20: five plain
    # sentences per rule, the last one held out); INSIDE (keep the box on the table) is never named
    Consequence.AVOID: ["don't touch the {item}", "avoid the {item}", "keep away from the {item}", "don't hit the {item}",
                        "don't bump into the {item}"],
    Consequence.INSIDE: [],
    Consequence.OUTSIDE: ["keep the {item} out of the {region}", "don't push the {item} onto the {region}",
                          "don't let the {item} go into the {region}", "keep the {item} off the {region}",
                          "don't let the {item} touch the {region}"],
    Consequence.UPRIGHT: ["don't knock over the {item}", "keep the {item} upright", "don't tip over the {item}",
                          "don't let the {item} fall over", "make sure the {item} stays standing"],
    Consequence.STAY: ["don't move the {item}", "leave the {item} where it is", "keep the {item} where it is",
                       "make sure the {item} stays in place", "don't let the {item} move"],
    Consequence.PARTICLES_INSIDE: ["don't spill the {item}", "don't spill any {item}", "don't get any {item} on the table",
                                   "don't drop any {item}", "no {item} on the table"],
    Consequence.PARTICLES_OUTSIDE: ["keep the {item} out of the {region}", "don't spill any {item} into the {region}",
                                    "don't drop any {item} into the {region}", "don't let any {item} land in the {region}",
                                    "no {item} in the {region}"],
}

# Templates for set rules (item_ids names several items). {category} is filled
# from the shared category's names/synonyms, or a generic word for mixed sets.
SET_TEMPLATES: dict[Consequence, list[str]] = {
    Consequence.AVOID: ["don't touch any {category}", "avoid all the {category}s"],
    # {items}: every member's spoken name, "the red cylinder and the cup"
    Consequence.STAY: ["don't move the {items}", "leave the {items} where they are"],
}

# Fallback names when a category is missing from objects.yaml or a set mixes categories.
_GENERIC_NAMES = ["object", "item"]

# Consequences that name regions.
_REGIONAL = {Consequence.INSIDE, Consequence.OUTSIDE, Consequence.PARTICLES_INSIDE, Consequence.PARTICLES_OUTSIDE}
# Consequences whose regions together form ONE rule (vs one rule per region).
_REGION_SET = {Consequence.PARTICLES_INSIDE, Consequence.PARTICLES_OUTSIDE}

_TEMPLATE_RESERVED = {"consequence", "items", "regions", "set_rule", "region_set"}


def enumerate_rules(items, regions, task_cfg, *, objects_cfg=None, splits_cfg=None, rng=None) -> list[RuleSpec]:
    """RuleSpecs from the task_cfg["rules"] entries.

    items / regions: mappings id -> spec (or iterables of specs). Enumeration
    is deterministic: entries are processed in order, items sorted by item_id,
    regions by region_id. params carry the entry's margin knobs (*_deg keys
    converted to radians) plus naming metadata for render_phrases /
    assign_split: "categories"/"colors" (per member, aligned with item_ids)
    and, when uniform across members, "category"/"color".

    When objects_cfg is given, rule.phrases are rendered (n=3); rng defaults
    to a per-rule deterministic stream. split comes from assign_split when
    splits_cfg is given (a task section merged with "all", see
    resolve_splits), else "train".
    """
    item_map = _as_id_map(items, "item_id")
    region_map = _as_id_map(regions, "region_id")
    # every name each item could be called in this scene (colored + bare):
    # a rule's phrases must not use a name another item also answers to
    # ("don't knock over the brick" with two blocks on the tray named no
    # referent, T5 audit 2026-10-04)
    scene_names = {iid: _names_for(it, objects_cfg) for iid, it in item_map.items()} if objects_cfg else {}
    # same for regions: with two zones, "keep the box out of the zone" names
    # neither (the bare kind word); colored names survive
    region_scene_names = {rid: set(_region_names(rid, objects_cfg)) for rid in region_map} if objects_cfg else {}
    rules: list[RuleSpec] = []
    for entry in task_cfg.get("rules", []) or []:
        consequence = Consequence(entry["consequence"])
        item_groups = _item_groups(entry, item_map)
        region_groups = _region_groups(entry, region_map, consequence)
        params_base = _entry_params(entry, consequence)
        for members in item_groups:
            params = {**params_base, **_naming_meta(members)}
            for region_ids in region_groups:
                rule_id = _rule_id(consequence, [it.item_id for it in members], region_ids)
                rule = RuleSpec(
                    rule_id=rule_id,
                    consequence=consequence,
                    item_ids=tuple(it.item_id for it in members),
                    region_ids=tuple(region_ids),
                    params=params,
                )
                if objects_cfg is not None:
                    rule_rng = rng if rng is not None else _rng_for(rule_id)
                    others = set().union(*[nm for iid, nm in scene_names.items() if iid not in rule.item_ids] or [set()])
                    others_r = set().union(*[nm for rid, nm in region_scene_names.items()
                                             if rid not in rule.region_ids] or [set()])
                    held_t = list((splits_cfg or {}).get("heldout_phrase_templates", []) or [])
                    rule.phrases = render_phrases(rule, objects_cfg, rule_rng, exclude_names=others,
                                                  exclude_region_names=others_r, exclude_templates=held_t)
                    if held_t:
                        rule.heldout_phrases = render_phrases(
                            rule, objects_cfg, rule_rng, n=2, exclude_names=others,
                            exclude_region_names=others_r, only_templates=held_t)
                    if not rule.phrases:  # every template is held out: the whole rule is
                        rule.phrases, rule.heldout_phrases = rule.heldout_phrases, []
                    if len(rule.item_ids) == 1:
                        own = _item_names(rule, objects_cfg)
                        rule.ambiguous = not any(nm not in others for nm in own)
                if splits_cfg is not None:
                    rule.split = assign_split(rule, splits_cfg)
                rules.append(rule)
    return rules


def render_phrases(rule: RuleSpec, objects_cfg: dict, rng: np.random.RandomState, n: int = 3,
                   exclude_names=(), exclude_region_names=(), exclude_templates=(), only_templates=None) -> list[str]:
    """Up to n natural-language variants for a rule. Never contain digits.

    Templates come from the module-level TEMPLATES (SET_TEMPLATES for set
    rules). {item}/{category} draw on the category's names and synonyms in
    objects.yaml (with a color prefix when the item has a color), {region} on
    the regions' names (with a color prefix from the region id, e.g.
    "zone_red" -> "red zone").
    """
    objects_cfg = objects_cfg or {}
    consequence = Consequence(rule.consequence)
    is_set = len(rule.item_ids) > 1
    templates = list((SET_TEMPLATES.get(consequence) if is_set else None) or TEMPLATES[consequence])
    if not rule.region_ids:
        templates = [t for t in templates if "{region}" not in t]
    templates = [t for t in templates if not _has_digit(t)]
    # held-out wordings (splits.yaml heldout_phrase_templates): training
    # phrases skip them, the evaluation phrases use only them
    templates = [t for t in templates if t not in set(exclude_templates or ())]
    if only_templates is not None:
        templates = [t for t in templates if t in set(only_templates)]
    exclude_names = set(exclude_names or ())
    item_names = _train_words(_unambiguous(_item_names(rule, objects_cfg), exclude_names))
    category_names = _train_words(_category_names(rule, objects_cfg))
    exclude_region_names = set(exclude_region_names or ())
    region_names = _train_words(_unambiguous([nm for rid in rule.region_ids for nm in _region_names(rid, objects_cfg)],
                                             exclude_region_names))
    # several regions in ONE per-region-kind rule (region_set: pairs): every
    # phrase names them all ("the red zone and the blue zone")
    joint_regions = len(rule.region_ids) > 1 and consequence not in _REGION_SET
    region_lists = [_colored_first(_unambiguous(_region_names(rid, objects_cfg), exclude_region_names), objects_cfg)
                    for rid in rule.region_ids]
    member_lists = [_unambiguous(_member_names(rule, objects_cfg, k), exclude_names)
                    for k in range(len(rule.item_ids))] if is_set else []
    phrases: list[str] = []
    attempts = 0
    while len(phrases) < n and attempts < 20 * max(n, 1) and templates:
        attempts += 1
        template = templates[int(rng.randint(len(templates)))]
        if "{item}" in template:
            template = template.replace("{item}", item_names[int(rng.randint(len(item_names)))])
        if "{category}" in template:
            template = template.replace("{category}", category_names[int(rng.randint(len(category_names)))])
        if "{items}" in template:
            template = template.replace(
                "{items}", " and the ".join(names[int(rng.randint(len(names)))] for names in member_lists)
            )
        if "{region}" in template:
            if not region_names:
                continue
            if joint_regions:
                joined = " and the ".join(names[int(rng.randint(len(names)))] for names in region_lists)
                template = template.replace("{region}", joined)
            else:
                template = template.replace("{region}", region_names[int(rng.randint(len(region_names)))])
        if not _has_digit(template) and template not in phrases:
            phrases.append(template)
    return phrases


_HELDOUT_WORDS: set | None = None


def _train_words(names: list[str]) -> list[str]:
    """names without the relabeling bank's held-out referent words
    (language/paraphrases.yaml heldout_words: "the tin can", "the flask" are
    evaluation-only); all of them when nothing else is left."""
    global _HELDOUT_WORDS
    if _HELDOUT_WORDS is None:
        import yaml
        from pathlib import Path

        bank = yaml.safe_load((Path(__file__).resolve().parent / "language" / "paraphrases.yaml").read_text())
        _HELDOUT_WORDS = {str(w) for w in bank.get("heldout_words", []) or []}
    keep = [nm for nm in names if not any(nm == w or nm.endswith(" " + w) for w in _HELDOUT_WORDS)]
    return keep or names


def assign_split(rule: RuleSpec, splits_cfg: dict) -> str:
    """Split for a rule from the task's section of splits.yaml merged with "all".

    Precedence: heldout_item (any member category or color) > heldout_region >
    heldout_phrase > "train". Category/color match against the per-member
    naming metadata in params and fall back to tokens of each item_id ("-"
    normalized to "_"). heldout_phrase matches when a rule phrase could have
    been rendered from a heldout template ({item}/{region} are wildcards).
    """
    splits_cfg = splits_cfg or {}
    categories = [str(c) for c in (rule.params.get("categories") or [rule.params.get("category") or ""])]
    colors = [c for c in (rule.params.get("colors") or [rule.params.get("color")]) if c]
    tokens = set()
    for iid in rule.item_ids:
        tokens |= {t.replace("-", "_") for t in iid.split("_")}
    for cat in categories:
        tokens |= {cat, cat.replace("-", "_")}
    heldout_categories = set(splits_cfg.get("heldout_item_categories", []) or [])
    if heldout_categories & tokens:
        return "heldout_item"
    heldout_colors = set(splits_cfg.get("heldout_colors", []) or [])
    if heldout_colors & set(colors) or heldout_colors & tokens:
        return "heldout_item"
    heldout_regions = set(splits_cfg.get("heldout_regions", []) or [])
    for rid in rule.region_ids:
        if rid in heldout_regions or any(rid.startswith(h + "_") for h in heldout_regions):
            return "heldout_region"
        if {t.replace("-", "_") for t in rid.split("_")} & heldout_regions:
            return "heldout_region"
    for template in splits_cfg.get("heldout_phrase_templates", []) or []:
        pattern = re.escape(template)
        pattern = pattern.replace(re.escape("{item}"), ".+").replace(re.escape("{region}"), ".+")
        if any(re.fullmatch(pattern, phrase) for phrase in rule.phrases):
            return "heldout_phrase"
    return "train"


def resolve_splits(splits_yaml: dict, task: str) -> dict:
    """The task's split section merged over the shared "all" section."""
    splits_yaml = splits_yaml or {}
    return {**(splits_yaml.get("all") or {}), **(splits_yaml.get(task) or {})}


def _item_groups(entry: dict, item_map: dict) -> list[tuple]:
    """Matched items grouped per the entry's set_rule mode."""
    match = dict(entry.get("items", {}) or {})
    min_per_cat = match.pop("same_category_min", None)
    set_rule = entry.get("set_rule")
    if set_rule is True and min_per_cat is None:
        min_per_cat = 2  # a set rule over a singleton is not a set
    matched = [it for iid, it in sorted(item_map.items()) if _matches(it, match)]
    if min_per_cat is not None:
        counts = Counter(it.category for it in matched)
        matched = [it for it in matched if counts[it.category] >= int(min_per_cat)]
    if set_rule in (None, False):
        return [(it,) for it in matched]
    if set_rule is True:
        by_cat: dict[str, list] = {}
        for it in matched:
            by_cat.setdefault(it.category, []).append(it)
        return [tuple(by_cat[cat]) for cat in sorted(by_cat)]
    if set_rule == "all":
        return [tuple(matched)] if matched else []
    if set_rule == "pairs":
        return [tuple(pair) for pair in itertools.combinations(matched, 2)]
    raise ValueError(f"unknown set_rule mode: {set_rule!r}")


def _region_groups(entry: dict, region_map: dict, consequence: Consequence) -> list[tuple[str, ...]]:
    """Region id groups per consequence: one rule per region, or one joint rule."""
    if consequence not in _REGIONAL:
        return [()]
    if "regions" not in entry:
        raise ValueError(f"{consequence.value} rule entry needs a regions selector: {entry}")
    selector = entry["regions"]
    if isinstance(selector, dict):
        ids = [rid for rid, r in sorted(region_map.items()) if _matches(r, selector)]
    else:
        ids = [rid for rid in (selector or []) if rid in region_map]
    region_set = entry.get("region_set")
    if consequence in _REGION_SET and region_set != "each":
        return [tuple(ids)] if ids else []
    if region_set == "each":  # particle rules: one rule per region (T6: one per neighbour container)
        return [(rid,) for rid in ids]
    if region_set == "pairs":
        return [tuple(pair) for pair in itertools.combinations(ids, 2)]
    if region_set not in (None, False):
        raise ValueError(f"unknown region_set mode: {region_set!r}")
    return [(rid,) for rid in ids]


def _entry_params(entry: dict, consequence: Consequence) -> dict:
    """Margin params from an entry: *_deg converted to radians, the rest raw SI."""
    params: dict = {}
    for key, value in entry.items():
        if key in _TEMPLATE_RESERVED:
            continue
        if key.endswith("_deg"):
            params[key[: -len("_deg")]] = float(np.deg2rad(value))
        else:
            params[key] = value
    if consequence is Consequence.UPRIGHT and "theta_max" not in params:
        raise ValueError(f"upright entry needs theta_max_deg: {entry}")
    if consequence is Consequence.STAY:
        params.setdefault("theta_tol", 0.35)
    # bead rules: no default soft-min temperature (2026-10-07): the margin is the hard min over beads
    # unless an entry sets T itself
    return params


def _naming_meta(members: tuple) -> dict:
    """Per-member category/color lists (+ uniform singulars) for phrases and splits."""
    categories = tuple(str(it.category) for it in members)
    colors = tuple(getattr(it, "color", None) for it in members)
    meta: dict = {"categories": categories, "colors": colors}
    if len(set(categories)) == 1:
        meta["category"] = categories[0]
    if len(set(colors)) == 1 and colors[0]:
        meta["color"] = colors[0]
    return meta


def _rule_id(consequence: Consequence, item_ids: list[str], region_ids) -> str:
    items_part = "+".join(item_ids) if item_ids else "-"
    regions_part = "+".join(region_ids) if region_ids else "-"
    return f"{consequence.value}|{items_part}|{regions_part}"


def _matches(obj, match: dict) -> bool:
    """All key/value pairs match the object's attributes; a missing attribute never matches."""
    for key, value in match.items():
        if getattr(obj, key, None) != value:
            return False
    return True


def _category_entry(rule: RuleSpec, objects_cfg: dict) -> tuple[str, dict]:
    """The rule's shared category key and its objects.yaml categories entry (may be empty)."""
    category = str(rule.params.get("category") or _strip_item_index(rule.item_ids[0]))
    entry = (objects_cfg.get("categories", {}) or {}).get(category, {}) or {}
    return category, entry


def _names_for(item, objects_cfg: dict) -> set[str]:
    """Every spoken name an item answers to: its category's names and
    synonyms, bare and color-prefixed."""
    entry = (objects_cfg.get("categories", {}) or {}).get(str(getattr(item, "category", "")), {}) or {}
    names = {str(nm) for nm in list(entry.get("names", [])) + list(entry.get("synonyms", []))}
    color = getattr(item, "color", None)
    if color:
        names |= {f"{color} {nm}" for nm in names}
    return names


def _unambiguous(names: list[str], exclude: set) -> list[str]:
    """names minus those another item in the scene also answers to (distinct
    colors keep the color-prefixed names); when none survive — same
    category, no color — the input stands (an unavoidable ambiguity)."""
    keep = [nm for nm in names if nm not in exclude]
    return keep or names


def _item_names(rule: RuleSpec, objects_cfg: dict) -> list[str]:
    """Candidate spoken names for the rule's item (color prefix mixed in)."""
    category, entry = _category_entry(rule, objects_cfg)
    names = [str(nm) for nm in list(entry.get("names", [])) + list(entry.get("synonyms", []))]
    color = rule.params.get("color")
    if color:
        names = [f"{color} {nm}" for nm in names] + names
    names = [nm for nm in names if not _has_digit(nm)]
    if not names:
        fallback = re.sub(r"[\d_]+", " ", category).strip()
        names = [fallback] if fallback else list(_GENERIC_NAMES)
    return names


def _member_names(rule: RuleSpec, objects_cfg: dict, k: int) -> list[str]:
    """Spoken names for set-rule member k (its own category, color-prefixed
    when colored — a pair phrase must tell the members apart)."""
    categories = rule.params.get("categories") or ()
    colors = rule.params.get("colors") or ()
    category = str(categories[k]) if k < len(categories) else _strip_item_index(rule.item_ids[k])
    color = colors[k] if k < len(colors) else None
    entry = (objects_cfg.get("categories", {}) or {}).get(category, {}) or {}
    names = [str(nm) for nm in list(entry.get("names", [])) + list(entry.get("synonyms", []))]
    if color:
        names = [f"{color} {nm}" for nm in names]
    names = [nm for nm in names if not _has_digit(nm)]
    if not names:
        fallback = re.sub(r"[\d_]+", " ", category).strip()
        names = [fallback] if fallback else list(_GENERIC_NAMES)
    return names


def _colored_first(names: list[str], objects_cfg: dict) -> list[str]:
    """The color-prefixed names when there are any (pair phrases must name
    which zone), else all names."""
    colors = set(_all_colors(objects_cfg))
    colored = [nm for nm in names if nm.split(" ", 1)[0] in colors]
    return colored or names


def _category_names(rule: RuleSpec, objects_cfg: dict) -> list[str]:
    """Spoken names for a set rule's shared category; generic words for mixed sets."""
    if not rule.params.get("category"):
        return list(_GENERIC_NAMES)
    _, entry = _category_entry(rule, objects_cfg)
    names = [str(nm) for nm in list(entry.get("names", [])) + list(entry.get("synonyms", []))]
    names = [nm for nm in names if not _has_digit(nm)]
    return names or list(_GENERIC_NAMES)


def _region_names(region_id: str, objects_cfg: dict) -> list[str]:
    """Candidate spoken names for a region; strips trailing tokens (e.g. a color
    like in "zone_red") until the id matches an objects.yaml regions key."""
    regions_cfg = objects_cfg.get("regions", {}) or {}
    colors = set(_all_colors(objects_cfg))
    tokens = region_id.split("_")
    for end in range(len(tokens), 0, -1):
        key = "_".join(tokens[:end])
        entry = regions_cfg.get(key)
        if entry is None:
            continue
        names = [str(nm) for nm in list(entry.get("names", [])) + list(entry.get("synonyms", []))]
        prefix = [t for t in tokens[end:] if t in colors]
        if prefix:
            names = [f"{prefix[0]} {nm}" for nm in names] + names
        names = [nm for nm in names if not _has_digit(nm)]
        if names:
            return names
    return [re.sub(r"[\d_]+", " ", region_id).strip() or "region"]


def _all_colors(objects_cfg: dict) -> list[str]:
    """Training + heldout color words, from either a flat list or a train/heldout dict."""
    colors = objects_cfg.get("colors", []) or []
    if isinstance(colors, dict):
        return [str(c) for c in list(colors.get("train", []) or []) + list(colors.get("heldout", []) or [])]
    return [str(c) for c in colors]


def _strip_item_index(item_id: str) -> str:
    """"bottle_0" -> "bottle"; drops a trailing all-digit index token."""
    parts = item_id.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def _as_id_map(objs, id_attr: str) -> dict:
    if hasattr(objs, "values"):
        objs = objs.values()
    return {getattr(o, id_attr): o for o in objs}


def _rng_for(rule_id: str) -> np.random.RandomState:
    """Deterministic per-rule RNG (phrase rendering without a caller-provided rng)."""
    seed = int.from_bytes(hashlib.sha256(rule_id.encode()).digest()[:4], "little")
    return np.random.RandomState(seed)


def _has_digit(text: str) -> bool:
    return any(c.isdigit() for c in text)
