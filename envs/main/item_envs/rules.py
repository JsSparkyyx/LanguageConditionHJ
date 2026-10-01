"""RuleSpec, candidate rule enumeration, phrase generation, split assignment.

Templates in task_cfg["rules"] look like:
    {"consequence": "upright", "items": {"role": "constrained_candidate", "tippable": true},
     "theta_max_deg": 45}
    {"consequence": "inside",  "items": {"role": "constrained_candidate"}, "regions": ["tray_top"]}
One RuleSpec is produced per (template, matching item, matching region).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Consequence(str, Enum):
    INSIDE = "inside"  # item footprint stays inside region polygon (and on its surface if kind=="surface")
    OUTSIDE = "outside"  # item footprint stays outside region polygon
    UPRIGHT = "upright"  # tilt(item) <= theta_max
    STAY = "stay"  # item stays within d_tol of its pose at rule activation
    PARTICLES_INSIDE = "particles_inside"  # every particle stays in the column above one of the regions
    PARTICLES_OUTSIDE = "particles_outside"  # no particle enters the column above any of the regions


@dataclass
class RuleSpec:
    rule_id: str  # stable string, e.g. "upright|bottle_0|-"
    consequence: Consequence
    item_id: str
    region_ids: tuple[str, ...]  # empty for UPRIGHT and STAY
    params: dict  # {"theta_max": rad} | {"d_tol": m, "theta_tol": rad} | {"T": m}
    phrases: list[str] = field(default_factory=list)
    split: str = "train"  # "train" | "heldout_item" | "heldout_region" | "heldout_phrase"


# Default phrase templates per consequence. Never contain digits.
TEMPLATES: dict[Consequence, list[str]] = {
    Consequence.INSIDE: ["keep the {item} on the {region}", "don't let the {item} fall off the {region}"],
    Consequence.OUTSIDE: ["keep the {item} out of the {region}", "don't let the {item} get into the {region}"],
    Consequence.UPRIGHT: ["don't knock over the {item}", "keep the {item} upright"],
    Consequence.STAY: ["don't disturb the {item}", "leave the {item} where it is"],
    Consequence.PARTICLES_INSIDE: ["don't spill the {item}", "keep the {item} in the {region}"],
    Consequence.PARTICLES_OUTSIDE: ["keep the {item} off the {region}", "don't get any {item} on the {region}"],
}

# Templates for consequences that name regions.
_REGIONAL = {Consequence.INSIDE, Consequence.OUTSIDE, Consequence.PARTICLES_INSIDE, Consequence.PARTICLES_OUTSIDE}
# Consequences whose regions together form ONE rule (vs one rule per region).
_REGION_SET = {Consequence.PARTICLES_INSIDE, Consequence.PARTICLES_OUTSIDE}

_TEMPLATE_RESERVED = {"consequence", "items", "regions"}


def enumerate_rules(items, regions, task_cfg, *, phrases_cfg=None, splits_cfg=None, rng=None) -> list[RuleSpec]:
    """One RuleSpec per (template, matching item, matching region).

    items / regions: mappings id -> spec (or iterables of specs). Item matching
    is by template["items"] key/value pairs against item attributes (role plus
    arbitrary attributes such as tippable). Template keys other than
    consequence/items/regions go into RuleSpec.params; *_deg keys are converted
    to radians. params also carry the item's "category" and "color" so
    render_phrases / assign_split work from the rule alone.

    When phrases_cfg is given, rule.phrases are rendered (n=3); rng defaults to
    a per-rule deterministic stream. split comes from assign_split when
    splits_cfg is given, else "train".
    """
    item_map = _as_id_map(items, "item_id")
    region_map = _as_id_map(regions, "region_id")
    rules: list[RuleSpec] = []
    for template in task_cfg.get("rules", []) or []:
        consequence = Consequence(template["consequence"])
        match = template.get("items", {}) or {}
        matched_items = [it for iid, it in sorted(item_map.items()) if _matches(it, match)]
        matched_regions = [region_map[rid] for rid in template.get("regions", []) or [] if rid in region_map]
        params_base = _template_params(template, consequence)
        for item in matched_items:
            params = {**params_base, "category": item.category}
            if getattr(item, "color", None):
                params["color"] = item.color
            if consequence in _REGION_SET:
                groups = [tuple(rid for rid in template.get("regions", []) or [] if rid in region_map)]
                groups = [g for g in groups if g]
            elif consequence in _REGIONAL:
                groups = [(r.region_id,) for r in matched_regions]
            else:
                groups = [()]
            for region_ids in groups:
                rule_id = f"{consequence.value}|{item.item_id}|{'+'.join(region_ids) if region_ids else '-'}"
                rule = RuleSpec(
                    rule_id=rule_id,
                    consequence=consequence,
                    item_id=item.item_id,
                    region_ids=tuple(region_ids),
                    params=params,
                )
                if phrases_cfg is not None:
                    rule_rng = rng if rng is not None else _rng_for(rule_id)
                    rule.phrases = render_phrases(rule, phrases_cfg, rule_rng)
                if splits_cfg is not None:
                    rule.split = assign_split(rule, splits_cfg)
                rules.append(rule)
    return rules


def render_phrases(rule: RuleSpec, phrases_cfg: dict, rng: np.random.RandomState, n: int = 3) -> list[str]:
    """Up to n natural-language variants for a rule. Never contain digits.

    Templates come from phrases_cfg["templates"][<consequence>] when present,
    else the module defaults. {item} draws on the item category's names and
    synonyms (with optional color prefix), {region} on the regions' names.
    """
    consequence = Consequence(rule.consequence)
    templates = (phrases_cfg.get("templates", {}) or {}).get(consequence.value) or TEMPLATES[consequence]
    if not rule.region_ids:
        templates = [t for t in templates if "{region}" not in t]
    templates = [t for t in templates if not _has_digit(t)]
    item_names = _item_names(rule, phrases_cfg)
    region_names = [_region_names(rid, phrases_cfg) for rid in rule.region_ids]
    phrases: list[str] = []
    attempts = 0
    while len(phrases) < n and attempts < 20 * max(n, 1) and templates:
        attempts += 1
        template = templates[int(rng.randint(len(templates)))]
        phrase = template.replace("{item}", item_names[int(rng.randint(len(item_names)))])
        if "{region}" in template:
            candidates = [nm for names in region_names for nm in names] if region_names else []
            if not candidates:
                continue
            phrase = phrase.replace("{region}", candidates[int(rng.randint(len(candidates)))])
        if not _has_digit(phrase) and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def assign_split(rule: RuleSpec, splits_cfg: dict) -> str:
    """Split for a rule from the task's section of splits.yaml.

    Precedence: heldout_item (category or color) > heldout_region >
    heldout_phrase > "train". Category/color match against the item's injected
    params and fall back to tokens of the item_id ("-" normalized to "_").
    heldout_phrase matches when a rule phrase could have been rendered from a
    heldout template ({item}/{region} are wildcards).
    """
    splits_cfg = splits_cfg or {}
    category = str(rule.params.get("category") or _strip_item_index(rule.item_id))
    tokens = {t.replace("-", "_") for t in rule.item_id.split("_")} | {category, category.replace("-", "_")}
    color = rule.params.get("color")
    heldout_categories = set(splits_cfg.get("heldout_item_categories", []) or [])
    if heldout_categories & tokens:
        return "heldout_item"
    heldout_colors = set(splits_cfg.get("heldout_colors", []) or [])
    if (color is not None and color in heldout_colors) or (heldout_colors & tokens):
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


def _template_params(template: dict, consequence: Consequence) -> dict:
    """Margin params from a template: *_deg converted to radians, the rest raw SI."""
    params: dict = {}
    for key, value in template.items():
        if key in _TEMPLATE_RESERVED:
            continue
        if key.endswith("_deg"):
            params[key[: -len("_deg")]] = float(np.deg2rad(value))
        else:
            params[key] = value
    if consequence is Consequence.UPRIGHT and "theta_max" not in params:
        raise ValueError(f"upright template needs theta_max_deg: {template}")
    if consequence is Consequence.STAY:
        params.setdefault("theta_tol", 0.35)
    if consequence in _REGION_SET:
        params.setdefault("T", 0.005)
    return params


def _matches(item, match: dict) -> bool:
    """All key/value pairs match item attributes; a missing attribute never matches."""
    for key, value in match.items():
        if getattr(item, key, None) != value:
            return False
    return True


def _item_names(rule: RuleSpec, phrases_cfg: dict) -> list[str]:
    """Candidate spoken names for the rule's item (color prefix mixed in)."""
    category = str(rule.params.get("category") or _strip_item_index(rule.item_id))
    entry = (phrases_cfg.get("items", {}) or {}).get(category, {}) or {}
    names = [str(nm) for nm in list(entry.get("names", [])) + list(entry.get("synonyms", []))]
    color = rule.params.get("color")
    if color:
        names = [f"{color} {nm}" for nm in names] + names
    names = [nm for nm in names if not _has_digit(nm)]
    if not names:
        fallback = re.sub(r"[\d_]+", " ", category).strip()
        names = [fallback if fallback else "object"]
    return names


def _region_names(region_id: str, phrases_cfg: dict) -> list[str]:
    """Candidate spoken names for a region; strips trailing tokens (e.g. a color
    like in "zone_red") until the id matches a phrases.yaml regions key."""
    regions_cfg = phrases_cfg.get("regions", {}) or {}
    colors = set(phrases_cfg.get("colors", []) or [])
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
