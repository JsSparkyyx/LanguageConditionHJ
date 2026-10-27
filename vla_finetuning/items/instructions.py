"""Plain task instructions of the item-scene demonstrations (Yuxuan 2026-10-27: the instruction is the task only, no
constraint sentence), in RoboTwin's template format and made by RoboTwin's own generator helpers, so the files read
like the collision-avoidance set's ({"seen": [...], "unseen": [...]}, 100 strings each, seeded by the demo seed):

    task               templates                                placeholders (item_instruction_info)
    selective_avoid    the host's RoboTwin templates            common.instruction_info(task, host)
    push_zone          task_instruction/push_zone.json          {A} "the red box"
    tray_transport     task_instruction/tray_transport.json     {A} "the tray"
    clean_pour         task_instruction/clean_pour.json         {A} "the white bowl" (the target), {B} "the mug"
    gap_place          task_instruction/gap_place.json          {A} "the wooden block", {B} {C} "the red block" and
                                                                "the blue block" (the two flankers, in an order drawn
                                                                from the seed)

The placeholders come from the scene alone, so they are built after the runner's setup() and before run() (run()
closes the env). T1's lists are exactly the collision-avoidance set's: common.make_instructions(host,
common.instruction_info(task, host), seed), unfiltered (only the empty-string and leftover-placeholder guards), so
they keep the hosts' RoboTwin object descriptions as they are, held-out words of the language filter included
("rectangular bright orange serving tray", "red container with yellow fries": place_burger_fries). The item tasks'
own templates hold no held-out word (paraphrases.yaml heldout_words: "container", "serving tray", ...); a string with
one is dropped there and the list topped up. HoloBrain trains on seen AND unseen (robotwin_lmdb_dataset joins them),
so "unseen" is not held out.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from functools import lru_cache
from pathlib import Path

from .. import common as C

ITEMS = Path(__file__).resolve().parent
TEMPLATE_DIR = ITEMS / "task_instruction"
T1 = "selective_avoid"
ITEM_TASKS = ("selective_avoid", "push_zone", "tray_transport", "clean_pour", "gap_place")   # T1, T3, T5, T6, T7
TEMPLATE_TASKS = ITEM_TASKS[1:]
KEYS = ("seen", "unseen")
PARAPHRASES = C.REPO / "envs" / "main" / "item_envs" / "language" / "paraphrases.yaml"
OVERDRAW = 4                # strings drawn per string kept when the held-out filter dropped some
AMBIGUOUS_T1 = {"stack_blocks_two": ("red", "green")}   # its sentences say "red block" / "green block"


# ------------------------------------------------------------------ RoboTwin's generator ----
def robotwin_root() -> Path:
    """RoboTwin's root: $ROBOTWIN_ROOT, else beside the checkout (main/envs/env.py), else the shared copy."""
    cands = [os.environ.get("ROBOTWIN_ROOT"), C.REPO / "RoboTwin", C.REPO.parent / "RoboTwin",
             "/work/nvme/bifq/project/RoboTwin"]
    for c in cands:
        if c and (Path(c) / "description" / "utils" / "generate_episode_instructions.py").is_file():
            return Path(c)
    raise FileNotFoundError(f"RoboTwin's description/ not found in {[str(c) for c in cands if c]}")


def generator():
    """RoboTwin's description/utils/generate_episode_instructions module (Env's prepare() puts RoboTwin's root on
    sys.path; CPU tests need it here)."""
    root = str(robotwin_root())
    if root not in sys.path:
        sys.path.append(root)
    import description.utils.generate_episode_instructions as gen

    return gen


# ------------------------------------------------------------------ the language filter's held-out words ----
@lru_cache(maxsize=1)
def heldout_words() -> tuple[str, ...]:
    import yaml

    words = (yaml.safe_load(PARAPHRASES.read_text()) or {}).get("heldout_words") or []
    if not words:
        raise ValueError(f"{PARAPHRASES}: no heldout_words")
    return tuple(str(w) for w in words)


def has_heldout_word(text: str) -> str | None:
    """The first held-out word in ``text`` (whole words, plurals too), else None."""
    for w in heldout_words():
        if re.search(rf"\b{re.escape(w)}(s|es)?\b", text, flags=re.IGNORECASE):
            return w
    return None


# ------------------------------------------------------------------ templates ----
def template_path(task: str, host: str | None = None) -> Path:
    if task == T1:
        if not host:
            raise ValueError("selective_avoid: the host's templates need the host")
        return robotwin_root() / "description" / "task_instruction" / f"{host}.json"
    if task not in TEMPLATE_TASKS:
        raise KeyError(task)
    return TEMPLATE_DIR / f"{task}.json"


def load_templates(task: str) -> dict:
    data = json.loads(template_path(task).read_text())
    for key in KEYS:
        if not isinstance(data.get(key), list) or not data[key]:
            raise ValueError(f"{task}: template file has no {key!r} list")
    return data


def instruction_md5(task: str, host: str | None = None) -> str:
    """Hash of what decides the text (pinned in settings.json): the template file, and for the item tasks the held-out
    words (T1's lists are unfiltered)."""
    h = hashlib.md5(template_path(task, host).read_bytes())
    if task != T1:
        h.update("\n".join(sorted(heldout_words())).encode())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ placeholders ----
def item_instruction_info(task: str, task_obj, host: str | None = None, seed: int = 0) -> dict:
    """The template placeholders, from the built scene (load_actors attributes; call before run())."""
    t = task_obj
    if task == T1:
        return C.instruction_info(t, host)
    if task == "push_zone":
        color = (t.push_objects[0] or {}).get("color")
        if not isinstance(color, str) or not color:
            raise ValueError(f"push_zone: the push object has no colour word ({color!r})")
        return {"{A}": f"the {color} box"}
    if task == "tray_transport":
        return {"{A}": "the tray"}
    if task == "clean_pour":
        if getattr(t, "src_kind", None) != "mug":   # ur5-wsg pours from the mug (granular_pour SRC_CONTAINER)
            raise ValueError(f"clean_pour: source {getattr(t, 'src_kind', None)!r}, the templates say the mug")
        return {"{A}": f"the {t.target_color} bowl", "{B}": "the mug"}
    if task == "gap_place":
        flankers = [b["color"] for b in t.standing if b.get("role_in_slot") == "flanker"]
        if len(flankers) != 2 or len(set(flankers)) != 2:
            raise ValueError(f"gap_place: flanker colours {flankers}, want two distinct")
        if random.Random(int(seed)).random() < 0.5:   # which flanker is named first: the seed's, not the layout's
            flankers.reverse()
        return {"{A}": "the wooden block", "{B}": f"the {flankers[0]} block", "{C}": f"the {flankers[1]} block"}
    raise KeyError(task)


def ambiguous_objects(host: str | None, items) -> list[str]:
    """T1 extra objects a host sentence would also name: on stack_blocks_two, a red or green box ("red block")."""
    colors = AMBIGUOUS_T1.get(host or "")
    if not colors:
        return []
    return [s.item_id for s in items if getattr(s, "role", None) == "candidate"
            and getattr(s, "category", None) in ("box", "block")
            and str(getattr(s, "color", "") or "").lower() in colors]


# ------------------------------------------------------------------ generation ----
def _generate(templates: dict, info: dict, n: int) -> dict:
    """RoboTwin's generate_episode_descriptions loop on our template dict (it reads task_instruction/<task>.json from
    RoboTwin's own tree): templates whose placeholders match info, shuffled, cycled until n of each."""
    gen = generator()
    seen = gen.filter_instructions(list(templates["seen"]), info)      # shuffles its argument: pass a copy
    unseen = gen.filter_instructions(list(templates["unseen"]), info)
    out = {}
    for key, temps, fill in (("seen", seen, gen.replace_placeholders),
                             ("unseen", unseen, gen.replace_placeholders_unseen)):
        out[key] = []
        while temps and len(out[key]) < n:
            for tpl in temps:
                if len(out[key]) >= n:
                    break
                out[key].append(fill(tpl, info))
    return out


def _draw(task: str, info: dict, seed: int, n: int, host: str | None) -> dict:
    if task == T1:
        if not host:
            raise ValueError("selective_avoid: the host's templates need the host")
        generator()                             # RoboTwin's root on sys.path for common.make_instructions' import
        try:
            return C.make_instructions(host, info, seed, n)
        except SystemExit as exc:               # RoboTwin's generator exit()s on a missing description file
            raise RuntimeError(f"instruction generator exited: {exc}") from exc
    for k, v in info.items():
        if "/" in str(v) or "\\" in str(v):     # RoboTwin would read it as a description file
            raise ValueError(f"{task}: placeholder {k} = {v!r} looks like a path")
    templates = load_templates(task)
    state = random.getstate()
    random.seed(int(seed))
    try:
        return _generate(templates, info, n)
    finally:
        random.setstate(state)


def make_item_instructions(task: str, info: dict, seed: int, n: int = 100, host: str | None = None) -> dict:
    """{"seen": [n strings], "unseen": [n strings]} for one demo. T1: common.make_instructions(host, info, seed, n) as it
    comes (the collision-avoidance set's lists). The item tasks: strings with a held-out word are dropped, drawn again
    with OVERDRAW x n, the first n clean ones kept (cycled if fewer)."""
    out = _draw(task, info, seed, n, host)
    heldout = task != T1
    if heldout and any(has_heldout_word(s) for key in KEYS for s in out.get(key, [])):
        big = _draw(task, info, seed, OVERDRAW * n, host)
        out = {}
        for key in KEYS:
            clean = [s for s in big.get(key, []) if not has_heldout_word(s)]
            out[key] = (clean * (n // max(len(clean), 1) + 1))[:n]
    out = {key: list(out.get(key, [])) for key in KEYS}
    check_instructions(out, heldout=heldout)
    return out


def check_instructions(instr: dict, heldout: bool = True) -> None:
    """Raise ValueError unless both lists exist and every string is non-empty and filled (no "{" / "}" left), and with
    ``heldout`` (the item tasks' own templates; not T1's RoboTwin lists) free of held-out words."""
    for key in KEYS:
        lst = instr.get(key)
        if not isinstance(lst, list) or not lst:
            raise ValueError(f"instructions: {key!r} missing or empty")
        for s in lst:
            if not isinstance(s, str) or not s.strip():
                raise ValueError(f"instructions: empty {key} string {s!r}")
            if "{" in s or "}" in s:
                raise ValueError(f"instructions: unfilled placeholder in {s!r}")
            w = has_heldout_word(s) if heldout else None
            if w:
                raise ValueError(f"instructions: held-out word {w!r} in {s!r}")
