"""The VLM failure function's light half (VLM_FAILURE_PLAN, Yuxuan 2026-10-26): which VLMs, which rules get a prompt,
the prompt text of a rule, the answer parser and the label files. No torch: the data side (hsource.py) and the
config import it; the GPU labeler is vlm_label.py.

A label file, ``<root>/vlm_labels/<vlm>/<episode_id>.json`` (written once, atomically, by the labeler):

    {"episode_id", "vlm", "prompt_hash", "every", "n_rows", "ticks": [0, 5, ...],
     "rules": {rule_id: {"idx": column of h_rules, "referent": "the green box",
                         "score": [1, 1, -1, ...], "explanation": [...], "answer": [...], "parsed": [true, ...]}},
     "error": null}

``score[k]`` is the VLM's answer at tick ``ticks[k]`` (-1 failure, 0 uncertain, 1 safe), the h of that tick as it
is; hsource.py spreads it over the ticks nearest to it. An episode the labeler could not label (no frames, no
record) gets a file with ``error`` set and no rules: it is never trained on.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from pathlib import Path

from . import ensure_item_path

ensure_item_path()

REPO = Path(__file__).resolve().parents[2]
PROMPTS = Path(__file__).resolve().parent / "vlm_prompts.yaml"
# the VLMs of VLM/README.md (RoboBrain 2.0 32B is fine-tuned from Qwen2.5-VL-32B-Instruct). The
# weights are not in git (VLM/*/ is ignored): every checkout (the frozen worktrees too) reads the main project's copy
VLM_ROOT = Path(os.environ.get("VLM_ROOT", "/work/nvme/bifq/project/LanguageConditionHJ/VLM"))
VLM_FP8_ROOT = Path(os.environ.get("VLM_FP8_ROOT", "/work/hdd/bifq/VLM_fp8"))
VLMS = {
    "qwen25vl32b": VLM_ROOT / "Qwen2.5-VL-32B-Instruct",
    "robobrain2_32b": VLM_ROOT / "RoboBrain2.0-32B",
    # the smaller ones compared on 2026-10-27 (Yuxuan): Qwen2.5-VL 7B, Qwen3-VL 8B and RoboBrain 2.5 8B (fine-tuned from
    # Qwen3-VL; the -NV variant, trained on NVIDIA GPUs; -MT is the same model trained on Moore-Threads GPUs)
    "qwen25vl7b": VLM_ROOT / "Qwen2.5-VL-7B-Instruct",
    "qwen3vl8b": VLM_ROOT / "Qwen3-VL-8B-Instruct",
    "robobrain25_8b": VLM_ROOT / "RoboBrain2.5-8B-NV",
    # 4-bit versions of the two 32B models, to fit one A40 (Yuxuan 2026-10-27: try them first): Qwen's official AWQ
    # checkpoint, and RoboBrain2.0-32B quantized at load time (bitsandbytes NF4: BAAI publishes no 4-bit version).
    # Both run on the vLLM backend (vlm_label --backend vllm, the separate env /work/nvme/bifq/conda/envs/vlm)
    "qwen25vl32b_awq": VLM_ROOT / "Qwen2.5-VL-32B-Instruct-AWQ",
    "robobrain2_32b_nf4": VLM_ROOT / "RoboBrain2.0-32B",
    # 2026-10-27, after the 4-bit RoboBrain lost most of T1 / T5: the same bf16 weights on vLLM (separates the
    # backend from the quantization) and 8-bit weights (FP8, quantized at load time, W8A16 kernels on A40 / A100)
    "robobrain2_32b_vllm": VLM_ROOT / "RoboBrain2.0-32B",
    "robobrain2_32b_fp8": VLM_ROOT / "RoboBrain2.0-32B",
    "qwen25vl32b_fp8": VLM_ROOT / "Qwen2.5-VL-32B-Instruct",
    # 2026-10-28, for the 2-A40 training job (one GPU trains, one labels; Yuxuan): the same 8-bit weights saved once
    # (scripts/lang/vlm/to_fp8.py), since vLLM's own fp8 above loads the 64 GB bf16 model first and so needs two
    # GPUs. On /work/hdd (the nvme project space is near its quota); the checkpoint's config names its quantization
    "qwen25vl32b_fp8saved": VLM_FP8_ROOT / "Qwen2.5-VL-32B-Instruct-FP8",
    "robobrain2_32b_fp8saved": VLM_FP8_ROOT / "RoboBrain2.0-32B-FP8",
}
# how each model's weights are quantized (vLLM's quantization argument); a model not listed runs in bf16
QUANT = {"qwen25vl32b_awq": "awq", "robobrain2_32b_nf4": "bitsandbytes", "robobrain2_32b_fp8": "fp8",
         "qwen25vl32b_fp8": "fp8"}
# models that always run on the vLLM backend (every QUANT model does too); the saved fp8 ones load quantized
VLLM_ONLY = {"robobrain2_32b_vllm", "qwen25vl32b_fp8saved", "robobrain2_32b_fp8saved"}
PLACEHOLDERS = ("{object}", "{block_description}", "{bowl_description}")
SCORES = (-1, 0, 1)


def vlm_name(h_source: str) -> str | None:
    """"vlm:<name>" -> name; "sim" -> None. Unknown names raise."""
    if h_source == "sim":
        return None
    if not h_source.startswith("vlm:") or h_source[4:] not in VLMS:
        raise ValueError(f"h_source {h_source!r}: 'sim' or 'vlm:<name>' with name in {sorted(VLMS)}")
    return h_source[4:]


@functools.lru_cache(maxsize=None)
def load_prompts(path: str | None = None) -> dict:
    import yaml

    d = yaml.safe_load(Path(path or PROMPTS).read_text())
    for task, cons in d["prompts"].items():
        for c, text in cons.items():
            if sum(p in text for p in PLACEHOLDERS) != 1:
                raise ValueError(f"{path or PROMPTS}: the {task} / {c} prompt needs exactly one of {PLACEHOLDERS}")
    return d


def prompt_hash(path: str | None = None) -> str:
    """Hash of everything the VLM is told (the images line and every prompt)."""
    d = load_prompts(path)
    return hashlib.md5(json.dumps({"images": d["images"], "prompts": d["prompts"]}, sort_keys=True)
                       .encode()).hexdigest()[:12]


def allowed_consequences(mode: str, task: str) -> frozenset | None:
    """The consequences a task names and samples under cfg.rule_consequences: "all" -> None (no restriction);
    "vlm_prompts" -> the task's prompts in vlm_prompts.yaml (None for a task it does not list)."""
    if mode == "all":
        return None
    if mode != "vlm_prompts":
        raise ValueError(f"rule_consequences {mode!r}: 'all' or 'vlm_prompts'")
    cons = load_prompts()["prompts"].get(task)
    return None if cons is None else frozenset(cons)


def labelled(rule: dict, record: dict) -> bool:
    """A rule the VLM answers for: a sampleable single-referent rule whose (task, consequence) has a prompt."""
    cons = load_prompts()["prompts"].get(record["task"], {})
    return bool(rule.get("sampleable")) and not rule.get("set") and rule["consequence"] in cons


def _specific(phrases: list[str], noun: str | None, mods) -> str | None:
    """The phrase that tells the item apart best: its main noun (the object's category, the region's first name)
    with every one of its colour / size words that keeps the phrase unique ("the tall red block" over "the red
    block"; "the wine glass", not the synonym "the glass"). Falls back to describe()'s first phrase."""
    if not phrases:
        return None
    mods = {m for m in mods if m}
    own = [p for p in phrases if noun and p.endswith(" " + noun) and set(p[4:-len(noun)].split()) <= mods]
    return max(own, key=lambda p: len(p.split())) if own else phrases[0]


def referent_phrase(rule: dict, record: dict) -> str | None:
    """The named item's phrase, unique in its scene (training words): the object for avoid / upright, the region
    (the bowl) for exclude_particles. None when nothing names it uniquely."""
    from main.item_envs import language

    if rule["consequence"] == "exclude_particles":
        ref = rule.get("reference") or {}
        if "region" not in ref:
            return None
        reg = record["regions"][ref["region"]]
        names = reg.get("names") or [None]
        return _specific(language.describe_region(reg, record, "train"), names[0], (reg.get("color"),))
    subj = rule.get("subject") or {}
    if "object" not in subj:
        return None
    obj = record["objects"][subj["object"]]
    return _specific(language.describe(obj, record, "train"), obj.get("category"), (obj.get("color"), obj.get("size")))


def build_prompt(rule: dict, record: dict) -> tuple[str, str] | None:
    """(text after the two images, referent phrase), or None when the rule has no prompt or no phrase."""
    d = load_prompts()
    text = d["prompts"].get(record["task"], {}).get(rule["consequence"])
    phrase = referent_phrase(rule, record)
    if text is None or phrase is None:
        return None
    for p in PLACEHOLDERS:
        text = text.replace(p, phrase)
    return d["images"] + "\n\n" + text, phrase


_SCORE_RE = re.compile(r'"?score"?\s*:\s*"?\s*([+-]?\d+(?:\.\d+)?)')


def parse_score(answer: str) -> tuple[int | None, str]:
    """(score in {-1, 0, 1} or None, explanation) from the VLM's answer. Takes the first {...} that parses as JSON
    with a numeric score, else a "score": <number> anywhere; a number off the three values rounds to the nearest."""
    expl = ""
    for m in re.finditer(r"\{[^{}]*\}", answer, flags=re.S):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "score" in d:
            try:
                v = float(d["score"])
            except (TypeError, ValueError):
                continue
            return int(max(-1, min(1, round(v)))), str(d.get("explanation", ""))
    m = _SCORE_RE.search(answer)
    if m:
        e = re.search(r'"?explanation"?\s*:\s*"([^"]*)', answer)
        expl = e.group(1) if e else ""
        return int(max(-1, min(1, round(float(m.group(1)))))), expl
    return None, expl


# ------------------------------------------------------------- label files --

def label_dir(root, vlm: str) -> Path:
    return Path(root) / "vlm_labels" / vlm


def label_path(root, vlm: str, episode_id: int) -> Path:
    return label_dir(root, vlm) / f"{int(episode_id)}.json"


def frames_dir(root, episode_id: int) -> Path:
    return Path(root) / "agent_view" / str(int(episode_id))


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)
