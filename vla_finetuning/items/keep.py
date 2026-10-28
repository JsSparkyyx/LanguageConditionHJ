"""The keep rule of the item-scene demonstrations (Yuxuan 2026-10-27, docs/lang_filter/safe_controllers.md): the task
succeeds and no protected rule is ever broken, h >= 0 at every capture tick of the runner's trace.npz (t = 0 and every
10 physics steps, the 1 s settle after play_once included), h = margins.rule_h with the bead rules' violating-bead
counts, as main/lang/eval_closed.py trace_violations scores it. T1 instead of its avoid rules: no physical contact of
an extra object with anything but the table (another extra object included) at any physics step (monitor.py
extra_contacts), force < 20 N.

Protected: every rule of the scene except the spill rule (particles_inside, T6's beads on the table: recorded, never
required), and for T1 also except avoid (the contact check replaces the filter's 1 cm buffer). Set and pair rules are
protected too (a min over their members: they break only with a member). T3's outside rules (the box off a mat) need
h >= 0.1 (5 mm) at every tick instead of 0: the box moves up to 1.6 cm between two captures, so a corner can clip a mat
in between. A NaN h counts as broken (nothing certifies it safe). Example: T6 with one bead in the blue bowl reads
h = -0.5 on particles_outside|beads|bowl_blue from that tick on -> "rule:particles_outside|beads|bowl_blue".

No simulator import here: CPU tests drive these on synthetic arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

T1 = "selective_avoid"
UNPROTECTED = ("particles_inside",)          # every task (T6's spill rule)
UNPROTECTED_T1 = ("avoid",)                  # T1: the contact check instead
OUTSIDE_H_MIN = 0.1                          # T3 outside (zone_exclude) rules: 5 mm of h_unit 5 cm
KEEP_FORCE_N = 20.0                          # vla_finetuning common.KEEP_FORCE_N (CrossSafe contact_force_threshold)
MIN_FRAMES = 10

# renderer / GPU errors (vla_finetuning/data/demos_safe attempts: "cannot create buffer" 685x, "vk::Queue::submit:
# ErrorDeviceLost" 3x; run logs also "vk::Device::waitIdle: ErrorDeviceLost"): the process is broken, not the seed
RENDER_ERRORS = ("cannot create buffer", "DeviceLost", "vk::", "VK_ERROR", "ErrorOutOfDeviceMemory",
                 "ErrorInitializationFailed", "ErrorMemoryMapFailed", "CUDA error", "cudaError", "CUDA out of memory",
                 "CUDA_ERROR")
# exceptions of a broken checkout or code (RoboTwin missing, a safe mode not merged yet: NotImplementedError, a bug in
# new code, a trace whose rule columns are not the runner's: TraceMismatch), never of a scene: the scenes fail with
# UnStableError, a layout RuntimeError, or the expert's "AssertionError: target_pose cannot be None" (no grasp pose);
# a seed they hit is tried again after the fix
CODE_ERRORS = ("AttributeError", "FileNotFoundError", "ImportError", "IndexError", "KeyError", "ModuleNotFoundError",
               "NameError", "NotImplementedError", "SyntaxError", "TraceMismatch", "TypeError", "UnboundLocalError")
CONTACT_ABORT = "ContactAbort"               # monitor.ContactAbort, as the runner records it ("<type>: <message>")
TIMEOUT = "ItemTimeout"


class TraceMismatch(ValueError):
    """trace.npz's rule columns are not the runner's rule table: the same for every seed of this code (a code_error)."""


def _consequence(rule) -> str:
    c = getattr(rule, "consequence", "")
    return str(getattr(c, "value", c))


def protected_rules(task: str, rules) -> dict[str, float]:
    """{rule_id: the least h every tick must keep}, in rule-table order."""
    skip = UNPROTECTED + (UNPROTECTED_T1 if task == T1 else ())
    out = {}
    for r in rules:
        name = _consequence(r)
        if name in skip:
            continue
        out[r.rule_id] = OUTSIDE_H_MIN if (task == "push_zone" and name == "outside") else 0.0
    return out


def read_trace(path: Path, rules) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t [T], margins [T, R], violation_counts [T, R]) of a runner trace.npz, columns in rule-table order."""
    with np.load(Path(path)) as z:
        t = np.asarray(z["t"], np.float64)
        m = np.asarray(z["margins"], np.float64).reshape(len(t), -1)
        c = (np.asarray(z["violation_counts"], np.int64) if "violation_counts" in z.files
             else np.full(m.shape, -1, np.int64))
        ids = [str(x) for x in z["rule_ids"]] if "rule_ids" in z.files else None
    want = [r.rule_id for r in rules]
    if ids is not None and ids != want:
        raise TraceMismatch(f"trace rule columns {ids} != the runner's rules {want}")
    if m.shape[1] != len(want):
        raise TraceMismatch(f"trace has {m.shape[1]} rule columns, the runner {len(want)} rules")
    return t, m, c


def rule_h_series(rule, margins: np.ndarray, counts: np.ndarray) -> np.ndarray:
    import main.lang  # noqa: F401 - puts envs/main on main.__path__ (main.item_envs)
    from main.item_envs.margins import rule_h

    return np.array([rule_h(rule, float(mk), int(ck)) for mk, ck in zip(margins, counts)], np.float64)


def score_trace(rules, protected: dict, t, margins, counts) -> tuple[dict, list[str]]:
    """Per rule {"min_h", "first_tick", "first_t"} (the first capture, index and seconds, with h below the rule's bar
    for a protected rule, h < 0 for the others; None if never), and the broken protected rule ids, earliest first
    (ties in rule-table order)."""
    t = np.asarray(t, np.float64)
    per, broken = {}, []
    for j, rule in enumerate(rules):
        h = rule_h_series(rule, margins[:, j], counts[:, j]) if len(t) else np.zeros(0)
        bar = protected.get(rule.rule_id, 0.0)
        bad = ~(h >= bar)                       # NaN counts as broken
        tick = int(np.argmax(bad)) if bad.any() else None
        first = None if tick is None else float(t[tick])
        finite = h[np.isfinite(h)]
        per[rule.rule_id] = {"min_h": float(finite.min()) if finite.size else None, "first_tick": tick,
                             "first_t": first}
        if rule.rule_id in protected and first is not None:
            broken.append((first, j, rule.rule_id))
    return per, [rid for _, _, rid in sorted(broken)]


def is_render_error(msg) -> bool:
    return bool(msg) and any(p in str(msg) for p in RENDER_ERRORS)


def error_reason(err: str) -> str:
    """The keep reason of an attempt that ended in an exception ("<type>: <message>", as the runner records it)."""
    err = str(err)
    if is_render_error(err):
        return "render_error"
    if err.startswith(CONTACT_ABORT):
        return "contact"
    if err.startswith(TIMEOUT):
        return "timeout"
    if err.split(":", 1)[0].strip() in CODE_ERRORS:
        return "code_error"
    return "error"


def keep_reason(facts: dict) -> str:
    """'kept' or why not, checked in this order: an exception (error_reason: "render_error" for a broken renderer,
    "contact" for T1's early abort, "timeout", "code_error" for a broken checkout or code, else "error"),
    "safe_fail:<why>" (the safe controller found no safe motion), "plan_failed" (the task's plan_success is False at
    the end, as collect_demos.py reads it), "task_failed", "rule:<rule_id>" (the first protected rule broken),
    "contact" / "force" (T1), "few_frames". A failed plan() the task recovered from (T5 / T7's step chain after a
    failed via path, the T6 pour's retries: plan_success set back to True) is no rejection; the runner's count of
    them (meta.plan_failures) is a row diagnostic only.

    facts: err (str | None), safe_fail (str | None), plan_success (bool, default True), task_success (bool), broken
    ([rule_id]), contact (None, or {"touched": bool, "force_max": N} for T1), n_frames (int)."""
    err = facts.get("err")
    if err:
        return error_reason(err)
    if facts.get("safe_fail") == "planner_error":
        return "planner_error"   # T1: cuRobo raised during the grasp search, the planner is broken, not the scene
    if facts.get("safe_fail"):
        return f"safe_fail:{facts['safe_fail']}"
    if not facts.get("plan_success", True):
        return "plan_failed"
    if not facts.get("task_success"):
        return "task_failed"
    if facts.get("broken"):
        return f"rule:{facts['broken'][0]}"
    contact = facts.get("contact")
    if contact is not None:
        if contact.get("touched"):
            return "contact"
        if float(contact.get("force_max") or 0.0) >= KEEP_FORCE_N:
            return "force"
    if int(facts.get("n_frames") or 0) < MIN_FRAMES:
        return "few_frames"
    return "kept"
