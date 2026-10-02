"""Risk-knob parameters and sampling. Pure numpy / stdlib."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskParams:
    """Risk knobs for one episode.

    time_scale applies to every plan in "risky" segments, safe_time_scale to
    approach / grasp / retreat segments. task holds task-specific knobs.
    """

    time_scale: float = 1.0
    safe_time_scale: float = 0.6
    task: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "time_scale": float(self.time_scale),
            "safe_time_scale": float(self.safe_time_scale),
            "task": {k: _to_python(v) for k, v in self.task.items()},
        }


def sample_risk(task_cfg: dict, rng: np.random.RandomState) -> RiskParams:
    """Sample RiskParams from task_cfg["risk"].

    Each knob is {"dist": "uniform"|"choice", "range"/"values": ...} or a plain
    scalar constant. task_cfg["risk"]["task"] holds the task-specific knobs.

    When task_cfg["risk_collect"] has a non-empty "cells" list (filled by the
    Phase 4 calibration, PLAN v3 section 8.4), the mixture there takes
    precedence over the "risk" section.
    """
    risk_cfg = task_cfg.get("risk", {}) or {}
    collect_cfg = task_cfg.get("risk_collect") or None
    if collect_cfg and collect_cfg.get("cells"):
        return _sample_collect(risk_cfg, collect_cfg, rng)
    time_scale = _sample_knob(risk_cfg.get("time_scale", 1.0), rng)
    safe_time_scale = _sample_knob(risk_cfg.get("safe_time_scale", 0.6), rng)
    task = {name: _sample_knob(spec, rng) for name, spec in (risk_cfg.get("task", {}) or {}).items()}
    return RiskParams(time_scale=float(time_scale), safe_time_scale=float(safe_time_scale), task=task)


def _sample_collect(risk_cfg: dict, collect_cfg: dict, rng: np.random.RandomState) -> RiskParams:
    """Sample from a calibration-derived mixture over grid cells.

    Each cell pins time_scale and the task knobs it lists; task knobs a cell
    does not list keep sampling from the "risk" section, and safe_time_scale
    comes from risk_collect (falling back to the "risk" section).
    """
    cells = list(collect_cfg["cells"])
    weights = np.array([float(c.get("weight", 1.0)) for c in cells], dtype=float)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f"risk_collect cell weights must be >= 0 with a positive sum, got {weights}")
    idx = int(np.searchsorted(np.cumsum(weights / weights.sum()), rng.rand(), side="right"))
    cell = cells[min(idx, len(cells) - 1)]
    safe_spec = collect_cfg.get("safe_time_scale", risk_cfg.get("safe_time_scale", 0.6))
    task = {name: _sample_knob(spec, rng) for name, spec in (risk_cfg.get("task", {}) or {}).items()}
    for key, value in (cell.get("task") or {}).items():
        task[key] = _to_python(value)
    return RiskParams(
        time_scale=float(cell.get("time_scale", 1.0)),
        safe_time_scale=float(_sample_knob(safe_spec, rng)),
        task=task,
    )


def _sample_knob(spec, rng: np.random.RandomState):
    """Sample one knob spec; plain scalars pass through as constants."""
    if not isinstance(spec, dict) or "dist" not in spec:
        return _to_python(spec)
    dist = spec["dist"]
    if dist == "uniform":
        lo, hi = spec["range"]
        return float(rng.uniform(lo, hi))
    if dist == "choice":
        values = list(spec["values"])
        return _to_python(values[int(rng.randint(len(values)))])
    raise ValueError(f"unknown knob dist: {dist!r}")


def _to_python(v):
    """Convert numpy scalars to plain python (JSON-safe)."""
    if isinstance(v, np.generic):
        return v.item()
    return v
