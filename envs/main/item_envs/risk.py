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
    """
    risk_cfg = task_cfg.get("risk", {}) or {}
    time_scale = _sample_knob(risk_cfg.get("time_scale", 1.0), rng)
    safe_time_scale = _sample_knob(risk_cfg.get("safe_time_scale", 0.6), rng)
    task = {name: _sample_knob(spec, rng) for name, spec in (risk_cfg.get("task", {}) or {}).items()}
    return RiskParams(time_scale=float(time_scale), safe_time_scale=float(safe_time_scale), task=task)


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
