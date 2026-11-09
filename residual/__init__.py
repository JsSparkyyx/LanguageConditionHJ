"""Residual safety policy on the don't-touch task (residual/PLAN.md).

PPO on privileged simulator state, on top of the cuRobo planner run as a 25 Hz policy, on T1
``selective_avoid`` with ur5-wsg. Run from the repo root inside the container:

    scripts/lang/in_container.sh python -m residual.throughput ...
    scripts/lang/in_container.sh python -m residual.train --run runs/residual/<name> ...

Importing this package sets up ``main.item_envs`` (``main.lang`` appends ``<repo>/envs/main`` to
``main.__path__``) without importing torch or the simulator.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def ensure_paths() -> None:
    """``main.item_envs`` next to CrossSafe's ``main.envs`` (idempotent, light: no torch, no sapien)."""
    import main.lang  # noqa: F401  (appends <repo>/envs/main to main.__path__)
