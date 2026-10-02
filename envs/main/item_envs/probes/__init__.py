"""Feasibility probes (PLAN v3 §6 Phase 3, STOP gate).

Each probe module exposes

    run(out_dir, embodiments=None, seeds=None, shard="0/1", **kw) -> dict
    summarize(out_dir) -> dict

and appends one JSON row per episode/cell to ``<out_dir>/<probe>_rows.jsonl``
(resumable: complete rows are skipped, a crash-truncated final line is
dropped). Keyframes: 4-8 representative PNGs TOTAL per probe run, under
``<out_dir>/keyframes/``.

PROBES maps probe name -> module and is extended as probes land: p1 / p3 /
p3_robot here (T1/T3 side), p2 / p4 / p5 from the T2 side. probe.py (the CLI
orchestrator) wires the subcommands — this package only provides the modules.
"""

from __future__ import annotations

import importlib

PROBES: dict = {}

KNOWN = ("p1", "p2", "p3", "p3_robot", "p4", "p5")


def get_probe(name: str):
    """Import (once) and return a probe module by name."""
    name = str(name).lower()
    if name not in PROBES:
        if name not in KNOWN:
            raise KeyError(f"unknown probe {name!r}; known: {KNOWN}")
        PROBES[name] = importlib.import_module(f".{name}", __package__)
    return PROBES[name]


def available() -> list[str]:
    """Probe names whose modules import cleanly on this node."""
    out = []
    for name in KNOWN:
        try:
            get_probe(name)
        except Exception:
            continue
        out.append(name)
    return out
