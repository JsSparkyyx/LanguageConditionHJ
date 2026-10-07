"""Language-conditioned item safety filter (docs/lang_filter/VANILLA_PLAN.md).

Run from the repo root (``python -m main.lang...``): there ``main`` is
``<repo>/main`` (CrossSafe). The item environments live at
``<repo>/envs/main/item_envs`` and are imported as ``main.item_envs``, so this
package appends ``<repo>/envs/main`` to ``main.__path__`` first — the idiom of
``main.train.collect.is_item_task`` and ``envs/main/item_envs/_bootstrap.py``.
"""

from __future__ import annotations

from pathlib import Path

import main as _main

REPO = Path(__file__).resolve().parents[2]
ITEM_MAIN = REPO / "envs" / "main"


def ensure_item_path() -> None:
    """Append <repo>/envs/main to main.__path__ (idempotent)."""
    if ITEM_MAIN.is_dir() and str(ITEM_MAIN) not in {str(p) for p in _main.__path__}:
        _main.__path__.append(str(ITEM_MAIN))


ensure_item_path()
