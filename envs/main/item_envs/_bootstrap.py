"""Shared `main` package path bootstrap.

The item_envs package lives at ``<repo>/envs/main/item_envs`` but is imported
as ``main.item_envs``; the CEHJ code (``main.envs``, ``main.train``) lives at
``<repo>/main``. Both are regular packages named ``main`` — whichever Python
finds first owns the package object. Appending the other directory to
``main.__path__`` makes both resolve inside ONE package namespace.

Who calls this, and what it does in each case:

- tests and ``python -m main.item_envs.probe`` run with cwd = ``<repo>/envs``,
  so ``main`` = ``<repo>/envs/main``; ``ensure_main_path()`` appends
  ``<repo>/main`` so ``main.envs.*`` / ``main.train.*`` resolve. The probe and
  runner entry points call ``ensure_main_path()`` before importing any CEHJ
  module (``main.envs.*``).
- CEHJ-side processes (cwd = ``<repo>``) have ``main`` = ``<repo>/main``.
  There is NO env.py branch for item tasks — CEHJ-side entry points that need
  ``main.item_envs`` must do the equivalent of this bootstrap themselves:
  append ``<repo>/envs/main`` to ``main.__path__`` BEFORE importing
  ``main.item_envs.*`` (e.g. Phase 5 collection hooks). Nothing in
  ``main/envs`` imports ``main.item_envs`` implicitly.

Idempotent; never raises (a missing sibling simply skips the append).
"""

from __future__ import annotations

from pathlib import Path


def ensure_main_path() -> None:
    import main

    repo_root = Path(__file__).resolve().parents[3]  # envs/main/item_envs/_bootstrap.py
    candidates = [repo_root / "main"]  # the CEHJ copy
    existing = {str(p) for p in getattr(main, "__path__", [])}
    for cand in candidates:
        if cand.is_dir() and str(cand) not in existing:
            main.__path__.append(str(cand))
