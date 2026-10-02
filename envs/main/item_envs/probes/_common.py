"""Shared probe machinery: resumable JSONL rows, sharding, tick wrapper,
per-item contacts, keyframes. Sapien/CEHJ imports are lazy so the row-log
logic loads (and is unit-tested) on the login node.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PHYSICS_FREQ = 250.0  # RoboTwin scene timestep is 1/250 s


# ---------------------------------------------------------------------------
# resumable JSONL rows
# ---------------------------------------------------------------------------


def load_rows(path) -> list[dict]:
    """Complete rows of a jsonl file; a crash-truncated final line is dropped.

    Unparseable lines elsewhere (not just the tail) are skipped and counted
    in the returned list's absence — they never crash a resume.
    """
    path = Path(path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                print(f"[probes] dropping corrupt mid-file line {i + 1} in {path}")
            # a corrupt LAST line is the crash-truncated row: dropped silently
    return rows


def row_key(row: dict, fields: tuple[str, ...]) -> tuple:
    """Hashable key from a row's id fields (booleans/numbers/stringified)."""
    return tuple(row.get(f) for f in fields)


def done_keys(path, fields: tuple[str, ...]) -> set:
    """Keys of complete rows already present (resume skips these)."""
    return {row_key(r, fields) for r in load_rows(path)}


def append_row(path, row: dict) -> None:
    """Append one row (mkdirs; flush+fsync so a crash loses at most this row)."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(row)) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def shard_cells(cells: list, shard: str) -> list:
    """Deterministic strided sharding: 'i/n' keeps cells c with index%n==i."""
    text = str(shard or "0/1")
    i, _, n = text.partition("/")
    n = max(1, int(n))
    i = int(i) % n
    return cells[i::n]


def _jsonable(obj):
    """numpy-safe json default."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# tick wrapper (runner.py's scene.step hook, 1e-9 epsilon kept)
# ---------------------------------------------------------------------------


class ProbeTimeout(Exception):
    """Raised from the scene.step hook when max_steps physics steps pass."""


@contextmanager
def tick_wrapper(task, control_freq: float, max_steps: int, on_tick):
    """Wrap task.scene.step so on_tick(n_physics) fires once per control tick.

    Same float-accumulator as RolloutController.run()/runner.py (10 x
    float(0.1) < 1.0, keep the 1e-9 epsilon). Restores the original step.
    """
    n_physics = 0
    tick_acc = 0.0
    orig_step = task.scene.step
    control_freq = float(control_freq)

    def hooked_step():
        nonlocal n_physics, tick_acc
        orig_step()
        n_physics += 1
        if n_physics > int(max_steps):
            raise ProbeTimeout(f"rollout exceeded {int(max_steps)} physics steps")
        tick_acc += control_freq / PHYSICS_FREQ
        if tick_acc >= 1.0 - 1e-9:
            tick_acc -= 1.0
            on_tick(n_physics)

    task.scene.step = hooked_step
    try:
        yield
    finally:
        task.scene.step = orig_step


# ---------------------------------------------------------------------------
# per-item contacts (distance.py obstacle_contact generalized to K items)
# ---------------------------------------------------------------------------


def item_contacts(env, items) -> tuple[np.ndarray, np.ndarray]:
    """([K] max force N, [K] touched) for contacts vs anything except the table.

    CEHJ's ``obstacle_contact`` semantics per item: force = max |impulse|/dt
    over the item's contact points this physics step; touched = any point
    with separation <= 0 (PhysX reports speculative contacts at 1-2 cm with
    zero impulse — real touches need the separation check). Item-internal
    contacts (particle-particle of one set) and item-table contacts are
    skipped; contacts between two DIFFERENT items count for both.
    """
    from ..items import entity_of

    specs = list(items)
    k = len(specs)
    forces = np.zeros(k, dtype=np.float64)
    touched = np.zeros(k, dtype=bool)
    if not specs:
        return forces, touched
    owner: dict[int, int] = {}
    for i, spec in enumerate(specs):
        entities = [entity_of(spec)]
        if getattr(spec, "kind", "rigid") == "particle_set":
            entities += [entity_of(p) for p in (spec.particles or [])]
        for ent in entities:
            owner[int(ent.per_scene_id)] = i
    scene = env.task.scene
    try:
        contacts = scene.get_contacts()
    except Exception:
        return forces, touched  # transient (before the first step), distance.py guard
    dt = float(getattr(scene, "timestep", 1.0 / PHYSICS_FREQ))
    for contact in contacts:
        bodies = contact.bodies
        ents = [bodies[0].entity, bodies[1].entity]
        idx = [owner.get(int(e.per_scene_id)) for e in ents]
        affected: set[int] = set()
        for a in range(2):
            i, j = idx[a], idx[1 - a]
            if i is None or i == j:
                continue
            if j is None and "table" in (getattr(ents[1 - a], "name", "") or "").lower():
                continue  # resting on the table is not a robot/other contact
            affected.add(i)
        if not affected:
            continue
        f_max = 0.0
        sep_min = np.inf
        for p in contact.points:
            f_max = max(f_max, float(np.linalg.norm(p.impulse)) / dt)
            sep_min = min(sep_min, float(p.separation))
        for i in affected:
            forces[i] = max(forces[i], f_max)
            if sep_min <= 0.0:
                touched[i] = True
    return forces, touched


# ---------------------------------------------------------------------------
# keyframes
# ---------------------------------------------------------------------------


class KeyframeWriter:
    """Saves at most `budget` PNGs under <out_dir>/keyframes/ (4-8 per probe run)."""

    def __init__(self, out_dir, budget: int = 8):
        self.dir = Path(out_dir) / "keyframes"
        self.budget = int(budget)
        self.saved = 0

    def maybe_save(self, name: str, rgb) -> bool:
        if rgb is None or self.saved >= self.budget:
            return False
        import imageio.v2 as imageio

        self.dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
        imageio.imwrite(self.dir / f"{safe}.png", np.asarray(rgb, dtype=np.uint8))
        self.saved += 1
        return True


def observer_frame(task):
    """Raw observer RGB, or None on failure (probe keyframes never mask errors)."""
    try:
        from .._bootstrap import ensure_main_path

        ensure_main_path()
        from main.envs.record import observer_rgb

        return observer_rgb(task)  # does its own _update_render
    except Exception as exc:
        print(f"[probes] observer frame failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# env construction (collect.py / runner.py parity)
# ---------------------------------------------------------------------------


def build_env(task_name: str, embodiment: str, seed: int, arm_distance: float | None = None,
              control_freq: float = 25.0):
    """CEHJ Env for a probe episode; CuroboIKController installed once per process."""
    from .._bootstrap import ensure_main_path

    ensure_main_path()  # main.envs.* lives in the sibling CEHJ copy
    from main.envs.controller import CuroboIKController
    from main.envs.env import Env

    from ..runner import ARM_DISTANCE

    CuroboIKController.install()
    return Env(
        task_name,
        embodiment,
        int(seed),
        arm_distance=(
            float(arm_distance)
            if arm_distance is not None
            else ARM_DISTANCE.get(embodiment, 0.6)
        ),
        control_freq=float(control_freq),
    )
