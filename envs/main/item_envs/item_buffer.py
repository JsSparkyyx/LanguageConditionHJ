"""ItemSidecarBuffer (PLAN v3 §5, Phase 5): per-tick item/rule state in a
flat memmap ring, written in LOCKSTEP with main.train.buffer.StepBuffer.

Ring semantics mirror StepBuffer exactly (read its docstring): header.json
with capacity + field shapes + n/head, head persistence, the oldest row
overwritten at capacity, ``discard_to`` valid only before the first wrap,
atomic header republish on flush (tmp + replace), lazy nothing — the field
shapes are fixed by the constructor maxima, so the memmaps are created
eagerly.

Lockstep protocol (who advances head, when, crash semantics): collect.py
hands RolloutController a ``LockstepItemWriter`` instead of the raw
StepBuffer (margin_hook.wire_item_episode). RolloutController appends
exactly one buffer entry per captured tick, in tick order (the pending
entry finalized at the next tick, plus the done-flush at episode end); the
h_fn is called once per capture and ``offer``s one sidecar row per call.
``LockstepItemWriter.append`` validates the oldest queued row FIRST — a
malformed row raises before either ring moves and stays queued — then
writes the StepBuffer entry and the sidecar row, so each ring advances its
own head once per append and slot i of either ring is the same transition.
An append with no queued row also raises before either ring moves: a
misaligned pair is worse than a crashed sweep. Rows still queued at
episode end (the final capture was never finalized — the done-flush path
was bypassed) are caught by ``LockstepItemWriter.end_episode``, which
raises; call it when the episode's rollout returns. Both rings must start
empty together: ``for_step_buffer`` refuses to pair a fresh sidecar with a
non-empty StepBuffer (non-item episodes have no item rows to pair with),
and a reopened sidecar must match the StepBuffer's capacity AND head/n
(resume parity).

Per-episode item/region/rule tables go to ``episodes.jsonl`` keyed by
episode_id (one JSON line per episode, logger's spec dicts). NO text in the
ring — phrases are regenerated from the rule table at training time.
``item_obb`` packs the item-frame OBB: [obb_center_local(3),
half_extents(3), up_axis_local(3)]. Rule order is the episode's rules list
order (stable rule_ids); item order is the episode's items list order;
region rows follow the episode's regions list (table_top first), token_mask
rows are items then nameable regions (masks.item_masks row order).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# name -> (dtype, shape in terms of the instance maxima)
# k_max covers T1's K <= 5 candidates + host items; r_max the rule table;
# g_max the regions (table + zones/containers)
_FIELD_SHAPES = {
    "item_pose": ("k_max", 7),
    "item_vel": ("k_max", 6),
    "item_obb": ("k_max", 9),
    "item_valid": ("k_max",),
    "held": (2,),
    "robot_item_dist": ("k_max",),
    "margins": ("r_max",),
    "rule_valid": ("r_max",),
    # relabeling (docs/item_envs/relabeling.md): h of EVERY rule this tick
    # (margins.rule_h, the env definition: clip(margin / h_unit, -1, 1)), and
    # whether it goes < 0 within the next FAILS_SOON_H ticks of the episode
    "h_rules": ("r_max",),
    # beads breaking each bead rule this tick (-1: not a bead rule; 2026-10-07): h_rules of a bead
    # rule = margins.particle_h(n, margin), so stored rows can be relabeled without simulating
    "n_violating": ("r_max",),
    "fails_soon": ("r_max",),
    "active_rule": (),  # the episode's active rule column, -1 = none (h = min over all)
    "t_in_ep": (),      # tick index in the episode (rows t - k / t - t_in_ep are its past)
    "episode_id": (),   # StepBuffer's episode_id (lockstep copy)
    "region_T": ("g_max", 4, 4),
    "token_masks": ("k_max+g_max", "n_cam", "n_tok"),
    "visibility": ("k_max", "n_cam"),
}
_FIELD_DTYPES = {
    "item_pose": np.float32,
    "item_vel": np.float32,
    "item_obb": np.float32,
    "item_valid": np.bool_,
    "held": np.int8,
    "robot_item_dist": np.float32,
    "margins": np.float32,
    "rule_valid": np.bool_,
    "h_rules": np.float32,
    "n_violating": np.int16,
    "fails_soon": np.bool_,
    "active_rule": np.int16,
    "t_in_ep": np.int16,
    "episode_id": np.int32,
    "region_T": np.float32,
    "token_masks": np.uint8,
    "visibility": np.float16,
}
# padding fill for rows shorter than the maxima (distances read +inf —
# "nothing there is closer than anything" — everything else reads 0/False)
_PAD_FILL = {"robot_item_dist": np.inf, "n_violating": -1}
# fields a row may omit (written with their pad fill): added after data was collected
_OPTIONAL_FIELDS = {"n_violating"}

FAILS_SOON_H = 25  # ticks (1 s): fails_soon[t, j] = min(h_rules[t : t + H, j]) < 0 within the episode


class ItemSidecarBuffer:
    def __init__(self, root: str | Path, capacity: int | None = None,
                 k_max: int = 8, r_max: int = 64, g_max: int = 8,
                 n_cam: int = 3, n_tok: int | None = None):
        # resolve NOW: RoboTwin chdirs during env setup (StepBuffer parity)
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        header_path = self.root / "header.json"
        existed = header_path.exists()
        if capacity is not None:
            if n_tok is None:
                raise ValueError("n_tok is required when creating a sidecar buffer")
            old = json.loads(header_path.read_text()) if header_path.exists() else {}
            header = {
                "capacity": int(capacity),
                "k_max": int(k_max), "r_max": int(r_max), "g_max": int(g_max),
                "n_cam": int(n_cam), "n_tok": int(n_tok),
            }
            for key, value in header.items():
                if old and old.get(key) != value:
                    raise ValueError(
                        f"{self.root}: header mismatch on {key}: "
                        f"{old.get(key)} on disk vs {value} requested"
                    )
            header["shapes"] = {
                name: list(self._shape(name, **header)) for name in _FIELD_SHAPES
            }
            header["n"] = int(old.get("n", 0))
            header["head"] = int(old.get("head", 0))
            header_path.write_text(json.dumps(header, indent=2))
        self.header = json.loads(header_path.read_text()) if header_path.exists() else {}
        if capacity is None:
            capacity = self.header.get("capacity")
            if capacity is None:
                raise ValueError(
                    f"{self.root}: no capacity given and no header.json found"
                )
        self.capacity = int(capacity)
        self.k_max = int(self.header.get("k_max", k_max))
        self.r_max = int(self.header.get("r_max", r_max))
        self.g_max = int(self.header.get("g_max", g_max))
        self.n_cam = int(self.header.get("n_cam", n_cam))
        self.n_tok = int(self.header.get("n_tok", n_tok))
        self.dims = {
            "k_max": self.k_max, "r_max": self.r_max, "g_max": self.g_max,
            "n_cam": self.n_cam, "n_tok": self.n_tok,
        }
        self.arrays: dict[str, np.ndarray] = {}
        for name in _FIELD_SHAPES:
            p = self.root / f"{name}.npy"
            shape = (self.capacity, *self._shape(name, **self.dims))
            if p.exists():
                arr = np.load(p, mmap_mode="r+")
                if arr.shape != shape:
                    raise ValueError(
                        f"{self.root}: {name}.npy has shape {arr.shape}, "
                        f"expected {shape} (capacity/dims mismatch)"
                    )
                self.arrays[name] = arr
            elif name in _OPTIONAL_FIELDS and existed:
                # a buffer written before the field existed: its rows read "unknown", kept in
                # memory (training jobs open shared data dirs concurrently, never add files there)
                self.arrays[name] = np.full(shape, _PAD_FILL.get(name, 0), dtype=_FIELD_DTYPES[name])
            else:
                self.arrays[name] = np.lib.format.open_memmap(
                    p, mode="w+", dtype=_FIELD_DTYPES[name], shape=shape
                )
                if name in _OPTIONAL_FIELDS:
                    self.arrays[name][:] = _PAD_FILL.get(name, 0)
        self.n = int(self.header.get("n", 0))
        self.head = int(self.header.get("head", self.n))  # next write position

    @staticmethod
    def _shape(name: str, **dims) -> tuple:
        out = []
        for part in _FIELD_SHAPES[name]:
            if isinstance(part, str):
                out.append(sum(int(dims[term]) for term in part.split("+")))
            else:
                out.append(int(part))
        return tuple(out)

    @classmethod
    def for_step_buffer(cls, step_buf, root: str | Path | None = None, **dims):
        """The lockstep pair for a StepBuffer: same capacity, same start.

        A FRESH sidecar pairs only with a fresh (n == 0) StepBuffer — rows
        already in the StepBuffer have no item rows to pair with. A
        pre-existing sidecar reopens and must match the StepBuffer's
        capacity AND n/head (resume parity: collection appends to both).
        """
        root = Path(root) if root is not None else step_buf.root.parent / "item_buffer"
        if (root / "header.json").exists():
            sidecar = cls(root)
            if sidecar.capacity != step_buf.capacity:
                raise ValueError(
                    f"{root}: sidecar capacity {sidecar.capacity} != "
                    f"StepBuffer capacity {step_buf.capacity}"
                )
            if (sidecar.n, sidecar.head) != (step_buf.n, step_buf.head):
                raise ValueError(
                    f"{root}: sidecar at n={sidecar.n} head={sidecar.head} but "
                    f"the StepBuffer is at n={step_buf.n} head={step_buf.head} — "
                    "the pair is out of lockstep; use a fresh pair"
                )
            return sidecar
        if step_buf.n > 0:
            raise ValueError(
                f"cannot pair a fresh sidecar with a StepBuffer holding "
                f"{step_buf.n} rows (non-item episodes have no item rows); "
                "use a fresh buffer pair"
            )
        return cls(root, capacity=step_buf.capacity, **dims)

    def append(self, row: dict) -> None:
        """Ring semantics (StepBuffer parity): at capacity the OLDEST row is
        overwritten. Rows may be short in the item/rule/region leading dims —
        padded to the maxima (robot_item_dist with +inf, the rest 0/False).
        """
        for name, arr in self.arrays.items():
            if name not in row and name in _OPTIONAL_FIELDS:
                arr[self.head] = _PAD_FILL.get(name, 0)
                continue
            value = np.asarray(row[name])
            arr[self.head] = self._padded(name, value)
        self.head = (self.head + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)
        self.header["n"] = self.n
        self.header["head"] = self.head

    def validate_row(self, row: dict) -> None:
        """Raise on any field missing from ``row`` or not fitting its array
        shape, WITHOUT writing anything. ``LockstepItemWriter`` runs this
        before touching either ring, so a malformed row can never put the
        pair out of lockstep."""
        for name in self.arrays:
            if name not in row:
                if name in _OPTIONAL_FIELDS:
                    continue
                raise KeyError(f"sidecar row is missing field {name!r}")
            self._padded(name, np.asarray(row[name]))

    def _padded(self, name: str, value: np.ndarray) -> np.ndarray:
        shape = self._shape(name, **self.dims)
        if value.shape == shape:
            return value.astype(_FIELD_DTYPES[name], copy=False)
        if value.ndim != len(shape) or any(
            v > s for v, s in zip(value.shape, shape)
        ):
            raise ValueError(
                f"{name}: row shape {tuple(value.shape)} does not fit {shape}"
            )
        fill = _PAD_FILL.get(name, 0)
        out = np.full(shape, fill, dtype=_FIELD_DTYPES[name])
        slices = tuple(slice(0, v) for v in value.shape)
        out[slices] = value
        return out

    def flush(self) -> None:
        for arr in self.arrays.values():
            if isinstance(arr, np.memmap):   # in-memory optional fields of older buffers have no file
                arr.flush()
        # atomic header write (StepBuffer parity): a concurrent reader must
        # never see a partially-written n
        tmp = self.root / "header.json.tmp"
        tmp.write_text(json.dumps(self.header, indent=2))
        tmp.replace(self.root / "header.json")

    def refresh(self) -> int:
        """Re-read n/head from the header (StepBuffer parity for async pairs)."""
        header_path = self.root / "header.json"
        try:
            header = json.loads(header_path.read_text())
        except (OSError, json.JSONDecodeError):
            return self.n  # mid-rename or partially written; keep last n
        self.n = int(header.get("n", self.n))
        self.head = int(header.get("head", self.head))
        return self.n

    def discard_to(self, n: int) -> None:
        """Roll back to n rows (success-only rollback, StepBuffer parity).
        Only valid before the ring has ever wrapped."""
        n = int(n)
        assert 0 <= n <= self.n, f"discard_to({n}) with only {self.n} rows"
        assert self.head == self.n, "discard_to after the ring wrapped"
        self.n = n
        self.head = n
        self.header["n"] = n
        self.header["head"] = n

    def __len__(self) -> int:
        return self.n

    def write_episode(self, episode_id: int, runner) -> None:
        """Append the episode's item/region/rule tables to episodes.jsonl,
        keyed by episode_id (logger's spec dicts; margins columns and item
        rows follow these orders). Rolled-back episodes (success-only
        retries) leave stale lines — readers key on episode_id present in
        the ring."""
        from .logger import item_spec_dict, region_dict, rule_spec_dict

        record = {
            "episode_id": int(episode_id),
            "task": runner.task_name,
            "host_task": runner.host if runner.entry.kind == "host" else None,
            "embodiment": runner.embodiment,
            "seed": runner.seed,
            "active_rule_id": runner.active_rule_id,
            "items": [item_spec_dict(s) for s in runner.items],
            "regions": [region_dict(r) for r in runner.regions],
            "rules": [rule_spec_dict(r) for r in runner.rules],
        }
        from .logger import _jsonable

        with open(self.root / "episodes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")
        # the relabeling record (docs/item_envs/relabeling.md): one file per
        # episode, rules[idx] = column idx of h_rules / fails_soon
        from .language import episode_record

        ep_dir = self.root / "episodes"
        ep_dir.mkdir(exist_ok=True)
        tmp = ep_dir / f"{int(episode_id)}.json.tmp"
        tmp.write_text(json.dumps(_jsonable(episode_record(runner, episode_id)), indent=1))
        tmp.replace(ep_dir / f"{int(episode_id)}.json")


class LockstepItemWriter:
    """Per-episode lockstep pair handed to RolloutController as ``buf``
    (margin_hook.wire_item_episode). Not a StepBuffer — only ``append`` is
    part of RolloutController's protocol.

    ``offer`` queues one sidecar row per h_fn call (one per captured
    tick); ``append`` validates the oldest queued row FIRST — a malformed
    row raises before either ring advances and stays queued — then writes
    the StepBuffer entry and the sidecar row. RolloutController finalizes
    entries in tick order, so queue order == append order == capture
    order. ``end_episode`` is the teardown check: rows still queued at
    episode end mean the last capture was never appended (the done-flush
    path was bypassed) and raise. See the module docstring for the crash
    semantics.
    """

    def __init__(self, step_buf, sidecar: ItemSidecarBuffer):
        self.step_buf = step_buf
        self.sidecar = sidecar
        self._rows: list[dict] = []
        self._ep_slots: list[int] = []  # ring slots of this episode's rows, in tick order

    def offer(self, row: dict) -> None:
        self._rows.append(row)

    def append(self, entry: dict) -> None:
        if not self._rows:
            raise RuntimeError(
                "lockstep: StepBuffer append with no queued item row — "
                "the h_fn must offer exactly one row per captured tick"
            )
        row = self._rows[0]
        self.sidecar.validate_row(row)  # raise before either ring moves
        self.step_buf.append(entry)
        slot = self.sidecar.head
        self.sidecar.append(row)
        self._rows.pop(0)
        self._mark_fails_soon(slot)

    def _mark_fails_soon(self, slot: int) -> None:
        """fails_soon, filled as the episode runs (collect.py has no end-of-
        episode hook): a rule negative at this tick marks this row and the
        previous FAILS_SOON_H - 1 rows of the episode — so row t ends up with
        min(h_rules[t : t + H]) < 0 once tick t + H - 1 has arrived (fewer
        ticks near the episode end)."""
        self._ep_slots.append(slot)
        arr_h = self.sidecar.arrays["h_rules"]
        arr_f = self.sidecar.arrays["fails_soon"]
        neg = arr_h[slot] < 0.0
        if not neg.any():
            return
        for k in self._ep_slots[-FAILS_SOON_H:]:
            arr_f[k] |= neg

    def end_episode(self) -> tuple[int, int]:
        """Teardown check, call when the episode's rollout returns: every
        offered row must have been paired with a StepBuffer append.
        Returns (len(step_buf), len(sidecar)) — equal by construction."""
        if self._rows:
            raise RuntimeError(
                f"lockstep: {len(self._rows)} queued item row(s) never "
                "appended at episode end — the last capture was not "
                "finalized (check the rollout's done-flush path)"
            )
        return len(self.step_buf), len(self.sidecar)

    @property
    def capacity(self) -> int:
        return self.step_buf.capacity
