"""Ring sidecars of a data root and their lockstep writer (V6, qwen_impl.md 4).

A sidecar sits next to ``buffer/`` (StepBuffer) and ``item_buffer/`` (ItemSidecarBuffer) with the same capacity,
ring and row indices: slot i of every ring is the same captured tick. ``RingSidecar`` is NeckBuffer's ring
(neck_buffer.py) with the fields given by a subclass: header.json with capacity, field shapes and dtypes, n / head
and the subclass's meta; the oldest row is overwritten at capacity; ``discard_to`` only before the first wrap;
flush = memmaps, then the header atomically. Field shapes are fixed, so the memmaps are created eagerly, before the
header (a reader never sees a header without its arrays), and a writable reopen refuses a missing array instead
of recreating it as zeros. Readers (``capacity=None``) open ``np.load(mmap_mode="r")`` and never create or change
a file. Every row carries the alignment fields ``episode_id`` / ``t_in_ep``.

``SidecarLockstepWriter`` generalises NeckLockstepWriter to any number of sidecars: StepBuffer rows are written one
capture late (the controller's ``_pending`` entry), so each sidecar row rides in that entry under its key
(``"_kv"``, ``"_frames"``) and is popped inside the same ``append``, before ``StepBuffer.append`` sees the entry
(an unknown key there raises only after other fields were written). numpy only (check_shards runs on the login
node).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ALIGN_FIELDS = {"episode_id": ((), np.dtype(np.int64)), "t_in_ep": ((), np.dtype(np.int32))}


def fields_json(fields: dict) -> dict:
    return {name: [list(shape), dt.name] for name, (shape, dt) in fields.items()}


def fields_from_json(d: dict) -> dict:
    return {name: (tuple(shape), np.dtype(dt)) for name, (shape, dt) in d.items()}


class RingSidecar:
    """Subclasses set ``KIND`` (the header key holding the version), ``VERSION``, ``DIRNAME`` and implement
    ``make_fields(**dims) -> {name: (shape, dtype)}`` (alignment fields are added here) and
    ``make_meta(**dims) -> dict`` (header entries a reopen must match)."""

    KIND = "sidecar_version"
    VERSION = 1
    DIRNAME = "sidecar"

    @classmethod
    def make_fields(cls, **dims) -> dict:
        raise NotImplementedError

    @classmethod
    def make_meta(cls, **dims) -> dict:
        return {}

    def __init__(self, root: str | Path, capacity: int | None = None, **dims):
        """capacity given: create, or reopen r+ (the header must match; an existing buffer missing an array
        raises FileNotFoundError); None: read-only, the header's fields (dims, if given, must match the header)."""
        self.root = Path(root).resolve()   # resolve NOW: RoboTwin chdirs during env setup
        header_path = self.root / "header.json"
        self.writable = capacity is not None
        fresh = False
        if self.writable:
            self.root.mkdir(parents=True, exist_ok=True)
            fields = dict(self.make_fields(**dims), **ALIGN_FIELDS)
            want = {"capacity": int(capacity), "fields": fields_json(fields), self.KIND: self.VERSION,
                    **self.make_meta(**dims)}
            old = json.loads(header_path.read_text()) if header_path.exists() else {}
            for key, value in want.items():
                if old and old.get(key) != value:
                    raise ValueError(f"{self.root}: header mismatch on {key}: {old.get(key)} on disk vs {value} "
                                     "requested")
            fresh = not old
            self.header = dict(want, n=int(old.get("n", 0)), head=int(old.get("head", old.get("n", 0))))
        else:
            if not header_path.exists():
                raise FileNotFoundError(f"{self.root}: no header.json (not a {self.DIRNAME})")
            self.header = json.loads(header_path.read_text())
            if self.header.get(self.KIND) != self.VERSION:
                raise ValueError(f"{self.root}: {self.KIND} {self.header.get(self.KIND)}, this code reads "
                                 f"{self.VERSION}")
            fields = fields_from_json(self.header["fields"])
            if dims:
                want_fields = dict(self.make_fields(**dims), **ALIGN_FIELDS)
                if fields_json(fields) != fields_json(want_fields):
                    raise ValueError(f"{self.root}: fields {fields_json(fields)} differ from {fields_json(want_fields)}")
                for key, value in self.make_meta(**dims).items():
                    if self.header.get(key) != value:
                        raise ValueError(f"{self.root}: header {key} = {self.header.get(key)}, expected {value}")
        self.fields = fields
        self.capacity = int(self.header["capacity"])
        self.arrays: dict[str, np.ndarray] = {}
        for name, (shape, dt) in fields.items():
            p = self.root / f"{name}.npy"
            full = (self.capacity, *shape)
            if p.exists():
                arr = np.load(p, mmap_mode="r+" if self.writable else "r")
                if arr.shape != full or arr.dtype != dt:
                    raise ValueError(f"{self.root}: {name}.npy is {arr.dtype} {arr.shape}, expected {dt} {full}")
            elif self.writable and fresh:
                arr = np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=full)
            elif self.writable:
                # a header means its arrays were all created first: recreating one as zeros would blank the rows
                # already recorded while their episode_id / t_in_ep still pass every alignment check
                raise FileNotFoundError(f"{p}: missing in an existing {self.DIRNAME} (header.json present; a "
                                        "partial copy?)")
            else:
                raise FileNotFoundError(f"{p}: missing (a partial copy?)")
            self.arrays[name] = arr
        self.n = int(self.header.get("n", 0))
        self.head = int(self.header.get("head", self.n))  # next write position
        if fresh:   # arrays first, then the header
            self._write_header()

    @classmethod
    def for_step_buffer(cls, step_buf, root: str | Path | None = None, **dims):
        """The lockstep sidecar of a StepBuffer: a FRESH one pairs only with a fresh (n == 0) StepBuffer; an
        existing one reopens and must match the StepBuffer's capacity AND n / head."""
        root = Path(root) if root is not None else step_buf.root.parent / cls.DIRNAME
        if (root / "header.json").exists():
            sc = cls(root, capacity=json.loads((root / "header.json").read_text())["capacity"], **dims)
            if sc.capacity != step_buf.capacity:
                raise ValueError(f"{root}: capacity {sc.capacity} != StepBuffer capacity {step_buf.capacity}")
            if (sc.n, sc.head) != (step_buf.n, step_buf.head):
                raise ValueError(f"{root}: at n={sc.n} head={sc.head} but the StepBuffer is at n={step_buf.n} "
                                 f"head={step_buf.head}: the pair is out of lockstep; use a fresh pair")
            return sc
        if step_buf.n > 0:
            raise ValueError(f"cannot pair a fresh {cls.DIRNAME} with a StepBuffer holding {step_buf.n} rows; use a "
                             "fresh buffer pair")
        return cls(root, capacity=step_buf.capacity, **dims)

    @classmethod
    def reopen(cls, root: str | Path):
        """Writable reopen with the header's own dims and fields (resume / rollback after a kill)."""
        root = Path(root)
        header = json.loads((root / "header.json").read_text())
        sc = cls.__new__(cls)
        sc._reopen(root, header)
        return sc

    def _reopen(self, root: Path, header: dict) -> None:
        self.root = root.resolve()
        self.writable = True
        self.header = header
        if header.get(self.KIND) != self.VERSION:
            raise ValueError(f"{self.root}: {self.KIND} {header.get(self.KIND)}, this code writes {self.VERSION}")
        self.fields = fields_from_json(header["fields"])
        self.capacity = int(header["capacity"])
        self.arrays = {}
        for name, (shape, dt) in self.fields.items():
            p = self.root / f"{name}.npy"
            if not p.exists():
                raise FileNotFoundError(f"{p}: missing in an existing {self.DIRNAME}")
            arr = np.load(p, mmap_mode="r+")
            if arr.shape != (self.capacity, *shape) or arr.dtype != dt:
                raise ValueError(f"{self.root}: {name}.npy is {arr.dtype} {arr.shape}")
            self.arrays[name] = arr
        self.n = int(header.get("n", 0))
        self.head = int(header.get("head", self.n))

    # ------------------------------------------------------------- writes ----
    def _check_writable(self) -> None:
        if not self.writable:
            raise RuntimeError(f"{self.root}: opened read-only (capacity=None)")

    def validate_row(self, row: dict) -> dict:
        """Raise on a missing, unknown or misshapen field, or a float written to an integer field, WITHOUT writing
        anything (the lockstep writer runs it before either ring moves). -> the row cast to the dtypes."""
        extra = set(row) - set(self.fields)
        if extra:
            raise ValueError(f"{self.DIRNAME} row has unknown field(s) {sorted(extra)}")
        out = {}
        for name, (shape, dt) in self.fields.items():
            if name not in row:
                raise KeyError(f"{self.DIRNAME} row is missing field {name!r}")
            v = np.asarray(row[name])
            if v.shape != tuple(shape):
                raise ValueError(f"{name}: row shape {v.shape}, expected {tuple(shape)}")
            if not np.can_cast(v.dtype, dt, casting="same_kind"):
                raise ValueError(f"{name}: {v.dtype} cannot be stored as {dt}")
            out[name] = v.astype(dt, copy=False)
        return out

    def append(self, row: dict) -> None:
        """Ring semantics (StepBuffer parity): at capacity the OLDEST row is overwritten."""
        self._check_writable()
        vals = self.validate_row(row)
        for name, v in vals.items():
            self.arrays[name][self.head] = v
        self.head = (self.head + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)
        self.header["n"] = self.n
        self.header["head"] = self.head

    def discard_to(self, n: int) -> None:
        """Roll back to n rows (StepBuffer parity). Only valid before the ring has wrapped."""
        self._check_writable()
        n = int(n)
        assert 0 <= n <= self.n, f"discard_to({n}) with only {self.n} rows"
        assert self.head == self.n, "discard_to after the ring wrapped"
        self.n = n
        self.head = n
        self.header["n"] = n
        self.header["head"] = n

    def _write_header(self) -> None:
        tmp = self.root / "header.json.tmp"
        tmp.write_text(json.dumps(self.header, indent=2))
        tmp.replace(self.root / "header.json")

    def flush(self) -> None:
        self._check_writable()
        for arr in self.arrays.values():
            arr.flush()
        self._write_header()

    # -------------------------------------------------------------- reads ----
    def refresh(self) -> int:
        try:
            header = json.loads((self.root / "header.json").read_text())
        except (OSError, json.JSONDecodeError):
            return self.n  # mid-rename or partially written; keep last n
        self.n = int(header.get("n", self.n))
        self.head = int(header.get("head", self.head))
        return self.n

    def read(self, rows) -> dict:
        """{field: array [len(rows), ...]} of these ring slots (any order, repeats allowed; copies)."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        top = self.capacity if self.n == self.capacity else self.n
        if rows.size and (rows.min() < 0 or rows.max() >= top):
            raise IndexError(f"{self.root}: rows {rows.min()}..{rows.max()} outside the {top} written rows")
        return {name: np.asarray(arr[rows]) for name, arr in self.arrays.items()}

    def check_rows(self, rows, episode_id, t_in_ep) -> None:
        """Raise unless these slots hold these (episode_id, t_in_ep): the lockstep check readers run against the
        item sidecar."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        e = np.asarray(self.arrays["episode_id"][rows])
        t = np.asarray(self.arrays["t_in_ep"][rows])
        bad = (e != np.asarray(episode_id).reshape(-1)) | (t != np.asarray(t_in_ep).reshape(-1))
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            raise RuntimeError(f"{self.root}: slot {int(rows[k])} holds episode {int(e[k])} tick {int(t[k])}, the item "
                               f"sidecar says episode {int(np.asarray(episode_id).reshape(-1)[k])} tick "
                               f"{int(np.asarray(t_in_ep).reshape(-1)[k])} (out of lockstep)")

    def __len__(self) -> int:
        return self.n


class SidecarLockstepWriter:
    """Wraps ro.buf (a LockstepItemWriter, or another wrapper) in collection.

    ``append(entry)`` pops every sidecar key (``"_kv"``, ``"_frames"``; KeyError if one is missing) BEFORE anything
    is written. Each row gets the alignment fields: ``episode_id`` from the entry, ``t_in_ep`` from the entry if it
    carries one, else from the inner writer's oldest queued item row (the row this append pairs with; it must be of
    the same episode). Every row is validated, then ``inner.append(entry)``, then each sidecar's append: a malformed
    row raises before any ring moves, and a failed append leaves the entry as it came. ``end_episode`` also checks
    that every sidecar stands where the StepBuffer does. Everything else passes through to inner.
    """

    def __init__(self, inner, sidecars: dict):
        self.inner = inner
        self.sidecars = dict(sidecars)

    def _align(self, entry: dict) -> dict:
        eid = int(entry["episode_id"])
        if "t_in_ep" in entry:
            return {"episode_id": eid, "t_in_ep": int(entry["t_in_ep"])}
        rows = getattr(self.inner, "_rows", None)
        if not rows:
            raise RuntimeError("sidecar lockstep: no queued item row to take t_in_ep from (the h_fn must offer one "
                               "row per captured tick)")
        item = rows[0]
        if int(item["episode_id"]) != eid:
            raise RuntimeError(f"sidecar lockstep: StepBuffer entry of episode {eid}, queued item row of episode "
                               f"{int(item['episode_id'])}")
        return {"episode_id": eid, "t_in_ep": int(item["t_in_ep"])}

    def append(self, entry: dict) -> None:
        missing = [k for k in self.sidecars if k not in entry]
        if missing:
            raise KeyError(f"sidecar lockstep: StepBuffer entry without {missing} (the controller keeps no such rows?)")
        popped = {k: entry.pop(k) for k in self.sidecars}
        try:
            align = self._align(entry)
            vals = {k: sc.validate_row(dict(popped[k], **align)) for k, sc in self.sidecars.items()}
            self.inner.append(entry)
        except BaseException:
            entry.update(popped)
            raise
        for k, sc in self.sidecars.items():
            sc.append(vals[k])

    def end_episode(self):
        out = self.inner.end_episode()
        step_buf = getattr(self.inner, "step_buf", None)
        if step_buf is not None:
            for k, sc in self.sidecars.items():
                if (sc.n, sc.head) != (step_buf.n, step_buf.head):
                    raise RuntimeError(f"sidecar lockstep: {k} ring at n={sc.n} head={sc.head}, the StepBuffer at "
                                       f"n={step_buf.n} head={step_buf.head}")
        return out

    def __getattr__(self, name):
        if name in ("inner", "sidecars"):   # not set yet (copy / unpickle): no recursion
            raise AttributeError(name)
        return getattr(self.inner, name)
