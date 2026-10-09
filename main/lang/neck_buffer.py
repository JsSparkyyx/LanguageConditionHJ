"""NeckBuffer (FUSION_PLAN 3.2, fusion_impl.md 3): the version-2 sidecar of a data root.

``<root>/neck_buffer/`` sits next to ``buffer/`` (StepBuffer) and ``item_buffer/`` (ItemSidecarBuffer), with the
same capacity, ring and row indices: slot i of the three rings is the same transition. It stores what the frozen
text-image fusion needs and the StepBuffer does not: the 4 neck maps and neck_3d levels 1, 2 of every camera,
plus the camera matrices, all sentence-independent. Scene tokens depend on the sentence once the fusion runs, and
relabeling draws a sentence per sample, so they are fused at training time from these rows; the StepBuffer keeps
the unfused tokens (V0 reads a version-2 shard unchanged).

Ring semantics mirror StepBuffer / ItemSidecarBuffer: header.json with capacity, field shapes and dtypes, n / head;
the oldest row is overwritten at capacity; ``discard_to`` only before the first wrap; flush = memmaps, then the
header atomically (tmp + replace). The field shapes are fixed by the level shapes, so the memmaps are created
eagerly, before the header (a reader never sees a header without its arrays). Readers (``capacity=None``) open
``np.load(mmap_mode="r")`` and never create or change a file.

Lockstep: StepBuffer rows are written one capture late (the controller's ``_pending`` entry), so the neck row
rides in that entry as ``entry["_neck"]`` and ``NeckLockstepWriter`` (wrapping the collection's
``LockstepItemWriter``) pops it inside the same ``append``. numpy only (check_shards runs it on the login node).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEVEL_SHAPES = ((32, 40), (16, 20), (8, 10), (4, 5))   # (h, w) per neck level at 320 x 256 (strides 8 .. 64)
TOKEN_LEVELS = (1, 2)                                   # the levels the spatial enhancer turns into scene tokens
N_CAM = 3                                               # [left_wrist, right_wrist, head]
NECK_VERSION = 1
# a neck_to_row dict holds these; the lockstep writer adds the alignment fields episode_id, t_in_ep
ROW_FIELDS = ("neck_maps", "neck_3d", "projection_mat", "image_wh", "T_ego2world")


def neck_fields(level_shapes=LEVEL_SHAPES, n_cam: int = N_CAM, d_model: int = 256, d_3d: int = 128) -> dict:
    """name -> (per-row shape, dtype). neck_maps: the 4 maps per camera, cells level-major, row-major within a
    level; neck_3d: levels TOKEN_LEVELS only. Small level shapes are for tests (fake shards)."""
    n_cells = sum(int(h) * int(w) for h, w in level_shapes)
    n_tok = sum(int(level_shapes[lv][0]) * int(level_shapes[lv][1]) for lv in TOKEN_LEVELS)
    return {
        "neck_maps": ((n_cam, n_cells, d_model), np.dtype(np.float16)),
        "neck_3d": ((n_cam, n_tok, d_3d), np.dtype(np.float16)),
        "projection_mat": ((n_cam, 4, 4), np.dtype(np.float32)),   # ego frame (obs float64, every use casts)
        "image_wh": ((n_cam, 2), np.dtype(np.float32)),
        "T_ego2world": ((4, 4), np.dtype(np.float32)),             # the positions' out_extrinsic
        "episode_id": ((), np.dtype(np.int64)),                    # alignment check against the item sidecar
        "t_in_ep": ((), np.dtype(np.int32)),
    }


NECK_FIELDS = neck_fields()


def _fields_json(fields: dict) -> dict:
    return {name: [list(shape), dt.name] for name, (shape, dt) in fields.items()}


class NeckBuffer:
    def __init__(self, root: str | Path, capacity: int | None = None, *, level_shapes=LEVEL_SHAPES,
                 n_cam: int = N_CAM):
        """capacity given: create, or reopen r+ (the header must match); None: read-only, the header's fields."""
        # resolve NOW: RoboTwin chdirs during env setup (StepBuffer parity)
        self.root = Path(root).resolve()
        header_path = self.root / "header.json"
        self.writable = capacity is not None
        fresh = False
        if self.writable:
            self.root.mkdir(parents=True, exist_ok=True)
            fields = neck_fields(level_shapes, n_cam)
            want = {"capacity": int(capacity), "fields": _fields_json(fields),
                    "level_shapes": [[int(h), int(w)] for h, w in level_shapes], "neck_version": NECK_VERSION}
            old = json.loads(header_path.read_text()) if header_path.exists() else {}
            for key, value in want.items():
                if old and old.get(key) != value:
                    raise ValueError(f"{self.root}: header mismatch on {key}: {old.get(key)} on disk vs {value} "
                                     "requested")
            fresh = not old
            self.header = dict(want, n=int(old.get("n", 0)), head=int(old.get("head", old.get("n", 0))))
        else:
            if not header_path.exists():
                raise FileNotFoundError(f"{self.root}: no header.json (not a neck buffer)")
            self.header = json.loads(header_path.read_text())
            if self.header.get("neck_version") != NECK_VERSION:
                raise ValueError(f"{self.root}: neck_version {self.header.get('neck_version')}, this code reads "
                                 f"{NECK_VERSION}")
            fields = {name: (tuple(shape), np.dtype(dt)) for name, (shape, dt) in self.header["fields"].items()}
            if set(fields) != set(NECK_FIELDS):
                raise ValueError(f"{self.root}: fields {sorted(fields)}, expected {sorted(NECK_FIELDS)}")
        self.fields = fields
        self.capacity = int(self.header["capacity"])
        self.level_shapes = tuple(tuple(int(x) for x in lv) for lv in self.header["level_shapes"])
        self.arrays: dict[str, np.ndarray] = {}
        for name, (shape, dt) in fields.items():
            p = self.root / f"{name}.npy"
            full = (self.capacity, *shape)
            if p.exists():
                arr = np.load(p, mmap_mode="r+" if self.writable else "r")
                if arr.shape != full or arr.dtype != dt:
                    raise ValueError(f"{self.root}: {name}.npy is {arr.dtype} {arr.shape}, expected {dt} {full}")
            elif self.writable:
                arr = np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=full)
            else:
                raise FileNotFoundError(f"{p}: missing (a partial copy?)")
            self.arrays[name] = arr
        self.n = int(self.header.get("n", 0))
        self.head = int(self.header.get("head", self.n))  # next write position
        if fresh:   # arrays first, then the header: a reader never opens a header without its arrays
            self._write_header()

    @classmethod
    def for_step_buffer(cls, step_buf, root: str | Path | None = None, **dims) -> "NeckBuffer":
        """The lockstep neck sidecar of a StepBuffer (ItemSidecarBuffer.for_step_buffer parity).

        A FRESH neck buffer pairs only with a fresh (n == 0) StepBuffer: rows already in the StepBuffer have no
        neck rows to pair with. A pre-existing one reopens and must match the StepBuffer's capacity AND n / head.
        """
        root = Path(root) if root is not None else step_buf.root.parent / "neck_buffer"
        if (root / "header.json").exists():
            nb = cls(root, capacity=json.loads((root / "header.json").read_text())["capacity"], **dims)
            if nb.capacity != step_buf.capacity:
                raise ValueError(f"{root}: neck capacity {nb.capacity} != StepBuffer capacity {step_buf.capacity}")
            if (nb.n, nb.head) != (step_buf.n, step_buf.head):
                raise ValueError(f"{root}: neck buffer at n={nb.n} head={nb.head} but the StepBuffer is at "
                                 f"n={step_buf.n} head={step_buf.head}: the pair is out of lockstep; use a fresh pair")
            return nb
        if step_buf.n > 0:
            raise ValueError(f"cannot pair a fresh neck buffer with a StepBuffer holding {step_buf.n} rows (they "
                             "have no neck rows); use a fresh buffer pair")
        return cls(root, capacity=step_buf.capacity, **dims)

    # ------------------------------------------------------------- writes ----
    def _check_writable(self) -> None:
        if not self.writable:
            raise RuntimeError(f"{self.root}: opened read-only (capacity=None)")

    def validate_row(self, row: dict) -> dict:
        """Raise on a missing, unknown or misshapen field, or a float written to an integer field, WITHOUT
        writing anything (NeckLockstepWriter runs it before either ring moves). -> the row cast to the dtypes."""
        extra = set(row) - set(self.fields)
        if extra:
            raise ValueError(f"neck row has unknown field(s) {sorted(extra)}")
        out = {}
        for name, (shape, dt) in self.fields.items():
            if name not in row:
                raise KeyError(f"neck row is missing field {name!r}")
            v = np.asarray(row[name])
            if v.shape != tuple(shape):
                raise ValueError(f"{name}: row shape {v.shape}, expected {tuple(shape)}")
            if not np.can_cast(v.dtype, dt, casting="same_kind"):
                raise ValueError(f"{name}: {v.dtype} cannot be stored as {dt}")
            out[name] = v.astype(dt, copy=False)
        return out

    def append(self, row: dict) -> None:
        """Ring semantics (StepBuffer parity): at capacity the OLDEST row is overwritten. Shapes are exact (no
        padding: every camera and level is always present)."""
        self._check_writable()
        vals = self.validate_row(row)
        for name, v in vals.items():
            self.arrays[name][self.head] = v
        self.head = (self.head + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)
        self.header["n"] = self.n
        self.header["head"] = self.head

    def discard_to(self, n: int) -> None:
        """Roll back to n rows (StepBuffer / ItemSidecarBuffer parity). Only valid before the ring has wrapped."""
        self._check_writable()
        n = int(n)
        assert 0 <= n <= self.n, f"discard_to({n}) with only {self.n} rows"
        assert self.head == self.n, "discard_to after the ring wrapped"
        self.n = n
        self.head = n
        self.header["n"] = n
        self.header["head"] = n

    def _write_header(self) -> None:
        # atomic header write (StepBuffer parity): a concurrent reader must never see a partially-written n
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
        """Re-read n / head from the header (StepBuffer parity for async pairs)."""
        try:
            header = json.loads((self.root / "header.json").read_text())
        except (OSError, json.JSONDecodeError):
            return self.n  # mid-rename or partially written; keep last n
        self.n = int(header.get("n", self.n))
        self.head = int(header.get("head", self.head))
        return self.n

    def read(self, rows) -> dict:
        """{field: array [len(rows), ...]} of these ring slots (any order, repeats allowed; copies). A slot that
        was never written raises IndexError."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        top = self.capacity if self.n == self.capacity else self.n
        if rows.size and (rows.min() < 0 or rows.max() >= top):
            raise IndexError(f"{self.root}: rows {rows.min()}..{rows.max()} outside the {top} written rows")
        return {name: np.asarray(arr[rows]) for name, arr in self.arrays.items()}

    def __len__(self) -> int:
        return self.n


class NeckLockstepWriter:
    """Wraps ro.buf (a LockstepItemWriter) in version-2 collection.

    ``append(entry)`` pops ``entry["_neck"]`` (a neck_to_row dict; KeyError if missing) BEFORE anything is
    written, so no neck key ever reaches ``StepBuffer.append`` (it would create a new .npy per unknown key). The
    row gets its alignment fields: ``episode_id`` from the entry, ``t_in_ep`` from the entry if it carries one,
    else from the inner writer's oldest queued item row (the row this append pairs with: CrossSafe's
    ``_collect_tick`` entries carry ``episode_id`` but no ``t_in_ep``; the queued row must be of the same
    episode). The full row is validated, then ``inner.append(entry)``, then ``neck.append(row)``: a malformed neck
    row raises before either ring moves, and a failed append leaves the entry as it came. Everything else
    (offer, end_episode, _rows, capacity, ...) is passed through to inner; ``end_episode`` also checks that the
    neck ring stands where the StepBuffer does.
    """

    def __init__(self, inner, neck: NeckBuffer):
        self.inner = inner
        self.neck = neck

    def _align(self, entry: dict) -> dict:
        eid = int(entry["episode_id"])
        if "t_in_ep" in entry:
            return {"episode_id": eid, "t_in_ep": int(entry["t_in_ep"])}
        rows = getattr(self.inner, "_rows", None)
        if not rows:
            raise RuntimeError("neck lockstep: no queued item row to take t_in_ep from (the h_fn must offer one "
                               "row per captured tick)")
        item = rows[0]
        if int(item["episode_id"]) != eid:
            raise RuntimeError(f"neck lockstep: StepBuffer entry of episode {eid}, queued item row of episode "
                               f"{int(item['episode_id'])}")
        return {"episode_id": eid, "t_in_ep": int(item["t_in_ep"])}

    def append(self, entry: dict) -> None:
        if "_neck" not in entry:
            raise KeyError("neck lockstep: StepBuffer entry without '_neck' (keep_neck off in the controller?)")
        neck_row = entry.pop("_neck")
        try:
            row = self.neck.validate_row(dict(neck_row, **self._align(entry)))   # raise before either ring moves
            self.inner.append(entry)
        except BaseException:
            entry["_neck"] = neck_row
            raise
        self.neck.append(row)

    def end_episode(self):
        out = self.inner.end_episode()
        step_buf = getattr(self.inner, "step_buf", None)
        if step_buf is not None and (self.neck.n, self.neck.head) != (step_buf.n, step_buf.head):
            raise RuntimeError(f"neck lockstep: neck ring at n={self.neck.n} head={self.neck.head}, the StepBuffer "
                               f"at n={step_buf.n} head={step_buf.head}")
        return out

    def __getattr__(self, name):
        if name in ("inner", "neck"):   # not set yet (copy / unpickle): no recursion
            raise AttributeError(name)
        return getattr(self.inner, name)
