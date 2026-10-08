"""Cluster-safe exclusive claims on a shared file system (review O1).

A claim is a directory ``<dir>/<name>.<g>``: generation g, created with ``mkdir`` (atomic on Lustre / NFS),
kept fresh by a heartbeat (mtime touched every ``beat_s``), and stale once its mtime is older than
``stale_s``. The live claim is the highest generation. Taking over a stale claim creates generation g + 1,
so exactly one of several racing processes wins, and a fresh claim is never mistaken for the stale one it
replaced (a rename-based takeover can steal a claim made a moment earlier). ``fcntl.flock`` is not used:
Lustre mounted with ``localflock`` keeps such locks node-local.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path


def _gens(d: Path, name: str) -> list[int]:
    out = []
    for p in d.glob(f"{name}.*"):
        tail = p.name[len(name) + 1:]
        if tail.isdigit() and p.is_dir():
            out.append(int(tail))
    return sorted(out)


def try_take(d: Path, name: str, stale_s: float) -> int | None:
    """Generation now held, or None when a live claim exists (or another process won the race)."""
    d = Path(d)
    for _ in range(3):              # the top claim may be released between listing and stat
        gens = _gens(d, name)
        target = 0
        if gens:
            try:
                age = time.time() - (d / f"{name}.{gens[-1]}").stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= stale_s:
                return None
            target = gens[-1] + 1
        try:
            (d / f"{name}.{target}").mkdir()
            return target
        except FileExistsError:     # another process took this generation first
            return None
    return None


def holds(d: Path, name: str, gen: int) -> bool:
    """Still the live claim (no higher generation was taken over)."""
    gens = _gens(Path(d), name)
    return bool(gens) and gens[-1] == gen


def release(d: Path, name: str, gen: int) -> None:
    try:
        (Path(d) / f"{name}.{gen}").rmdir()
    except OSError:
        pass


class WriterLock:
    """One writer per data root (a shard or a rounds ring) for the process lifetime, with a heartbeat.

        lock = WriterLock(root)        # raises LockHeld if another live process writes there
    """

    NAME = ".writer"

    class LockHeld(RuntimeError):
        pass

    def __init__(self, root: str | Path, stale_s: float = 180.0, beat_s: float = 30.0, wait_s: float = 0.0):
        """wait_s: keep retrying this long (a process that just crashed leaves a lock that goes stale after
        stale_s; a restart in the same job waits it out instead of giving up)."""
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        t_end = time.time() + float(wait_s)
        self.gen = try_take(self.root, self.NAME, stale_s)
        while self.gen is None and time.time() < t_end:
            time.sleep(min(10.0, max(t_end - time.time(), 0.1)))
            self.gen = try_take(self.root, self.NAME, stale_s)
        if self.gen is None:
            raise WriterLock.LockHeld(f"{self.root}: another process is writing here "
                                      f"({self.NAME}.* fresher than {stale_s:.0f}s)")
        self._stop = threading.Event()
        self._beat = threading.Thread(target=self._heartbeat, args=(beat_s,), daemon=True)
        self._beat.start()
        atexit.register(self.release)

    @property
    def path(self) -> Path:
        return self.root / f"{self.NAME}.{self.gen}"

    def _heartbeat(self, beat_s: float) -> None:
        while not self._stop.wait(beat_s):
            try:
                os.utime(self.path)
            except FileNotFoundError:
                pass

    def release(self) -> None:
        if getattr(self, "_stop", None) is not None and not self._stop.is_set():
            self._stop.set()
            release(self.root, self.NAME, self.gen)
