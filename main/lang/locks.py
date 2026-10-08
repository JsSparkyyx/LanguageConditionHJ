"""Cluster-safe exclusive claims on a shared file system (review O1; round 2 B1).

A claim is a directory ``<dir>/<name>.<g>``: generation g, created with ``mkdir`` (atomic on Lustre / NFS).
Its holder runs a heartbeat that increments a counter in ``<dir>/<name>.<g>/beat``. The live claim is the
highest generation. A claim is stale when its counter has not changed during a full ``stale_s`` window
that the TAKER measured on its own clock (no comparison of file times between nodes, whose clocks may
differ: Lustre sets mtime from the client clock). Claims without a beat file (written by the code before
this one) fall back to the old rule, mtime older than ``stale_s``.

Taking over a stale claim creates generation g + 1, so exactly one of several racing processes wins, and a
fresh claim is never mistaken for the stale one it replaced (a rename-based takeover can steal a claim made
a moment earlier). A holder checks ``holds()`` (its generation is still the highest and its directory
exists) before every write; ``WriterLock.check()`` raises ``LockLost`` otherwise. ``fcntl.flock`` is not
used: Lustre mounted with ``localflock`` keeps such locks node-local.
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


def read_owner(d: Path, name: str, gen: int) -> str | None:
    """The random token the claim's creator wrote: generation numbers restart at 0 once every claim directory
    is gone, so the token, not the number, identifies a claim (review)."""
    try:
        return (Path(d) / f"{name}.{gen}" / "owner").read_text().strip() or None
    except FileNotFoundError:
        return None


def read_beat(d: Path, name: str, gen: int) -> int | None:
    try:
        return int((Path(d) / f"{name}.{gen}" / "beat").read_text().strip() or 0)
    except (FileNotFoundError, ValueError):
        return None


def beat(d: Path, name: str, gen: int, count: int) -> bool:
    """Write the heartbeat counter (atomic); False if the claim directory is gone."""
    p = Path(d) / f"{name}.{gen}"
    if not p.is_dir():
        return False
    tmp = p / f"beat.{os.getpid()}.tmp"
    try:
        tmp.write_text(str(int(count)))
        tmp.replace(p / "beat")
        os.utime(p)
        return True
    except FileNotFoundError:
        return False


def try_take(d: Path, name: str, stale_s: float, seen: dict | None = None) -> int | None:
    """Generation now held, or None (see try_take_token)."""
    got = try_take_token(d, name, stale_s, seen)
    return None if got is None else got[0]


def try_take_token(d: Path, name: str, stale_s: float, seen: dict | None = None) -> tuple[int, str] | None:
    """(generation, owner token) now held, or None when the top claim is live or not yet shown stale (or
    another process won the race). ``seen``: the caller's memory of observed counters,
    {(dir, name, gen, owner): (count, t_local)}, kept across calls; a claim with a counter is stale only when
    the same count of the same owner was seen ``stale_s`` ago."""
    d = Path(d)
    seen = {} if seen is None else seen
    for _ in range(3):              # the top claim may be released between listing and reading
        gens = _gens(d, name)
        target = 0
        if gens:
            top = gens[-1]
            count = read_beat(d, name, top)
            if count is None:       # no counter: a claim of the earlier code, judged by mtime
                try:
                    age = time.time() - (d / f"{name}.{top}").stat().st_mtime
                except FileNotFoundError:
                    continue
                if age <= stale_s:
                    return None
            else:
                key, now = (str(d), name, top, read_owner(d, name, top)), time.monotonic()
                prev = seen.get(key)
                if prev is None or prev[0] != count:
                    seen[key] = (count, now)
                    return None
                if now - prev[1] < stale_s:
                    return None
            target = top + 1
        try:
            (d / f"{name}.{target}").mkdir()
        except FileExistsError:     # another process took this generation first
            return None
        import uuid

        token = uuid.uuid4().hex
        (d / f"{name}.{target}" / "owner").write_text(token)
        beat(d, name, target, 0)
        # the claims below are dead (that is why this generation could be taken): remove them, so a later
        # taker does not wait out a frozen counter again after this claim is released, and a stalled holder
        # that wakes up finds its directory gone (holds() False) instead of the lock back (review)
        for g in gens:
            release(d, name, g)
        return target, token
    return None


def holds(d: Path, name: str, gen: int, token: str | None = None) -> bool:
    """Still the live claim: its directory exists, no higher generation was taken over, and (with a token)
    it is still this holder's claim, not a later one that reused the number."""
    d = Path(d)
    gens = _gens(d, name)
    if not (gens and gens[-1] == gen and (d / f"{name}.{gen}").is_dir()):
        return False
    return token is None or read_owner(d, name, gen) == token


def release(d: Path, name: str, gen: int) -> None:
    p = Path(d) / f"{name}.{gen}"
    for _ in range(3):              # a heartbeat of another process may write a beat file meanwhile
        for f in list(p.glob("beat*")) + list(p.glob("owner")):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            p.rmdir()
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.05)


class Heartbeat:
    """Background counter for one claim; ``lost`` turns True when the claim directory disappears."""

    def __init__(self, d: Path, name: str, gen: int, beat_s: float = 30.0, token: str | None = None):
        self.d, self.name, self.gen, self.token = Path(d), name, gen, token
        self.count, self.lost = 0, False
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(beat_s,), daemon=True)
        self._t.start()

    def _run(self, beat_s: float) -> None:
        while not self._stop.wait(beat_s):
            self.count += 1
            mine = self.token is None or read_owner(self.d, self.name, self.gen) == self.token
            if not (mine and beat(self.d, self.name, self.gen, self.count)):   # never beat a successor's claim
                if not self.lost:
                    print(f"[locks] {self.d / f'{self.name}.{self.gen}'} disappeared: the claim is lost", flush=True)
                self.lost = True

    def stop(self) -> None:
        """Stop and wait for the thread: a beat in flight must not recreate the beat file after release."""
        self._stop.set()
        if self._t.is_alive() and self._t is not threading.current_thread():
            self._t.join(timeout=10)


class WriterLock:
    """One writer per data root (a shard or a rounds ring) for the process lifetime.

        lock = WriterLock(root)     # raises LockHeld if another live process writes there
        lock.check()                # before every write: raises LockLost if the lock was taken over
    """

    NAME = ".writer"

    class LockHeld(RuntimeError):
        pass

    class LockLost(BaseException):
        """A BaseException: the base rollout catches Exception around play_once and would finish the episode
        (writing rows) after a lost lock; this stops the process at the next check instead."""

    def __init__(self, root: str | Path, stale_s: float = 180.0, beat_s: float = 30.0, wait_s: float = 0.0):
        """wait_s: keep retrying this long. A crashed writer's lock is taken over once its counter stood
        still for stale_s, measured here; wait_s must exceed stale_s for a restart to wait it out."""
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        t_end = time.monotonic() + float(wait_s)
        seen = {}
        got = try_take_token(self.root, self.NAME, stale_s, seen)
        while got is None and time.monotonic() < t_end:
            time.sleep(min(10.0, max(t_end - time.monotonic(), 0.1)))
            got = try_take_token(self.root, self.NAME, stale_s, seen)
        if got is None:
            raise WriterLock.LockHeld(f"{self.root}: another process is writing here "
                                      f"({self.NAME}.* heartbeat alive within {stale_s:.0f}s)")
        self.gen, self.token = got
        self._hb = Heartbeat(self.root, self.NAME, self.gen, beat_s, self.token)
        self._released = False
        self._last_full = time.monotonic()
        atexit.register(self.release)

    @property
    def path(self) -> Path:
        return self.root / f"{self.NAME}.{self.gen}"

    def check(self) -> None:
        if self._released or self._hb.lost or not holds(self.root, self.NAME, self.gen, self.token):
            raise WriterLock.LockLost(f"{self.root}: writer lock {self.NAME}.{self.gen} is no longer held "
                                      "(taken over after a stall?): stopping without writing")
        self._last_full = time.monotonic()

    def check_fast(self, full_every_s: float = 2.0) -> None:
        """Before every row write (review round 3, C1): the heartbeat's in-memory flag, plus the file check
        whenever the last one is older than full_every_s. A takeover needs this process silent for stale_s
        (180 s), so a writer that wakes from such a stall always runs the file check before its next write."""
        if self._released or self._hb.lost or time.monotonic() - self._last_full > full_every_s:
            self.check()

    def release(self) -> None:
        if not getattr(self, "_released", True):
            self._released = True
            self._hb.stop()
            if holds(self.root, self.NAME, self.gen, self.token):   # never remove a successor's claim
                release(self.root, self.NAME, self.gen)
