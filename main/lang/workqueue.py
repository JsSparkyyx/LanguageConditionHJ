"""A shared work queue for chained interactive jobs of any size.

    python -m main.lang.workqueue --cmds <file> --state <dir> --slots 5

Every line of <file> is a shell command (quoting kept). Each worker (one per GPU, any number of jobs and
nodes over time) runs up to --slots commands at once, each claimed with an atomic ``mkdir`` of
``<state>/<id>.lock.<g>`` (main.lang.locks: heartbeat-touched every 30 s; a claim older than 90 s belongs
to a dead job and is taken over as generation g + 1, which one worker alone can create). The command
itself also takes its data root's writer lock (collect.py, rounds.py), so even a duplicated claim never
gives a shard two writers. A command that exits 0 is done (``<id>.done``); 75 (paused, resumable) releases the
lock for a later job; any other exit counts a failure (``<id>.fail``), and a command with 3 failures is
skipped. No new command starts within --margin seconds of $STAGE_DEADLINE. Exit 0 when every command
is done (or failed out), else 75.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

STALE_S = 90   # heartbeat every 30 s; a killed job's claims free up before the next job starts
MAX_FAILS = 3


def cmd_id(cmd: str) -> str:
    return hashlib.md5(cmd.encode()).hexdigest()[:16]


def read_cmds(path: Path) -> list[str]:
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def weight_of(cmd: str, slots: int) -> tuple[int, str]:
    """'@w<k> <cmd>' occupies k slots (e.g. a trainer + simulator needing a whole GPU)."""
    if cmd.startswith("@w"):
        k, rest = cmd[2:].split(" ", 1)
        return min(int(k), slots), rest
    return 1, cmd


def fails(state: Path, cid: str) -> int:
    p = state / f"{cid}.fail"
    return int(p.read_text() or 0) if p.exists() else 0


def try_claim(state: Path, cid: str, seen: dict | None = None) -> tuple[int, str] | None:
    """(generation, owner token) of the claim held (<cid>.lock.<g>, main.lang.locks), or None. Taking over a dead job's stale
    claim creates the next generation: one winner among racing workers (review O1). A claim is stale once
    its heartbeat counter stood still for STALE_S on THIS worker's clock (``seen`` remembers what it saw;
    round 2 B1). A fresh claim of the older code (<cid>.lock) still counts."""
    from .locks import try_take_token

    legacy = state / f"{cid}.lock"
    try:
        if time.time() - legacy.stat().st_mtime <= STALE_S:
            return None
    except FileNotFoundError:
        pass
    return try_take_token(state, f"{cid}.lock", STALE_S, seen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmds", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--margin", type=float, default=600.0, help="no new command this close to the deadline (s)")
    ap.add_argument("--logs", default=None)
    args = ap.parse_args(argv)
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    logs = Path(args.logs or state / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ.get("STAGE_DEADLINE", "inf"))
    host = os.uname().nodename
    running: dict[str, subprocess.Popen] = {}
    weights: dict[str, int] = {}
    reserved = None   # (cid, cmd, weight): a heavy command claimed, waiting for enough free slots
    held: dict[str, tuple] = {}   # cid -> (claim generation, owner token)
    seen: dict = {}             # heartbeat counters observed on other workers' claims (locks.try_take)
    stop = threading.Event()

    def heartbeat():
        from .locks import beat, read_owner

        n = 0
        while not stop.is_set():
            n += 1
            for cid, (g, tok) in list(held.items()):
                # only this worker's own claim: a reused generation number belongs to someone else (round 3)
                if read_owner(state, f"{cid}.lock", g) != tok or not beat(state, f"{cid}.lock", g, n):
                    # the claim vanished (taken over after a stall?): the command's own writer lock still
                    # keeps its data root single-writer; report it
                    print(f"[workqueue] {host} claim {cid}.lock.{g} disappeared", flush=True)
            stop.wait(30)

    def unclaim(cid):
        from .locks import holds, release

        got = held.pop(cid, None)
        if got is not None and holds(state, f"{cid}.lock", got[0], got[1]):   # never another worker's claim
            release(state, f"{cid}.lock", got[0])

    threading.Thread(target=heartbeat, daemon=True).start()
    paused: set[str] = set()   # exited 75 in this worker: left to a later job
    while True:
        # reap finished commands FIRST, so the to-do list below sees their .done
        for cid, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[cid]
            weights.pop(cid, None)
            if rc == 0:
                (state / f"{cid}.done").write_text(f"{host} {time.ctime()}\n")
            elif rc == 75:
                paused.add(cid)
            else:
                (state / f"{cid}.fail").write_text(str(fails(state, cid) + 1))
            unclaim(cid)
            print(f"[workqueue] {host} {cid} rc={rc}", flush=True)
        cmds = read_cmds(Path(args.cmds))       # re-read: commands can be appended while running
        todo = [(cmd_id(c), c) for c in cmds
                if not (state / f"{cmd_id(c)}.done").exists() and fails(state, cmd_id(c)) < MAX_FAILS]
        claimable = [(cid, c) for cid, c in todo if cid not in paused]
        open_slots = int(args.slots) - sum(weights.values())
        late = time.time() > deadline - float(args.margin)

        def launch(cid, c, w):
            nonlocal open_slots
            _, body = weight_of(c, int(args.slots))
            log = open(logs / f"{cid}.log", "a")
            log.write(f"\n=== {host} {time.ctime()} $ {body}\n")
            log.flush()
            running[cid] = subprocess.Popen(["bash", "-c", body], stdout=log, stderr=subprocess.STDOUT)
            weights[cid] = w
            open_slots -= w
            print(f"[workqueue] {host} start {cid} (w{w}): {body[:160]}", flush=True)

        if reserved is not None:
            if late:   # too late to start it: release the claim for a later job
                unclaim(reserved[0])
                reserved = None
            elif open_slots >= reserved[2]:
                launch(*reserved)
                reserved = None
        elif open_slots > 0 and not late:
            for cid, c in claimable:
                if open_slots <= 0:
                    break
                if cid in running:
                    continue
                g = try_claim(state, cid, seen)
                if g is None:            # generation 0 is a valid claim: test for None
                    continue
                held[cid] = g
                w, _ = weight_of(c, int(args.slots))
                if w <= open_slots:
                    launch(cid, c, w)
                else:
                    reserved = (cid, c, w)   # claimed: take no lighter work until it fits
                    break
        if not running and reserved is None and (not claimable or late):
            # everything done, or too close to the deadline to start more: leave the rest to a later job
            stop.set()
            return 0 if not todo else 75
        time.sleep(10)   # otherwise keep waiting: claims held elsewhere are released or go stale


if __name__ == "__main__":
    raise SystemExit(main())
