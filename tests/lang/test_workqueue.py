"""workqueue: done commands never rerun, paused ones wait for a later job, failures stop after 3."""

import os
import time

from main.lang.workqueue import cmd_id, main


def test_workqueue(tmp_path, monkeypatch):
    cmds = ["exit 0", "exit 75", "exit 3", f"touch {tmp_path}/ran_d"]
    (tmp_path / "cmds").write_text("\n".join(cmds) + "\n")
    monkeypatch.setenv("STAGE_DEADLINE", str(time.time() + 900))
    rc = main(["--cmds", str(tmp_path / "cmds"), "--state", str(tmp_path / "st"), "--slots", "2", "--margin", "60"])
    st = tmp_path / "st"
    assert rc == 75                                    # the paused command is left for a later job
    assert (st / f"{cmd_id('exit 0')}.done").exists() and (tmp_path / "ran_d").exists()
    assert (st / f"{cmd_id('exit 3')}.fail").read_text() == "3"
    assert not (st / f"{cmd_id('exit 75')}.done").exists()
    assert not any(".lock" in p.name for p in st.iterdir())   # every claim released
    # a second job: the paused command runs again (still 75), everything else is skipped
    rc2 = main(["--cmds", str(tmp_path / "cmds"), "--state", str(st), "--slots", "2", "--margin", "60"])
    assert rc2 == 75
    log = (st / "logs" / f"{cmd_id('exit 0')}.log").read_text()
    assert log.count("===") == 1                       # never re-run


def test_stale_lock_taken_over(tmp_path, monkeypatch):
    (tmp_path / "cmds").write_text("exit 0\n")
    st = tmp_path / "st"
    st.mkdir()
    lock = st / f"{cmd_id('exit 0')}.lock"
    lock.mkdir()
    old = time.time() - 600
    os.utime(lock, (old, old))                          # a dead job's claim
    monkeypatch.setenv("STAGE_DEADLINE", str(time.time() + 900))
    assert main(["--cmds", str(tmp_path / "cmds"), "--state", str(st), "--slots", "1", "--margin", "60"]) == 0
    assert (st / f"{cmd_id('exit 0')}.done").exists()


def test_claims_have_one_winner(tmp_path):
    """O1 / B1: a claim is stale only after its heartbeat counter stood still for stale_s on the taker's own
    clock; it is taken over as the next generation by exactly one taker; a claim taken a moment ago is never
    taken again."""
    from main.lang.locks import beat, holds, release, try_take

    assert try_take(tmp_path, "c.lock", 0.3) == 0
    seen_a, seen_b = {}, {}
    assert try_take(tmp_path, "c.lock", 0.3, seen_a) is None         # first look: remember the counter
    beat(tmp_path, "c.lock", 0, 1)                                    # the holder is alive
    time.sleep(0.35)
    assert try_take(tmp_path, "c.lock", 0.3, seen_a) is None         # counter changed: not stale
    assert try_take(tmp_path, "c.lock", 0.3, seen_b) is None
    time.sleep(0.35)                                                  # no beat for a full window
    assert try_take(tmp_path, "c.lock", 0.3, seen_a) == 1             # the first taker wins ...
    assert try_take(tmp_path, "c.lock", 0.3, seen_b) is None          # ... the second sees generation 1
    assert holds(tmp_path, "c.lock", 1) and not holds(tmp_path, "c.lock", 0)
    release(tmp_path, "c.lock", 1)
    assert not holds(tmp_path, "c.lock", 1)


def test_legacy_claims_by_mtime(tmp_path):
    """Claims of the earlier code (no beat file) keep the mtime rule."""
    from main.lang.locks import try_take

    (tmp_path / "c.lock.0").mkdir()
    assert try_take(tmp_path, "c.lock", 90) is None
    old = time.time() - 600
    os.utime(tmp_path / "c.lock.0", (old, old))
    assert try_take(tmp_path, "c.lock", 90) == 1


def test_writer_lock_lost_after_a_stall(tmp_path):
    """B1: a writer whose heartbeat stalls past the window loses its lock to the next generation, and its
    check() raises LockLost (a BaseException: the rollout's except Exception does not swallow it)."""
    import pytest

    from main.lang.locks import WriterLock

    a = WriterLock(tmp_path / "shard", stale_s=0.3, beat_s=60)       # beats only every 60 s: a "stall"
    a.check()
    with pytest.raises(WriterLock.LockHeld):
        WriterLock(tmp_path / "shard", stale_s=0.3, beat_s=60, wait_s=0.0)
    b = WriterLock(tmp_path / "shard", stale_s=0.3, beat_s=60, wait_s=1.5)   # waits out the stalled counter
    assert b.gen == a.gen + 1
    with pytest.raises(WriterLock.LockLost):
        a.check()
    assert not isinstance(WriterLock.LockLost("x"), Exception)
    a.release()                                                       # never removes b's claim
    b.check()
    b.release()


def test_heartbeat_reports_a_missing_claim(tmp_path):
    from main.lang.locks import WriterLock, release

    w = WriterLock(tmp_path / "shard", beat_s=0.05)
    release(w.root, w.NAME, w.gen)                                    # someone removed the claim
    time.sleep(0.3)
    assert w._hb.lost
    import pytest

    with pytest.raises(WriterLock.LockLost):
        w.check()


def test_workqueue_takes_over_a_dead_claim(tmp_path, monkeypatch):
    """B2: the g -> g + 1 takeover through workqueue.main (a dead job's claim with a beat counter)."""
    import main.lang.workqueue as wq
    from main.lang.locks import beat

    monkeypatch.setattr(wq, "STALE_S", 1.0)
    (tmp_path / "cmds").write_text("exit 0\n")
    st = tmp_path / "st"
    st.mkdir()
    cid = cmd_id("exit 0")
    (st / f"{cid}.lock.0").mkdir()
    beat(st, f"{cid}.lock", 0, 7)                                     # its holder died at count 7
    monkeypatch.setenv("STAGE_DEADLINE", str(time.time() + 900))
    assert main(["--cmds", str(tmp_path / "cmds"), "--state", str(st), "--slots", "1", "--margin", "60"]) == 0
    assert (st / f"{cid}.done").exists()
    assert "=== " in (st / "logs" / f"{cid}.log").read_text()


def test_takeover_removes_dead_claims_and_a_stalled_holder_stays_out(tmp_path):
    """Review: after g -> g + 1 the dead claim g is removed, so (1) the next taker after g + 1 is released
    does not wait on g's frozen counter, (2) a stalled holder of g gets LockLost even after g + 1 is gone."""
    import pytest

    from main.lang.locks import WriterLock, holds, try_take

    a = WriterLock(tmp_path / "shard", stale_s=0.3, beat_s=60)
    b = WriterLock(tmp_path / "shard", stale_s=0.3, beat_s=60, wait_s=1.5)
    assert not (tmp_path / "shard" / f".writer.{a.gen}").exists()   # a's dead claim removed by the takeover
    b.release()
    with pytest.raises(WriterLock.LockLost):
        a.check()                                                    # its directory is gone
    c = WriterLock(tmp_path / "shard", stale_s=0.3, beat_s=60)        # no frozen counter left: taken at once
    assert c.gen == a.gen                                            # the number is reused ...
    with pytest.raises(WriterLock.LockLost):
        a.check()                                                    # ... but not the claim: a stays out
    assert holds(tmp_path / "shard", ".writer", c.gen, c.token) and not holds(tmp_path / "shard", ".writer", a.gen, a.token)
    c.release()


def test_writes_are_checked_and_a_woken_writer_stops(tmp_path):
    """Review round 3, C1: guard_writes checks the lock right before every row write; check_fast reads the
    lock files whenever the last look is old (always after a stall), so a writer that wakes after a takeover
    writes nothing more."""
    import pytest

    from main.lang.collect import guard_writes
    from main.lang.locks import WriterLock

    a = WriterLock(tmp_path / "shard", stale_s=0.3, beat_s=60)
    rows = []

    class Buf:
        def append(self, row):
            rows.append(row)

    buf = Buf()
    guard_writes(a, buf)
    guard_writes(a, buf)                      # idempotent: one check per write, not two
    buf.append(1)
    b = WriterLock(tmp_path / "shard", stale_s=0.3, beat_s=60, wait_s=1.5)   # takes over the "stalled" a
    a._last_full -= 10.0                      # a wakes: its last look at the lock files is old
    with pytest.raises(WriterLock.LockLost):
        buf.append(2)
    assert rows == [1]
    a.release()
    b.release()
