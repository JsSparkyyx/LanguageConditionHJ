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
    """O1: a stale claim is taken over as the next generation by exactly one taker; a claim taken a moment
    ago is never taken again (no rename race)."""
    from main.lang.locks import holds, release, try_take

    assert try_take(tmp_path, "c.lock", 90) == 0
    assert try_take(tmp_path, "c.lock", 90) is None              # live
    old = time.time() - 600
    os.utime(tmp_path / "c.lock.0", (old, old))                   # its job died
    assert try_take(tmp_path, "c.lock", 90) == 1                  # the first taker wins ...
    assert try_take(tmp_path, "c.lock", 90) is None               # ... a second one sees generation 1 live
    assert holds(tmp_path, "c.lock", 1) and not holds(tmp_path, "c.lock", 0)
    release(tmp_path, "c.lock", 1)
    assert try_take(tmp_path, "c.lock", 90) == 1                  # gen 0 is stale: next is 1 again


def test_writer_lock(tmp_path):
    """O1: one writer per data root; a second process refuses; a dead writer's lock is taken over."""
    import pytest

    from main.lang.locks import WriterLock

    a = WriterLock(tmp_path / "shard")
    with pytest.raises(WriterLock.LockHeld):
        WriterLock(tmp_path / "shard")
    a.release()
    b = WriterLock(tmp_path / "shard")
    old = time.time() - 600
    b._stop.set()                                                 # b's process died: no more heartbeats
    os.utime(b.path, (old, old))
    c = WriterLock(tmp_path / "shard")
    assert c.gen == b.gen + 1
    c.release()
