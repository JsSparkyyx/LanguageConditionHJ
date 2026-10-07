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
    assert not any(p.suffix == ".lock" for p in st.iterdir())
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
