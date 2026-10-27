"""vla_finetuning/items/jobs/collect_items_loop.sh with a fake `python` on PATH that exits with scripted codes: the
relaunch on 75, the crash tolerance, the stop after quick exits, and the pass-through of every other code:

    python -m pytest -q vla_finetuning/tests/test_items_loop.py
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

LOOP = Path(__file__).resolve().parents[1] / "items" / "jobs" / "collect_items_loop.sh"
FAKE = """#!/bin/bash
n=$(cat "$FAKE_STATE" 2>/dev/null || echo 0)
echo $((n + 1)) > "$FAKE_STATE"
codes=($FAKE_CODES)
exit ${codes[$n]:-99}
"""


@pytest.mark.parametrize("codes, rc, runs", [
    ("75 0", 0, 2),                     # process recycling, then the target is reached
    ("3", 3, 1),                        # a planner that is not cuRobo: the loop ends with it
    ("75 75 75 75 75 0", 75, 5),        # five quick exits in a row: leave the shard to a later job
    ("134 0", 0, 2),                    # a crash (SIGABRT) is relaunched
    ("134 139 134 0", 75, 3),           # the third crash ends the loop
    ("1", 1, 1),
])
def test_loop(tmp_path, codes, rc, runs):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(FAKE)
    (bin_dir / "python").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "STAGE_DEADLINE"}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", FAKE_STATE=str(tmp_path / "n"), FAKE_CODES=codes)
    p = subprocess.run(["bash", str(LOOP), "--task", "gap_place"], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == rc, p.stdout + p.stderr
    assert int((tmp_path / "n").read_text()) == runs


def test_loop_stops_near_the_deadline(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(FAKE)
    (bin_dir / "python").chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FAKE_STATE=str(tmp_path / "n"),
               FAKE_CODES="75 75 0", STAGE_DEADLINE=str(int(time.time()) + 300))
    p = subprocess.run(["bash", str(LOOP)], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 75 and int((tmp_path / "n").read_text()) == 1
