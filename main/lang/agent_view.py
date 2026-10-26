"""Agent-view frames for the VLM failure function (VLM_FAILURE_PLAN, Yuxuan 2026-10-26).

With cfg.agent_view_every = N, collection saves the observer camera, the agent view of the episode videos (the whole
table from the front; the three encoder cameras are sometimes covered by the arm), at tick 0 and every N ticks:
``<root>/agent_view/<episode_id>/t<tick:04d>.jpg``, tick = the item sidecar row's t_in_ep. When the episode's rows are
on disk, ``done.json`` ({"episode_id", "n_rows", "ticks", "every", "size"}) tells the labeler (vlm_label.py) the
episode is complete. Rendering only: the physics never sees it. An episode whose rows were rolled back loses its
frames (discard).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .vlm_prompts import frames_dir, write_json_atomic

JPEG_QUALITY = 95


class AgentViewWriter:
    def __init__(self, root, episode_id: int, every: int):
        assert int(every) > 0, every
        self.dir = frames_dir(root, episode_id)
        self.episode_id, self.every = int(episode_id), int(every)
        if self.dir.exists():   # a stale attempt of the same seed (a resumed collection): start clean
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.ticks: list[int] = []
        self.size = None

    def wants(self, tick: int) -> bool:
        return tick % self.every == 0

    def save(self, tick: int, rgb: np.ndarray) -> None:
        from PIL import Image

        p = self.dir / f"t{int(tick):04d}.jpg"
        tmp = p.with_suffix(".tmp.jpg")
        Image.fromarray(np.ascontiguousarray(rgb)).save(tmp, quality=JPEG_QUALITY)
        tmp.replace(p)
        self.ticks.append(int(tick))
        self.size = [int(rgb.shape[1]), int(rgb.shape[0])]

    def finish(self, n_rows: int) -> None:
        """The episode's n_rows rows are on disk: frames past them (none in practice) are dropped."""
        ticks = [t for t in self.ticks if t < int(n_rows)]
        for t in set(self.ticks) - set(ticks):
            (self.dir / f"t{t:04d}.jpg").unlink(missing_ok=True)
        write_json_atomic(self.dir / "done.json", {"episode_id": self.episode_id, "n_rows": int(n_rows),
                                                   "ticks": ticks, "every": self.every, "size": self.size})

    def discard(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def read_done(root, episode_id: int) -> dict | None:
    p = frames_dir(root, episode_id) / "done.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def frame_path(root, episode_id: int, tick: int) -> Path:
    return frames_dir(root, episode_id) / f"t{int(tick):04d}.jpg"


def check_frames(root) -> list[str]:
    """Problems of a data root's agent-view frames: per episode with rows (lang_episodes.jsonl), done.json must list
    every `every`-th tick below its n_rows, with a jpg each, and n_rows must equal the episode's sidecar rows with
    t_in_ep 0 .. n - 1. [] = all good."""
    root = Path(root)
    n = int(json.loads((root / "item_buffer" / "header.json").read_text())["n"])
    ep = np.asarray(np.load(root / "item_buffer" / "episode_id.npy", mmap_mode="r")[:n]).reshape(-1)
    t = np.asarray(np.load(root / "item_buffer" / "t_in_ep.npy", mmap_mode="r")[:n]).reshape(-1)
    out = []
    for ln in (root / "lang_episodes.jsonl").read_text().splitlines():
        d = json.loads(ln)
        rows = d.get("rows") or [0, 0]
        if d.get("skipped") or int(rows[1]) <= int(rows[0]):
            continue
        eid = int(d["episode_id"])
        done = read_done(root, eid)
        if done is None:
            out.append(f"{eid}: no done.json")
            continue
        r = np.flatnonzero(ep == eid)
        if r.size != int(done["n_rows"]) or not np.array_equal(t[r], np.arange(r.size)):
            out.append(f"{eid}: {r.size} sidecar rows, done.json n_rows {done['n_rows']}")
        want = list(range(0, int(done["n_rows"]), int(done["every"])))
        if done["ticks"] != want or not all(frame_path(root, eid, k).exists() for k in want):
            out.append(f"{eid}: ticks {done['ticks'][:3]}... != every {done['every']} below {done['n_rows']}, or a jpg missing")
    return out


if __name__ == "__main__":   # python -m main.lang.agent_view <root> [<root> ...]: exit 1 on any problem
    import sys

    bad = 0
    for r in sys.argv[1:]:
        p = check_frames(r)
        print(f"{r}: {'OK' if not p else p}")
        bad += len(p)
    raise SystemExit(1 if bad else 0)
