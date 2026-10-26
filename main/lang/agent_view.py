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
