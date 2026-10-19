"""Demonstration frames -> classic RoboTwin-2.0 episode files, the format HoloBrain's packer reads
(robo_orchard_lab/dataset/robotwin/robotwin_packer.py; RoboTwin before commit e71140e):

    <root>/<task>/<config>/seed.txt                      "s0 s1 ..."  (episode i was collected with seed i)
    <root>/<task>/<config>/data/episode{i}.hdf5
    <root>/<task>/<config>/instructions/episode{i}.json  {"seen": [...], "unseen": [...]}

Per episode (T frames, one every 15 physics steps):
    endpose                       [T, 14]  left / right ee pose (x y z qw qx qy qz); unused by training
    joint_action/vector           [T, 14]  drive targets [L6, gripL, R6, gripR] (RoboTwin get_*_jointState)
    observation/<cam>/rgb         [T]      JPEG of the RGB array, encoded as RoboTwin does (cv2.imencode on
                                           the RGB-ordered array; decoding returns RGB)
    observation/<cam>/depth       [T,H,W]  uint16 millimetres (RoboTwin stores float mm; the packer casts
                                           with astype(uint16), the same values)
    observation/<cam>/intrinsic_cv [T,3,3], extrinsic_cv [T,3,4] (world -> camera, OpenCV)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .common import CAMS, atomic_write


class FrameRecorder:
    """Called with the physics step count; records one frame of task.get_obs() (the observation the
    policy reads at test time)."""

    def __init__(self, task_obj):
        self.task = task_obj
        self.frames: list[dict] = []
        self.steps: list[int] = []

    def __call__(self, n: int = 0) -> None:
        import cv2

        obs = self.task.get_obs()
        joint = np.asarray(obs["joint_action"]["vector"], np.float64)
        if joint.shape != (14,):
            raise ValueError(f"joint_action/vector {joint.shape}, expected (14,) for ur5-wsg")
        pose = []
        for arm in ("left", "right"):
            p = np.asarray(self.task.get_arm_pose(arm), np.float64).reshape(-1)
            pose.append(p[:7] if p.size >= 7 else np.pad(p, (0, 7 - p.size)))
        fr = {"joint": joint, "endpose": np.concatenate(pose), "cams": {}}
        for cam in CAMS:
            o = obs["observation"][cam]
            rgb = np.asarray(o["rgb"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3:
                raise ValueError(f"{cam} rgb {rgb.dtype} {rgb.shape}")
            ok, jpg = cv2.imencode(".jpg", rgb)
            if not ok:
                raise RuntimeError(f"{cam}: JPEG encoding failed")
            depth = np.asarray(o["depth"], np.float64)
            if not np.isfinite(depth).all():
                raise ValueError(f"{cam}: non-finite depth")
            fr["cams"][cam] = {
                "rgb": jpg.tobytes(),
                "depth": np.clip(depth, 0, 65535).astype(np.uint16),
                "K": np.asarray(o["intrinsic_cv"], np.float64).reshape(3, 3),
                "ext": np.asarray(o["extrinsic_cv"], np.float64).reshape(3, 4),
            }
        self.frames.append(fr)
        self.steps.append(int(n))


def write_episode(path: Path, frames: list[dict]) -> None:
    """Classic RoboTwin-2.0 episode hdf5, written to a temp file and renamed."""
    import h5py

    if not frames:
        raise ValueError("no frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("endpose", data=np.stack([fr["endpose"] for fr in frames]))
        ja = f.create_group("joint_action")
        ja.create_dataset("vector", data=np.stack([fr["joint"] for fr in frames]))
        obs = f.create_group("observation")
        for cam in CAMS:
            g = obs.create_group(cam)
            jpgs = [fr["cams"][cam]["rgb"] for fr in frames]
            width = max(len(j) for j in jpgs)
            g.create_dataset("rgb", data=np.array(jpgs, dtype=f"S{width}"))
            g.create_dataset("depth", data=np.stack([fr["cams"][cam]["depth"] for fr in frames]),
                             compression="gzip", compression_opts=4)
            g.create_dataset("intrinsic_cv", data=np.stack([fr["cams"][cam]["K"] for fr in frames]))
            g.create_dataset("extrinsic_cv", data=np.stack([fr["cams"][cam]["ext"] for fr in frames]))
    os.replace(tmp, path)


def write_instructions(path: Path, instr: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps({"seen": instr["seen"], "unseen": instr["unseen"]}, indent=1))


def write_seed_txt(path: Path, seeds: list[int]) -> None:
    """Space-separated, index = episode index (the packer reads seeds[ep_id])."""
    atomic_write(Path(path), " ".join(str(int(s)) for s in seeds))


def write_video(path: Path, frames: list[dict], cam: str = "head_camera", fps: float = 250 / 15) -> None:
    """MP4 of one camera's frames, for looking at a demonstration."""
    import cv2
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.mp4")
    w = imageio.get_writer(str(tmp), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    try:
        for fr in frames:
            img = cv2.imdecode(np.frombuffer(fr["cams"][cam]["rgb"], np.uint8), cv2.IMREAD_UNCHANGED)
            w.append_data(img)                    # imdecode of the RGB-ordered JPEG gives RGB back
    finally:
        w.close()
    os.replace(tmp, path)
