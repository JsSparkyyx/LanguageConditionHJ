"""demo_io -> HoloBrain's packer -> RoboTwinLmdbDataset round trip on synthetic frames (no simulator).
Needs the container env (robo_orchard_lab, h5py, lmdb, cv2):

    scripts/lang/in_container.sh python -m pytest -q vla_finetuning/tests/test_demo_io.py
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from vla_finetuning.common import CAMS
from vla_finetuning.demo_io import write_episode, write_instructions, write_seed_txt


def _frames(T: int, rng) -> list[dict]:
    import cv2

    frames = []
    for t in range(T):
        fr = {"joint": rng.normal(size=14), "endpose": rng.normal(size=14), "cams": {}}
        for i, cam in enumerate(CAMS):
            rgb = np.zeros((240, 320, 3), np.uint8)
            rgb[..., 0] = 10 * t + i          # R channel carries (frame, camera)
            rgb[..., 2] = 200
            ok, jpg = cv2.imencode(".jpg", rgb)
            assert ok
            fr["cams"][cam] = {"rgb": jpg.tobytes(),
                               "depth": np.full((240, 320), 500 + 7 * t + i, np.uint16),
                               "K": np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]]),
                               "ext": np.hstack([np.eye(3), np.array([[0.1 * t], [0.0], [0.5]])])}
        frames.append(fr)
    return frames


def test_pack_round_trip(tmp_path):
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import RoboTwinLmdbDataset

    rng = np.random.default_rng(0)
    root = tmp_path / "demos"
    d = root / "stack_blocks_two" / "cfg_a"
    eps = [_frames(5, rng), _frames(3, rng)]
    for k, fr in enumerate(eps):
        write_episode(d / "data" / f"episode{k}.hdf5", fr)
        write_instructions(d / "instructions" / f"episode{k}.json",
                           {"seen": [f"stack the blocks {k}"], "unseen": [f"unseen {k}"]})
    write_seed_txt(d / "seed.txt", [11, 7])
    out = tmp_path / "lmdb"
    subprocess.run([sys.executable, "-m", "robo_orchard_lab.dataset.robotwin.robotwin_packer",
                    "--input_path", str(root), "--output_path", str(out)], check=True)
    ds = RoboTwinLmdbDataset(paths=[str(out)], lazy_init=False, transforms=None, dataset_name="t",
                             cam_names=list(CAMS))
    assert len(ds) == 8
    # the packer sorts by seed: episode 1 (seed 7, 3 frames) first, then episode 0 (seed 11)
    order = [(1, t) for t in range(3)] + [(0, t) for t in range(5)]
    for i, (k, t) in enumerate(order):
        item = ds[i]
        assert int(item["step_index"]) == t
        joints = np.asarray(item["joint_state"])          # the whole episode, [T, 14]
        assert joints.shape == (len(eps[k]), 14)
        assert np.allclose(joints[t], eps[k][t]["joint"])
        for c, cam in enumerate(CAMS):
            assert np.allclose(np.asarray(item["T_world2cam"][c])[:3], eps[k][t]["cams"][cam]["ext"])
            assert np.allclose(np.asarray(item["intrinsic"][c])[:3, :3], eps[k][t]["cams"][cam]["K"])
        imgs = item["imgs"]
        for c, cam in enumerate(CAMS):
            img = np.asarray(imgs[c])
            # RGB-ordered JPEG decoded unchanged: channel 0 is R (allow JPEG error)
            assert abs(float(img[..., 0].mean()) - (10 * t + c)) < 3, (i, cam)
            assert abs(float(img[..., 2].mean()) - 200) < 3
            depth = np.asarray(item["depths"][c])
            assert np.allclose(depth, (500 + 7 * t + c) / 1000.0)
        assert item["text"] in (f"stack the blocks {k}", f"unseen {k}")
