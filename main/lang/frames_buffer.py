"""FramesBuffer (V6, qwen_impl.md 4): the native camera renders of every captured tick.

``<root>/frames/`` is a RingSidecar (sidecar.py) in lockstep with ``buffer/`` and ``item_buffer/``. It keeps what
any front end needs to recompute its tokens offline: the three cameras' native 320 x 240 RGB and masked depth (the
renders both the GD 320 x 256 and the Qwen 308 x 252 inputs are resized from), their intrinsics and world -> camera
extrinsics, and T_base2ego. Every front-end change so far forced a re-collection because shards hold encoder
outputs; with these rows the GD reference on V6 episodes (QWEN_PLAN V0'') and later front ends are offline jobs.

Depth is stored as float16 (about 0.5 mm resolution below 2 m), checked finite before the cast.
"""

from __future__ import annotations

import numpy as np

from .sidecar import RingSidecar

NATIVE_WH = (320, 240)   # the D435 renders (RoboTwin _camera_config.yml)
N_CAM = 3                # [left_wrist, right_wrist, head]
FRAMES_VERSION = 1
CAMERA_NAMES = ("left_wrist", "right_wrist", "head")


class FramesBuffer(RingSidecar):
    KIND = "frames_version"
    VERSION = FRAMES_VERSION
    DIRNAME = "frames"

    @classmethod
    def make_fields(cls, native_wh=NATIVE_WH, n_cam: int = N_CAM, **_) -> dict:
        w, h = (int(x) for x in native_wh)
        n = int(n_cam)
        return {
            "rgb": ((n, h, w, 3), np.dtype(np.uint8)),
            "depth": ((n, h, w), np.dtype(np.float16)),
            "K": ((n, 3, 3), np.dtype(np.float64)),
            "ext": ((n, 4, 4), np.dtype(np.float64)),      # world -> camera (OpenCV convention, Env.camera_extrinsic)
            "T_base2ego": ((4, 4), np.dtype(np.float64)),  # NaN when the bundle had none
        }

    @classmethod
    def make_meta(cls, native_wh=NATIVE_WH, n_cam: int = N_CAM, **_) -> dict:
        return {"native_wh": [int(x) for x in native_wh], "n_cam": int(n_cam), "channel_order": "RGB",
                "camera_names": list(CAMERA_NAMES[: int(n_cam)]), "depth_unit": "m"}


def frames_to_row(native: dict, T_base2ego=None) -> dict:
    """qwen_obs's ``native`` sub-dict (rgb uint8 [3,H,W,3], depth float32 [3,H,W], K [3,3,3], ext [3,4,4]) -> a
    FramesBuffer row (without the alignment fields)."""
    depth = np.asarray(native["depth"], dtype=np.float32)
    if not np.isfinite(depth).all():
        raise ValueError("frames_to_row: non-finite depth")
    if float(np.abs(depth).max(initial=0.0)) > 6.0e4:
        raise ValueError("frames_to_row: depth beyond the float16 range")
    rgb = np.asarray(native["rgb"])
    if rgb.dtype != np.uint8:
        raise TypeError(f"frames_to_row: rgb is {rgb.dtype}, expected uint8")
    T = np.full((4, 4), np.nan) if T_base2ego is None else np.asarray(T_base2ego, dtype=np.float64)
    return {"rgb": rgb.copy(), "depth": depth.astype(np.float16), "K": np.asarray(native["K"], dtype=np.float64),
            "ext": np.asarray(native["ext"], dtype=np.float64), "T_base2ego": T}
