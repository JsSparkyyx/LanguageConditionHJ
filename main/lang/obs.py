"""The HoloBrain-Qwen observation bundle (V6, qwen_impl.md 3).

``qwen_obs(env, kin)`` is ``Env.get_encoder_obs``'s camera loop (main/envs/env.py, which stays untouched) at
HoloBrain-Qwen's 308 x 252 input: the same cameras [left_wrist, right_wrist, head], the same RT sync, the same
depth mask (-Position.z x clip(alpha, 0, 1) x (Position.w < 1)), the same ego frame (the head camera) and robot
state. What differs: raw uint8 RGB (the Qwen image processor normalises inside), and cv2 bilinear for RGB AND
depth (HoloBrain-Qwen's Resize, which its weights were trained with; the GD path uses nearest for depth). The
native renders ride along under ``native`` for the frames sidecar (frames_buffer.py).
"""

from __future__ import annotations

import numpy as np

QWEN_IMAGE_WH = (308, 252)
CAMERA_NAMES = ["left_wrist", "right_wrist", "head"]


def resize_frame(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, image_wh=QWEN_IMAGE_WH):
    """One camera's native render -> (rgb uint8, depth float32, K) at image_wh (cv2 bilinear for both; K rows 0 / 1
    scaled by W / w0 and H / h0, as HoloBrain's Resize)."""
    import cv2

    W, H = int(image_wh[0]), int(image_wh[1])
    h0, w0 = rgb.shape[:2]
    K = np.asarray(K, dtype=np.float64).copy()
    if (w0, h0) == (W, H):
        return rgb, depth, K
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    K[0, :] *= W / w0
    K[1, :] *= H / h0
    return rgb, depth, K


def ego_projections(Ks, exts):
    """Per-camera K [3, 3] at the model resolution and world -> camera ext [4, 4] (head LAST) -> (projection_mat
    [n, 4, 4] in the ego (head camera) frame, K_c @ ext_c @ inv(ext_head), and T_ego2world = inv(ext_head))."""
    T_ego2world = np.linalg.inv(exts[-1])
    projs = []
    for K, ext in zip(Ks, exts):
        proj = np.eye(4, dtype=np.float64)
        proj[:3, :4] = K @ (ext @ T_ego2world)[:3]
        projs.append(proj)
    return np.stack(projs), T_ego2world


def qwen_obs(env, kin=None) -> dict:
    """The bundle HoloBrainQwenEncoder.encode_frame and the robot-state encoder read (one frame).

    Keys: qwen (True), rgb uint8 [3, 252, 308, 3], depth float32 [3, 252, 308] metres, projection_mat [3, 4, 4]
    (ego frame at 308 x 252), image_wh [[308, 252]] x 3, T_ego2world [4, 4], T_base2ego [4, 4] or None,
    joint_state torch fp32 [1, 1, n] (measured), joint_cmd_gap, camera_names, and native = {rgb uint8
    [3, 240, 320, 3], depth float32 [3, 240, 320], K [3, 3, 3], ext [3, 4, 4] world -> camera}."""
    import torch

    env.task._update_render()   # the RT renderer keeps its own scene copy: sync before any picture
    cams = env.task.cameras
    cam_list = [cams.left_camera, cams.right_camera, cams.static_camera_list[cams.head_camera_id]]
    rgbs, depths, Ks, exts = [], [], [], []
    nat_rgb, nat_depth, nat_K = [], [], []
    for cam in cam_list:
        cam.take_picture()
        rgba = cam.get_picture("Color")
        rgb = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
        pos = cam.get_picture("Position")
        alpha = np.clip(rgba[:, :, 3], 0, 1).astype(np.float32)
        valid = (pos[:, :, 3] < 1).astype(np.float32)
        depth = np.asarray(-pos[:, :, 2], dtype=np.float32) * alpha * valid
        K = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
        nat_rgb.append(rgb), nat_depth.append(depth), nat_K.append(K)
        r, d, k = resize_frame(rgb, depth, K)
        rgbs.append(r), depths.append(d), Ks.append(k)
        exts.append(env.camera_extrinsic(cam))
    projs, T_ego2world = ego_projections(Ks, exts)
    T_base2ego = exts[-1] @ env.compute_T_base2world(kin) if kin is not None else None
    robot = env.robot
    left, right = robot.get_left_arm_real_jointState(), robot.get_right_arm_real_jointState()
    cmd = np.array(robot.get_left_arm_jointState() + robot.get_right_arm_jointState(), dtype=np.float64)
    meas = np.array(left + right, dtype=np.float64)
    cmd_gap = float(np.abs(cmd - meas).max()) if len(cmd) == len(meas) else float("nan")
    return {
        "qwen": True,
        "rgb": np.stack(rgbs),
        "depth": np.stack(depths).astype(np.float32),
        "projection_mat": projs,
        "image_wh": np.array([list(QWEN_IMAGE_WH)] * len(cam_list)),
        "T_ego2world": T_ego2world,
        "T_base2ego": T_base2ego,
        "T_cam_world": exts[-1],
        "joint_state": torch.tensor([left + right], dtype=torch.float32)[None],
        "joint_cmd_gap": cmd_gap,
        "camera_names": list(CAMERA_NAMES),
        "native": {"rgb": np.stack(nat_rgb), "depth": np.stack(nat_depth).astype(np.float32),
                   "K": np.stack(nat_K), "ext": np.stack(exts)},
    }


def encode_qwen_obs(qwen, obs):
    """qwen_obs bundle -> HoloBrainQwenEncoder.encode_frame on it as one frame: (tokens [1, 3, 99, 256],
    pos [1, 3, 99, 3], (K, V) [1, 2, 325, 128])."""
    if not obs.get("qwen"):
        raise ValueError("encode_qwen_obs: not a qwen_obs bundle (the GD get_encoder_obs one?)")
    return qwen.encode_frame(obs["rgb"][None], obs["depth"][None], obs["projection_mat"][None],
                             np.asarray(obs["T_ego2world"])[None])
