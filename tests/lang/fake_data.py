"""Small synthetic data roots (StepBuffer + sidecar + records) for sampler tests.

Scene tokens per frame: 3 cameras x N_TOK_CAM tokens. Every row encodes its own
identity in scene_pos[:, 0]: (global write index, t_in_ep, episode index), so a
test can check which row a frame came from, across ring wraps. With neck=True
the root is a version-2 shard: a neck_buffer/ sidecar (small level shapes,
FAKE_LEVEL_SHAPES) whose neck_maps[:, 0, :3] and neck_3d[:, 0, :3] hold the same
(write index, t_in_ep, episode index) (neck_ident).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from main.item_envs.item_buffer import FAILS_SOON_H, ItemSidecarBuffer
from main.lang.neck_buffer import ROW_FIELDS, NeckBuffer, neck_fields
from main.train.buffer import StepBuffer

FIX = Path(__file__).resolve().parent / "fixtures"
N_TOK_CAM = 4
# levels 1 + 2 = 3 + 1 cells = N_TOK_CAM tokens per camera; 13 cells per camera in all
FAKE_LEVEL_SHAPES = ((2, 4), (1, 3), (1, 1), (1, 1))


# UR5 layout: 7 tokens per arm (6 joints + gripper), action columns 0-5 and 8-13
BODY_MASK = np.zeros(20, bool)
BODY_MASK[:14] = True
JOINT_INDEX = np.full(20, -1, np.int8)
for _a in range(2):
    for _i in range(6):
        JOINT_INDEX[_a * 7 + _i] = _a * 8 + _i
ACT_COLS = np.zeros(18, np.float32)
ACT_COLS[JOINT_INDEX[JOINT_INDEX >= 0]] = 1.0


def body_tokens(rng) -> np.ndarray:
    bt = rng.standard_normal((20, 16)).astype(np.float32)
    bt[:, 14] = 0.0
    bt[7:14, 14] = 1.0
    return bt


def template(task="gap_place") -> dict:
    return json.loads((FIX / f"template_{task}.json").read_text())


def h_pattern(n_rules: int, L: int, e: int) -> np.ndarray:
    """Rule j goes negative from tick 10 + 5 j onwards in odd episodes for
    even j (harmful), stays positive otherwise -> mixed rows exist."""
    H = np.ones((L, n_rules), dtype=np.float32) * 0.8
    for j in range(n_rules):
        if (e % 2 == 1 and j % 2 == 0) or (e % 2 == 0 and j == 1):
            start = 10 + 5 * j
            H[start:, j] = -0.3
    return H


def step_entry(rng, w: int, t: int, e: int, eid: int, h: float, done: bool) -> dict:
    """One StepBuffer entry; scene_pos[0] = (w, t, e)."""
    pos = np.zeros((3 * N_TOK_CAM, 3), np.float32)
    pos[0] = (w, t, e)
    tok = np.full((3 * N_TOK_CAM, 256), (w % 1000) / 1000.0, np.float16)
    return {
        "scene_tokens": tok, "scene_pos": pos,
        "state_tokens": rng.standard_normal((16, 256)).astype(np.float16) * BODY_MASK[:16, None],
        "body_tokens": body_tokens(rng), "link_pos": rng.standard_normal((20, 3)).astype(np.float32) * 0.3,
        "Jlin": (rng.standard_normal((20, 3, 18)) * 0.1).astype(np.float16),
        "Jang": (rng.standard_normal((20, 3, 18)) * 0.1).astype(np.float16), "qpos": np.zeros(16, np.float32),
        "qpos_raw": np.zeros(18, np.float32), "dtheta": np.full(18, w, np.float32) * ACT_COLS * 1e-4,
        "dtheta_max": ACT_COLS * 0.02, "h": np.float32(h),
        "per_link_h": np.zeros(20, np.float32), "d_arm_arm": np.float32(1.0),
        "body_mask": BODY_MASK, "joint_index": JOINT_INDEX,
        "embodiment_id": np.int8(0), "task_id": np.int8(0), "action_source": np.int8(0),
        "episode_id": np.int32(eid), "done": np.bool_(done),
    }


def item_row(rec: dict, H_t: np.ndarray, t: int, eid: int, active_rule=None) -> dict:
    """One item sidecar row of record rec at tick t (h_rules = H_t; token mask of object 1 on camera 0)."""
    K, R = len(rec["objects"]), len(rec["rules"])
    tm = np.zeros((K, 3, N_TOK_CAM), np.uint8)
    tm[1, 0, t % N_TOK_CAM] = 255
    return {
        "item_pose": np.zeros((K, 7), np.float32), "item_vel": np.zeros((K, 6), np.float32),
        "item_obb": np.zeros((K, 9), np.float32), "item_valid": np.ones(K, bool), "held": np.full(2, -1, np.int8),
        "robot_item_dist": np.ones(K, np.float32), "margins": H_t, "rule_valid": np.ones(R, bool),
        "h_rules": H_t, "fails_soon": np.zeros(R, bool),
        "active_rule": np.int16(-1 if active_rule is None else active_rule), "t_in_ep": np.int16(t),
        "episode_id": np.int32(eid), "region_T": np.zeros((1, 4, 4), np.float32),
        "token_masks": tm, "visibility": np.ones((K, 3), np.float16),
    }


def neck_row(w: int, t: int, e: int, level_shapes=FAKE_LEVEL_SHAPES) -> dict:
    """A neck_to_row-style dict (no episode_id / t_in_ep): (w, t, e) in neck_maps[:, 0, :3], neck_3d[:, 0, :3]
    and T_ego2world[:3, 3]. fp16 holds integers exactly up to 2048."""
    assert w < 2048, "fake neck identities are fp16"
    F = neck_fields(level_shapes)
    row = {f: np.zeros(F[f][0], F[f][1]) for f in ROW_FIELDS}
    row["neck_maps"][:, 0, :3] = (w, t, e)
    row["neck_3d"][:, 0, :3] = (w, t, e)
    row["projection_mat"][:] = np.eye(4, dtype=np.float32)
    row["image_wh"][:] = (320, 256)
    row["T_ego2world"][:] = np.eye(4, dtype=np.float32)
    row["T_ego2world"][:3, 3] = (w, t, e)
    return row


def neck_ident(neck_maps_row) -> tuple[int, int, int]:
    """(write index, t_in_ep, episode index) of a fake neck row's neck_maps [3, cells, 256]."""
    w, t, e = np.asarray(neck_maps_row)[0, 0, :3].astype(np.float32)
    return int(w), int(t), int(e)


def make_root(root: Path, episodes, capacity: int, task="gap_place", lengths=None, active_rule=None, eid0=1000,
              neck: bool = False):
    """episodes: number of episodes; writes len per lengths[e] (default 40). neck: also a neck_buffer/ sidecar
    in lockstep (a version-2 shard); the other files are the same either way."""
    root = Path(root)
    buf = StepBuffer(root / "buffer", capacity=capacity, header={"fake": True})
    side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=N_TOK_CAM, k_max=8, r_max=16, g_max=4)
    nb = NeckBuffer.for_step_buffer(buf, level_shapes=FAKE_LEVEL_SHAPES) if neck else None
    rec0 = template(task)
    lines = []
    w = 0
    rng = np.random.default_rng(eid0)
    for e in range(episodes):
        eid = eid0 + e
        L = (lengths or [40] * episodes)[e]
        rec = dict(rec0, episode_id=eid, seed=eid)
        if active_rule is not None:
            rec["active_rule"] = int(active_rule)
        R = len(rec["rules"])
        H = h_pattern(R, L, e)
        (side.root / "episodes").mkdir(exist_ok=True)
        (side.root / "episodes" / f"{eid}.json").write_text(json.dumps(rec))
        slots = []
        for t in range(L):
            buf.append(step_entry(rng, w, t, e, eid, H[t].min(), t == L - 1))
            slot = side.head
            side.append(item_row(rec, H[t], t, eid, active_rule))
            if nb is not None:
                nb.append(dict(neck_row(w, t, e), episode_id=eid, t_in_ep=t))
            slots.append(slot)
            neg = side.arrays["h_rules"][slot] < 0
            if neg.any():
                for k in slots[-FAILS_SOON_H:]:
                    side.arrays["fails_soon"][k] |= neg
            w += 1
        lines.append({"episode_id": eid, "task": task, "mask_region_rows": {}})
    buf.flush(), side.flush()
    if nb is not None:
        nb.flush()
    (root / "lang_episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return root
