"""Per-tick logging for item-safety episodes (PLAN v3 §5, §2.6).

Data contracts: StepRecord (one per control tick, 25 Hz) and EpisodeMeta.
Field names are the contract with the downstream training code — do not
rename. EpisodeTrace accumulates StepRecords and writes trace.npz +
meta.json; OverlayVideoWriter renders the debug mp4; token_grid_meta()
describes the encoder token grid the masks module pools into.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ._bootstrap import ensure_main_path

ensure_main_path()  # main.envs.record lives in the sibling CEHJ copy

from main.envs.record import save_video  # noqa: E402

from .items import ItemSpec  # noqa: E402
from .masks import CAMERA_NAMES, ENCODER_IMG_WH, TOKEN_STRIDES, token_grid  # noqa: E402
from .regions import Region  # noqa: E402
from .rules import RuleSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Data contracts (PLAN v3 §5 — exact field names, exact order)
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One control tick (25 Hz)."""

    t: float  # sim seconds
    tick: int
    qpos: np.ndarray  # raw joints, both arms
    ee_pose: np.ndarray  # [2, 7] TCP poses, wxyz
    gripper: np.ndarray  # [2]
    held: np.ndarray  # [2] item index held by left/right arm, -1 if none
    item_pose: np.ndarray  # [K, 7] world, wxyz
    item_vel: np.ndarray  # [K, 6] linear then angular, world
    particle_pos: dict  # item_id -> [N, 3] for particle sets
    region_T: np.ndarray  # [G, 4, 4] world_T_region for every region this tick
    margins: np.ndarray  # [R] raw SI, in rule-table order
    robot_item_dist: np.ndarray  # [K] min signed distance robot links + payload to item (m)
    robot_item_force: np.ndarray  # [K] robot-item contact force (N)
    visibility: np.ndarray  # [K, n_cam] visible pixel fraction
    token_masks: np.ndarray  # [K + G_nameable, n_cam, n_tok] uint8, cell coverage 0..255
    action_source: int  # 0 planner, 1 actor, 2 perturbation (always 0 in Phases 2-4)
    time_scale: float  # active retime factor this tick


@dataclass
class EpisodeMeta:
    task: str
    host_task: str | None  # T1's CrossSafe host; None for robotwin-kind tasks
    embodiment: str
    seed: int
    risk: dict  # RiskParams.as_dict()
    physics: dict  # sampled friction, mass, CoM offset per item
    items: list[dict]  # ItemSpec without entity handles
    regions: list[dict]
    rules: list[dict]  # RuleSpec table; margins columns follow this order
    token_grid: dict  # image size, strides, n_tok, token order (matches the encoder)
    active_rule_id: str | None  # the rule shown to the filter in filtered runs
    task_success: bool
    violations: dict  # rule_id -> bool (min margin over episode < 0)
    min_margin: dict  # rule_id -> float
    first_violation_t: dict  # rule_id -> float | None
    plan_failures: int
    spawn_rejections: int
    wall_time_s: float
    diag: dict = field(default_factory=dict)  # non-fatal diagnostics (e.g. held-detection errors)


# ---------------------------------------------------------------------------
# JSON-safe dict views of the spec dataclasses (no sim handles)
# ---------------------------------------------------------------------------

_SPEC_FIELDS = {
    "item_id",
    "entity",
    "category",
    "color",
    "kind",
    "role",
    "group",
    "particles",
    "half_extents",
    "obb_center_local",
    "up_axis_local",
    "mass",
}


def item_spec_dict(spec: ItemSpec) -> dict:
    """ItemSpec as a JSON-safe dict, WITHOUT entity/particle handles.

    Extra task attributes attached to the spec (e.g. tippable, rolls) are
    included when they are simple JSON-safe scalars.
    """
    out = {
        "item_id": spec.item_id,
        "category": spec.category,
        "color": spec.color,
        "kind": spec.kind,
        "role": spec.role,
        "group": spec.group,
        "half_extents": np.asarray(spec.half_extents, dtype=np.float64).tolist(),
        "obb_center_local": np.asarray(spec.obb_center_local, dtype=np.float64).tolist(),
        "up_axis_local": np.asarray(spec.up_axis_local, dtype=np.float64).tolist(),
        "mass": float(spec.mass),
    }
    if spec.kind == "particle_set":
        out["n_particles"] = len(spec.particles or [])
    for key, value in vars(spec).items():
        if key in _SPEC_FIELDS or key in out:
            continue
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
    return out


def region_dict(region: Region) -> dict:
    """Region as a JSON-safe dict (frame_fn is not serializable)."""
    return {
        "region_id": region.region_id,
        "kind": region.kind,
        "polygon": np.asarray(region.polygon, dtype=np.float64).tolist(),
        "z": float(region.z),
        "z_band": float(region.z_band),
        "nameable": bool(region.nameable),
        "color": region.color,
    }


def rule_spec_dict(rule: RuleSpec) -> dict:
    """RuleSpec as a JSON-safe dict; margins columns follow this table order."""
    return {
        "rule_id": rule.rule_id,
        "consequence": rule.consequence.value
        if hasattr(rule.consequence, "value")
        else str(rule.consequence),
        "item_ids": list(rule.item_ids),
        "region_ids": list(rule.region_ids),
        "params": _jsonable(rule.params),
        "phrases": list(rule.phrases),
        "split": rule.split,
    }


def token_grid_meta() -> dict:
    """The encoder token grid (PLAN v3 §5): image size, strides, n_tok, order.

    Built from the masks module constants so the meta always matches the
    recorded token_masks. Token order: cameras in CAMERA_NAMES order
    (encoder order); within a camera, scale by scale, finest stride first,
    row-major (row, col) within each scale — the flattened buffer index is
    cam * n_tok + scale_offset + row * cols + col.
    """
    scales, n_tok = token_grid()
    return {
        "cameras": list(CAMERA_NAMES),
        "image_wh": [int(ENCODER_IMG_WH[0]), int(ENCODER_IMG_WH[1])],
        "strides": [int(s) for s in TOKEN_STRIDES],
        "scales": [
            {"stride": int(s), "rows": int(r), "cols": int(c)} for s, r, c in scales
        ],
        "n_tok": int(n_tok),
        "token_order": "camera-major; per camera: scale by scale (finest stride first), row-major within each scale",
    }


def _jsonable(obj):
    """Recursively convert numpy/scalars to plain JSON-safe python.

    Non-finite floats map to None (strict-JSON safe; occurs only on empty
    traces).
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "value"):  # str-Enum safety (Consequence)
        return obj.value
    return str(obj)


# ---------------------------------------------------------------------------
# EpisodeTrace: trace.npz + meta.json
# ---------------------------------------------------------------------------


class EpisodeTrace:
    """Accumulates StepRecords; finalize() writes trace.npz + meta.json.

    violations / min_margin / first_violation_t are computed HERE from the
    recorded margins (single source of truth) and filled into the EpisodeMeta
    handed to finalize().
    """

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[StepRecord] = []

    def add(self, rec: StepRecord) -> None:
        self._records.append(rec)

    def __len__(self) -> int:
        return len(self._records)

    def finalize(self, meta: EpisodeMeta) -> tuple[Path | None, Path]:
        """Write trace.npz (when records exist) and meta.json; fill meta stats."""
        rule_ids = [r["rule_id"] for r in meta.rules]
        n_rules = len(rule_ids)
        if self._records:
            t = np.array([r.t for r in self._records], dtype=np.float64)
            margins = np.stack(
                [np.asarray(r.margins, dtype=np.float64) for r in self._records]
            )
        else:
            t = np.zeros(0, dtype=np.float64)
            margins = np.zeros((0, n_rules), dtype=np.float64)

        for j, rid in enumerate(rule_ids):
            col = margins[:, j] if j < margins.shape[1] else np.zeros(0)
            min_m = float(col.min()) if col.size else float("nan")
            meta.min_margin[rid] = min_m
            violated = bool(col.size and (col < 0.0).any())
            meta.violations[rid] = violated
            meta.first_violation_t[rid] = (
                float(t[int(np.argmax(col < 0.0))]) if violated else None
            )

        npz_path = None
        if self._records:
            npz_path = self.out_dir / "trace.npz"
            arrays = {
                "t": t,
                "tick": np.array([r.tick for r in self._records], dtype=np.int64),
                "qpos": np.stack([np.asarray(r.qpos) for r in self._records]),
                "ee_pose": np.stack([np.asarray(r.ee_pose) for r in self._records]),
                "gripper": np.stack([np.asarray(r.gripper) for r in self._records]),
                "held": np.stack([np.asarray(r.held, dtype=np.int64) for r in self._records]),
                "item_pose": np.stack([np.asarray(r.item_pose) for r in self._records]),
                "item_vel": np.stack([np.asarray(r.item_vel) for r in self._records]),
                "region_T": np.stack([np.asarray(r.region_T) for r in self._records]),
                "margins": margins,
                "robot_item_dist": np.stack(
                    [np.asarray(r.robot_item_dist, dtype=np.float64) for r in self._records]
                ),
                "robot_item_force": np.stack(
                    [np.asarray(r.robot_item_force, dtype=np.float64) for r in self._records]
                ),
                "visibility": np.stack(
                    [np.asarray(r.visibility, dtype=np.float64) for r in self._records]
                ),
                "token_masks": np.stack(
                    [np.asarray(r.token_masks, dtype=np.uint8) for r in self._records]
                ),
                "action_source": np.array(
                    [r.action_source for r in self._records], dtype=np.int8
                ),
                "time_scale": np.array(
                    [r.time_scale for r in self._records], dtype=np.float64
                ),
                # self-describing traces: margins columns / item rows
                "item_ids": np.array([d["item_id"] for d in meta.items], dtype=str),
                "rule_ids": np.array(rule_ids, dtype=str),
            }
            particle_keys = sorted(
                {k for r in self._records for k in (r.particle_pos or {})}
            )
            for key in particle_keys:
                arrays[f"particle_pos__{key}"] = np.stack(
                    [np.asarray(r.particle_pos[key]) for r in self._records]
                )
            np.savez_compressed(npz_path, **arrays)

        json_path = self.write_meta(meta)
        return npz_path, json_path

    def write_meta(self, meta: EpisodeMeta, error: str | None = None) -> Path:
        """meta.json; an optional top-level "error" key records sweep failures."""
        payload = _jsonable(asdict(meta))
        if error is not None:
            payload["error"] = str(error)
        json_path = self.out_dir / "meta.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return json_path


# ---------------------------------------------------------------------------
# Overlay video (PLAN v3 §2.6)
# ---------------------------------------------------------------------------


def draw_overlay(frame_rgb: np.ndarray, lines: list[str], flash: bool = False) -> np.ndarray:
    """RGB frame with the text overlay (and optional red border) drawn on a copy.

    Shared by OverlayVideoWriter and the runner's keyframe writer so videos
    and PNGs render identical margins.
    """
    import cv2

    img = np.ascontiguousarray(np.asarray(frame_rgb).copy())
    scale = max(img.shape[0] / 240.0, 1.0)
    # shrink the text so every line fits the frame height (per-rule margin
    # lists run 10-15 lines); lines starting with "!" are drawn red
    fit = (img.shape[0] - 8) / max(1, len(lines)) / (20.0 * scale)
    scale_txt = scale * min(1.0, max(0.45, fit))
    thick = max(1, int(round(scale_txt)))
    y = int(round(18 * scale_txt))
    dy = int(round(20 * scale_txt))
    for line in lines:
        line = str(line)
        fg = (255, 255, 255)
        if line.startswith("!"):
            line, fg = line[1:], (255, 60, 60)
        for color, th in (((0, 0, 0), thick + 2), (fg, thick)):
            cv2.putText(
                img,
                line,
                (6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * scale_txt,
                color,
                th,
                cv2.LINE_AA,
            )
        y += dy
    if flash:
        cv2.rectangle(
            img,
            (0, 0),
            (img.shape[1] - 1, img.shape[0] - 1),
            (40, 40, 235),
            max(2, 2 * thick),
        )
    return img


class OverlayVideoWriter:
    """mp4 from observer frames with a text overlay.

    flash() arms a red border for the next `flash_frames` frames (the
    first-violation marker). Lines are caller-supplied; the runner passes the
    active rule phrase, its margin, the min margin over rules, and the time
    scale.
    """

    def __init__(self, out_path, fps: float = 25.0, flash_frames: int = 8):
        self.out_path = Path(out_path)
        self.fps = float(fps)
        self.flash_frames = int(flash_frames)
        self._frames: list[np.ndarray] = []
        self._flash_left = 0
        self.path: Path | None = None

    def flash(self) -> None:
        """Arm the red border for the next frames (first violation)."""
        self._flash_left = self.flash_frames

    def add(self, frame_rgb: np.ndarray, lines: list[str]) -> None:
        flash = self._flash_left > 0
        if flash:
            self._flash_left -= 1
        self._frames.append(draw_overlay(frame_rgb, lines, flash=flash))

    def close(self) -> Path | None:
        if self._frames:
            self.path = save_video(self._frames, self.out_path, self.fps)
        self._frames = []
        return self.path
