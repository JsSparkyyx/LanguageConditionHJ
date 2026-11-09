"""The policy's state and the named rule's margin, both from one measurement per tick.

``NamedProbe.measure()`` computes the named object's distance to the robot exactly as
``robot_distance.robot_item_distances`` does for that one object (arm collision spheres and the held payload against
the object's collision shapes, a held named object reads at most -0.5 cm), split by part: per arm link (13 per
arm) and per held object. The avoid margin is that distance minus 1 cm (``margins.m_avoid``). The stay margin comes
from the runner's own ``one_margin`` (3 cm of horizontal displacement, tilt counted as movement).

``build_obs`` lays out the state (residual/PLAN.md, "State"); every position is in the world frame:

- robot: arm joints q [12] and their velocities [12], gripper openings [2], link positions (the centre of each
  link's collision spheres) [26 x 3], per arm the held object (flag, position, rotation as 6D, half extents) [2 x 13];
- planner: a_nom [12] and the plan rows 5, 10, 15, 20, 25 ticks ahead as offsets from the live joints [5 x 12],
  per arm whether a plan plays [2];
- the named object: position [3], rotation 6D [6], half extents [3], linear and angular velocity [6], displacement
  from its start [3], tilt [1], stay flag (1 = don't move, 0 = don't touch) [1];
- per link and per held object: distance to the named object (clipped at ``dist_clip``) [28] and the unit direction
  toward the closest point of its box [28 x 3];
- w [1].
"""

from __future__ import annotations

import numpy as np

from . import ensure_paths
from .config import N_ARM

ensure_paths()


def rot6d(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=np.float64)[:3, :2].T.reshape(-1)


def _quat_to_R(q_wxyz) -> np.ndarray:
    w, x, y, z = [float(v) for v in q_wxyz]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _closest_on_obb(p: np.ndarray, origin, rot, half) -> np.ndarray:
    local = rot.T @ (p - origin)
    return origin + rot @ np.clip(local, -half, half)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


class NamedProbe:
    """Per-episode measuring tool for the named object (see module docstring)."""

    def __init__(self, runner, named, cfg):
        from main.item_envs import robot_distance as rd

        self.rd = rd
        self.runner, self.named, self.cfg = runner, named, cfg
        self.env = runner.env
        self.specs = list(runner.items)
        self.k = int(named.item_index)
        self.spec = self.specs[self.k]
        self.rule = runner.rules[named.rule_index]
        self.cehj, load = rd._cehj()
        self.spheres = rd._item_collision_spheres(self.env.embodiment, load)
        self.shape = rd._shape(None, self.spec)
        self.link_names: list[tuple[str, str]] | None = None   # (side, link) in a fixed order, set on first use
        self.start_pos = self._T()[:3, 3].copy()   # the stay rule's reference is the pose at t = 0 too
        self._payload_shapes: dict = {}

    # ------------------------------------------------------------------------
    def _T(self) -> np.ndarray:
        from main.item_envs.runner import _world_T

        return _world_T(self.spec)

    def _arm_spheres(self):
        """[(side, link, center, radius)] of both arms, the sphere set of robot_distance._arm_spheres_world."""
        out = []
        for side in ("left", "right"):
            entity = self.env.robot.left_entity if side == "left" else self.env.robot.right_entity
            arm = self.cehj._arm_collision_spheres(self.env, side, self.spheres)
            for c, r, name in self.cehj.spheres_with_names(entity, arm):
                out.append((side, name, np.asarray(c, dtype=np.float64), float(r)))
        return out

    def _payload_shape(self, kind, ref):
        key = (kind, ref if kind == "spec" else id(ref))
        if key not in self._payload_shapes:
            self._payload_shapes[key] = (self.rd._shape(None, self.specs[ref]) if kind == "spec"
                                         else self.rd._shape(ref))
        return self._payload_shapes[key]

    # ------------------------------------------------------------------------
    def measure(self) -> dict:
        from main.item_envs import item_shapes
        from main.item_envs.runner import _velocity6

        rd, cehj = self.rd, self.cehj
        held_idx, payloads = rd._held_info(self.env, self.specs, cehj)
        origin, rot, half = rd._spec_obb(self.spec)
        T = self._T()

        spheres = self._arm_spheres()
        if self.link_names is None:
            seen = []
            for side, name, _c, _r in spheres:
                if (side, name) not in seen:
                    seen.append((side, name))
            self.link_names = seen
        groups: dict[tuple[str, str], list[int]] = {ln: [] for ln in self.link_names}
        for i, (side, name, _c, _r) in enumerate(spheres):
            if (side, name) in groups:
                groups[(side, name)].append(i)
        centers = np.stack([s[2] for s in spheres])
        radii = np.asarray([s[3] for s in spheres])

        n_links = len(self.link_names)
        link_d = np.full(n_links, np.inf)
        link_dir = np.zeros((n_links, 3))
        link_pos = np.zeros((n_links, 3))
        for j, ln in enumerate(self.link_names):
            idx = groups[ln]
            if not idx:
                continue
            c, r = centers[idx], radii[idx]
            link_pos[j] = c.mean(0)
            if self.shape is not None:
                d, a = item_shapes.spheres_min_distance(c, r, self.shape)
            else:   # no collision shape: the OBB, as robot_distance's fallback
                ds = [cehj.point_obb_signed_distance(ci, origin, rot, half) - ri for ci, ri in zip(c, r)]
                a = int(np.argmin(ds)); d = float(ds[a])
            link_d[j] = float(d)
            ca = c[max(int(a), 0)]
            link_dir[j] = _unit(_closest_on_obb(ca, origin, rot, half) - ca)
            if not np.any(link_dir[j]):
                link_dir[j] = _unit(origin - ca)
        d_arm = float(link_d.min()) if n_links else float("inf")

        # held payload, exactly robot_item_distances' loop for item k
        held = []   # (side, d, dir, pos, R, half)
        d_held = float("inf")
        for kind, ref in payloads:
            if kind == "spec" and ref == self.k:
                continue
            try:
                p_shape = self._payload_shape(kind, ref) if self.shape is not None else None
                if p_shape is not None:
                    d = item_shapes.shape_distance(p_shape, self.shape)
                elif kind == "spec":
                    d = rd._spec_obb_distance(cehj, rd._spec_obb(self.specs[ref]), origin, rot, half)
                else:
                    d = cehj._actor_obb_distance(ref, origin, rot, half)
            except (TypeError, AttributeError):
                continue
            d = float(d)
            d_held = min(d_held, d)
            if kind == "spec":
                p_o, p_R, p_h = rd._spec_obb(self.specs[ref])
            else:
                p_o, p_R, p_h = cehj._obb_from_actor(ref)
            p_o = np.asarray(p_o, dtype=np.float64)
            held.append((d, _unit(_closest_on_obb(p_o, origin, rot, half) - p_o), p_o, np.asarray(p_R), np.asarray(p_h)))
        dist = min(d_arm, d_held)
        named_held = self.k in held_idx
        if named_held:
            dist = min(dist, -rd.HELD_CONTACT_DEPTH)

        margin = self._margin(dist)
        vel = _velocity6(self.spec)
        from main.item_envs import geometry

        tilt = float(geometry.tilt_angle(T, self.spec.up_axis_local))
        return {
            "dist": float(dist), "d_arm": d_arm, "d_held": d_held, "named_held": bool(named_held),
            "margin": float(margin), "link_d": link_d, "link_dir": link_dir, "link_pos": link_pos,
            "held": held, "T": T, "vel": vel, "tilt": tilt, "obb": (np.asarray(origin), np.asarray(rot),
                                                                       np.asarray(half)),
        }

    def _margin(self, dist: float) -> float:
        from main.item_envs import margins as margin_fns
        from main.item_envs.runner import one_margin

        if self.named.kind == "avoid":
            return float(margin_fns.m_avoid(dist, margin=self.rule.params.get("margin", 0.0)))
        indexed = [(self.k, self.spec)]
        return float(one_margin(self.rule, indexed, [], None, self.runner._stay_ref))


def _held_per_arm(m: dict, ee_xyz: dict) -> np.ndarray:
    """[2 x 13]: per arm (flag, position, rotation 6D, half extents) of the held object nearest that hand."""
    out = np.zeros((2, 13))
    taken = set()
    for slot, side in enumerate(("left", "right")):
        best, bj = np.inf, -1
        for j, (_d, _dir, p, _R, _h) in enumerate(m["held"]):
            if j in taken:
                continue
            dd = float(np.linalg.norm(p - ee_xyz[side]))
            if dd < best and dd < 0.25:
                best, bj = dd, j
        if bj >= 0:
            taken.add(bj)
            _d, _dir, p, R, h = m["held"][bj]
            out[slot] = np.concatenate([[1.0], p, rot6d(R), h])
    return out


def held_dist_per_arm(m: dict, ee_xyz: dict, clip: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.full(2, clip)
    dirs = np.zeros((2, 3))
    taken = set()
    for slot, side in enumerate(("left", "right")):
        best, bj = np.inf, -1
        for j, (_d, _dir, p, _R, _h) in enumerate(m["held"]):
            if j in taken:
                continue
            dd = float(np.linalg.norm(p - ee_xyz[side]))
            if dd < best and dd < 0.25:
                best, bj = dd, j
        if bj >= 0:
            taken.add(bj)
            d[slot] = min(float(m["held"][bj][0]), clip)
            dirs[slot] = m["held"][bj][1]
    return d, dirs


def robot_reading(env) -> dict:
    robot = env.robot
    out = {}
    for side in ("left", "right"):
        ent = robot.left_entity if side == "left" else robot.right_entity
        joints = robot.left_arm_joints if side == "left" else robot.right_arm_joints
        active = ent.get_active_joints()
        qpos, qvel = ent.get_qpos(), ent.get_qvel()
        idx = [active.index(j) for j in joints]
        out[f"q_{side}"] = np.asarray([qpos[i] for i in idx], dtype=np.float64)
        out[f"qd_{side}"] = np.asarray([qvel[i] for i in idx], dtype=np.float64)
        ee = robot.get_left_ee_pose() if side == "left" else robot.get_right_ee_pose()
        out[f"ee_{side}"] = np.asarray(ee, dtype=np.float64).reshape(-1)[:7]
    out["grip"] = np.array([robot.get_left_gripper_val(), robot.get_right_gripper_val()], dtype=np.float64)
    return out


def build_obs(cfg, probe: NamedProbe, m: dict, rob: dict, view: dict, w: float) -> np.ndarray:
    ee_xyz = {s: rob[f"ee_{s}"][:3] for s in ("left", "right")}
    T = m["T"]
    pos = T[:3, 3]
    clip = float(cfg.dist_clip)
    held_d, held_dir = held_dist_per_arm(m, ee_xyz, clip)
    parts = [
        rob["q_left"], rob["q_right"], rob["qd_left"], rob["qd_right"], rob["grip"],
        m["link_pos"].reshape(-1),
        _held_per_arm(m, ee_xyz).reshape(-1),
        view["a_nom"], view["future"].reshape(-1), view["active"],
        pos, rot6d(T[:3, :3]), probe.spec.half_extents, m["vel"], pos - probe.start_pos, [m["tilt"]],
        [1.0 if probe.named.kind == "stay" else 0.0],
        np.minimum(m["link_d"], clip), held_d, m["link_dir"].reshape(-1), held_dir.reshape(-1),
        [float(w)],
    ]
    obs = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])
    if not np.all(np.isfinite(obs)):
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
    return obs.astype(np.float32)


def obs_dim(cfg, n_links: int = 26) -> int:
    n_fut = len(cfg.future_ticks)
    return (4 * N_ARM + 2 + 3 * n_links + 26 + 2 * N_ARM * (1 + n_fut) + 2 + 3 + 6 + 3 + 6 + 3 + 1 + 1
            + n_links + 2 + 3 * (n_links + 2) + 1)
