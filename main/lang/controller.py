"""ResumeViaController: after a filter release inside a via path, the nominal CONTINUES the path (2026-11-17).

The item tasks drag along a "via path": dense Cartesian waypoints joined into one joint trajectory and timed by a
profile (envs/main/item_envs/controller.py ``_plan_via``, via_path.py). The task keeps the waypoints and the profile
of the WHOLE move set while the move runs. When the filter intervenes and then releases, the tick controller replans
to the same target, and RiskTickController plans through every original waypoint again, with the original time law:

- the tray is first dragged BACK to the first waypoint (about 3 cm from the drag's start) and then forward again;
- the original profile's length is played on this longer path, so every speed is scaled by new length / old length
  (measured 1.9-2.2 x on T8, review of 2026-11-17);
- the time law starts from rest while the arm may still move: the first rows hold the arm at its live position,
  which is a hard stop.

So a filter that slows the tray correctly and hands back was followed by a reversal and a harsher drag: in the 1115
T8 rings slow drags, which never tip anything by themselves, tipped the glass in 10 of 43 episodes with an
intervention. This subclass changes only the REPLAN of a via path (``_count_next_plan_as_replan``), nothing under
envs/:

1. keep the waypoints still ahead of the live end effector along the path (remaining_via);
2. time the remaining length R with the same behaviour (resume_law): from the arm's CURRENT speed along the path,
   speed up to the profile's cruise speed with the original peak acceleration, cruise, brake at the pad over the
   original stop distance. When R is too short to reach the cruise speed, the peak speed is lowered so that neither
   acceleration exceeds the original's. A ``speed_change`` profile (the recover behaviours) keeps its slow-down
   where it was along the drag, or continues at its second speed when the slow-down had begun.

Example (medium drag: 0.45 m/s, 15 cm speed-up, 4 cm brake, 35 cm long): released 20 cm along with the tray at
0.2 m/s, the new plan covers the remaining 15 cm, starts at 0.2 m/s, reaches 0.44 m/s and brakes over the last
4 cm as the nominal does. Before: back 17 cm to the first waypoint, then the full drag again at 1.5 x the speed.

Used when ``LangConfig.resume_via_replan`` is set (controller_for); off by default, so earlier runs keep the
controller they trained and were evaluated with.
"""

from __future__ import annotations

import numpy as np

from . import ensure_item_path

ensure_item_path()

from main.item_envs.controller import RiskTickController  # noqa: E402
from main.item_envs.via_path import PHYSICS_DT, time_law  # noqa: E402

AHEAD_EPS = 0.005     # m: a waypoint closer than this ahead of the end effector counts as passed
GEOM_DURATION = 4.0   # s: the placeholder law that samples the remaining path's geometry (1001 rows)


def remaining_via(ee, via, target):
    """The via poses still ahead of the end effector, and the length left.

    ee [3]: the live end-effector position; via: the move's intermediate poses ([x, y, z, quat], in path order);
    target: its final pose. The path is the polyline via[0] .. via[-1], target. The end effector is projected on
    it (the closest point, arc length s); a via pose is kept when its arc length exceeds s + AHEAD_EPS.
    -> (kept via poses, R = |ee - first kept point| + the polyline's length from there to the target)."""
    ee = np.asarray(ee, np.float64).reshape(-1)[:3]
    pts = np.stack([np.asarray(p, np.float64).reshape(-1)[:3] for p in list(via) + [target]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    best, s_ee = np.inf, 0.0
    for k in range(len(pts) - 1):          # closest point of every segment
        d = pts[k + 1] - pts[k]
        u = float(np.clip(np.dot(ee - pts[k], d) / max(float(np.dot(d, d)), 1e-12), 0.0, 1.0))
        dist = float(np.linalg.norm(ee - (pts[k] + u * d)))
        if dist < best - 1e-12:
            best, s_ee = dist, float(arc[k] + u * seg[k])
    if len(pts) == 1:
        s_ee = 0.0
    keep = [k for k in range(len(pts) - 1) if arc[k] > s_ee + AHEAD_EPS]
    first = pts[keep[0]] if keep else pts[-1]
    rest = float(arc[-1] - (arc[keep[0]] if keep else arc[-1]))
    return [list(via[k]) for k in keep], float(np.linalg.norm(ee - first) + rest)


def _seg_time(kind: str, a: float, b: float, dist: float) -> float:
    """Duration of a segment: a smoothstep speed ramp a -> b (mean speed (a + b) / 2) or a cruise at a."""
    if dist <= 1e-9:
        return 0.0
    v = 0.5 * (a + b) if kind == "ramp" else a
    return dist / v if v > 1e-9 else 0.0


def _seg_dist(kind: str, a: float, b: float, T: float, t):
    """Distance covered t seconds into a segment of duration T (as via_path's ramps: the speed follows a smoothstep,
    peak acceleration 0.75 |b^2 - a^2| / distance)."""
    if kind != "ramp":
        return a * t
    x = np.clip(t / T, 0.0, 1.0)
    return a * T * x + (b - a) * T * (x ** 3 - 0.5 * x ** 4)


def resume_segments(R: float, v0: float, profile: dict, travelled: float) -> list:
    """The speed plan of the remaining length R as (kind, v_from, v_to, distance) segments, from the current speed
    v0. profile: the move's own (via_path kinds hard_stop / speed_change: v, accel_dist, stop_dist, and v2, switch,
    change_dist); travelled: m of the original length already covered."""
    v = float(profile["v"])
    da, ds = float(profile.get("accel_dist", 0.05)), float(profile.get("stop_dist", 0.02))
    v2 = profile.get("v2")
    sw, dc = float(profile.get("switch", da)), float(profile.get("change_dist", 0.05))
    pending = profile.get("kind") == "speed_change" and v2 is not None and abs(float(v2) - v) > 1e-9
    if pending and travelled >= sw:          # the slow-down had begun: go on at the second speed
        v, pending = float(v2), False
        da = da * (v / float(profile["v"])) ** 2     # the same peak acceleration for the lower cruise speed
        ds = float(profile.get("stop_dist", 0.02))   # the profile brakes from v2 over stop_dist
    v0 = float(np.clip(v0, 0.0, 1.5 * max(float(profile.get("v_ref", v)), v, 1e-9)))
    a_acc = 0.75 * v * v / max(da, 1e-9)             # the profile's peak accelerations
    v_brake = float(v2) if pending else v
    a_stop = 0.75 * v_brake * v_brake / max(ds, 1e-9)

    def d_ramp(a, b, acc):
        return 0.75 * abs(b * b - a * a) / max(acc, 1e-9)

    if pending:
        d1 = d_ramp(v0, v, a_acc)
        gap = max(sw - travelled, d1)                # the slow-down starts where it was, or right after the speed-up
        if gap + dc + ds <= R:
            return [("ramp", v0, v, d1), ("cruise", v, v, gap - d1), ("ramp", v, float(v2), dc),
                    ("cruise", float(v2), float(v2), R - gap - dc - ds), ("ramp", float(v2), 0.0, ds)]
        v, a_stop = float(v2), 0.75 * float(v2) ** 2 / max(ds, 1e-9)   # no room for it: continue at the second speed
        a_acc = max(a_acc, 1e-9)
    d1, d2 = d_ramp(v0, v, a_acc), d_ramp(v, 0.0, a_stop)
    if d1 + d2 <= R:
        return [("ramp", v0, v, d1), ("cruise", v, v, R - d1 - d2), ("ramp", v, 0.0, d2)]
    # too short to reach the cruise speed: the peak speed vp with d(v0 -> vp) + d(vp -> 0) = R at the same accelerations
    if v0 <= v:
        vp2 = (R + 0.75 * v0 * v0 / a_acc) / (0.75 / a_acc + 0.75 / a_stop)
        if vp2 >= v0 * v0:
            vp = float(np.sqrt(vp2))
            return [("ramp", v0, vp, d_ramp(v0, vp, a_acc)), ("ramp", vp, 0.0, d_ramp(vp, 0.0, a_stop))]
    return [("ramp", v0, 0.0, R)]                    # even braking from v0 needs all of R: brake over it


def capped_profile(profile: dict, cap: float) -> dict:
    """The drag profile with its cruise speed capped at ``cap`` m/s (2026-11-18, "slow down, then finish slowly"):
    one cruise speed v' = min(v, v2, cap), the original peak acceleration (speed-up distance x (v' / v)^2), the
    original stop distance (so the brake is gentler: 0.75 v'^2 / stop_dist). v_ref keeps the original speed, so an
    arm handed over faster than v' slows to it (resume_segments) instead of being clipped. Example: medium (0.45 m/s,
    15 cm speed-up, 4 cm brake) capped at 0.15 m/s: speed-up 1.7 cm, brake peak 0.42 m/s^2 instead of 3.8."""
    v = float(profile["v"])
    vc = min(v, float(profile.get("v2") or v), float(cap))
    out = {k: x for k, x in profile.items() if k not in ("v2", "switch", "change_dist")}
    out.update({"kind": "hard_stop", "v": vc, "v_ref": v,
                "accel_dist": float(profile.get("accel_dist", 0.05)) * (vc / max(v, 1e-9)) ** 2})
    return out


def resume_law(R: float, v0: float, profile: dict, travelled: float, dt: float = PHYSICS_DT) -> np.ndarray:
    """Distance s(t) [n] along the remaining length R on one time grid of step dt: s[0] = 0, s[-1] = R
    (resume_segments)."""
    if R <= 1e-6:
        return np.array([0.0, max(R, 0.0)])
    segs = [(k, a, b, d, _seg_time(k, a, b, d)) for k, a, b, d in resume_segments(R, v0, profile, travelled)]
    segs = [x for x in segs if x[4] > 0.0]
    if not segs:
        return np.array([0.0, R])
    t_end = np.cumsum([x[4] for x in segs])
    t_start = np.concatenate([[0.0], t_end[:-1]])
    s_start = np.concatenate([[0.0], np.cumsum([x[3] for x in segs])[:-1]])
    t = np.arange(0.0, float(t_end[-1]), dt)
    j = np.minimum(np.searchsorted(t_end, t, side="right"), len(segs) - 1)
    s = np.zeros_like(t)
    for q, (kind, a, b, _, T) in enumerate(segs):
        m = j == q
        s[m] = s_start[q] + _seg_dist(kind, a, b, T, t[m] - t_start[q])
    s = np.append(np.clip(s, 0.0, R), R)
    s = np.maximum.accumulate(s)
    s[0] = 0.0
    return s


def retime(pos: np.ndarray, lam: np.ndarray, s: np.ndarray, R: float, dt: float = PHYSICS_DT):
    """Joint rows of a path sampled at fractions ``lam`` (rows of ``pos``), re-sampled at the distances s(t) / R.
    -> (position [n, dof], velocity [n, dof]); first row = pos[0], last = pos[-1], the last velocity zero."""
    lam_new = np.clip(np.asarray(s, np.float64) / max(R, 1e-12), 0.0, 1.0)
    lam = np.asarray(lam, np.float64)
    order = np.concatenate([[True], np.diff(lam) > 1e-12])      # np.interp needs increasing fractions
    out = np.stack([np.interp(lam_new, lam[order], pos[order, j]) for j in range(pos.shape[1])], axis=1)
    out[0], out[-1] = pos[0], pos[-1]
    vel = np.gradient(out, dt, axis=0) if len(out) > 2 else np.zeros_like(out)
    vel[-1] = 0.0
    return out, vel


def path_speed(pos: np.ndarray, lam: np.ndarray, qvel: np.ndarray, R: float) -> float:
    """The arm's current speed along the path (m/s): its joint velocity projected on the path's joint-space tangent
    at the start (d pos / d lam over the first 2 % of the path), times the length."""
    k = int(np.searchsorted(lam, 0.02))
    k = min(max(k, 1), len(pos) - 1)
    g = (pos[k] - pos[0]) / max(float(lam[k] - lam[0]), 1e-9)
    return float(max(np.dot(np.asarray(qvel, np.float64), g) / max(float(np.dot(g, g)), 1e-12), 0.0) * R)


class ResumeViaController(RiskTickController):
    """RiskTickController whose replans inside a via path continue it (module docstring)."""

    controller_name = "risk_tick_resume"
    resume_speed_cap = None   # m/s: a resumed drag cruises at most this fast (capped_profile); None = the profile's

    def _plan_via(self, arm: str, target_pose, task, plan_kw):
        via0 = getattr(task, "plan_via", None)
        if not getattr(self, "_count_next_plan_as_replan", False) or via0 is None:
            return super()._plan_via(arm, target_pose, task, plan_kw)
        prof0 = getattr(task, "plan_profile", None)
        ee = np.asarray(task.get_arm_pose(arm), dtype=np.float64).reshape(7)[:3]
        via, R = remaining_via(ee, via0, target_pose)
        geom = {"kind": "minjerk", "duration": GEOM_DURATION}
        timed = isinstance(prof0, dict) and "v" in prof0 and "length" in prof0
        if not timed:   # a duration-timed via path (other tasks): the remaining share of its duration, from rest
            total = remaining_via(np.asarray(via0[0], np.float64)[:3], via0, target_pose)[1] if len(via0) else R
            geom = dict(prof0 or {"kind": "minjerk", "duration": 1.0})
            if "duration" in geom:
                geom["duration"] = float(geom["duration"]) * float(np.clip(R / max(total, 1e-9), 0.2, 1.0))
        task.plan_via, task.plan_profile = via, geom
        try:
            res = super()._plan_via(arm, target_pose, task, plan_kw)
        finally:
            task.plan_via, task.plan_profile = via0, prof0
        if not timed or not res or res.get("status") != "Success":
            return res
        pos = np.asarray(res["position"], dtype=np.float64)
        lam = time_law(geom, PHYSICS_DT)[0]
        if len(lam) != len(pos):
            lam = np.linspace(0.0, 1.0, len(pos))
        mplib = self.robot.left_mplib_planner if arm == "left" else self.robot.right_mplib_planner
        entity = self.robot.left_entity if arm == "left" else self.robot.right_entity
        idx = np.asarray(mplib.planner.move_group_joint_indices, dtype=np.int64)
        qvel = np.asarray(entity.get_qvel(), dtype=np.float64)[idx]
        v0 = path_speed(pos, lam, qvel, R)
        travelled = max(float(prof0["length"]) - R, 0.0)
        cap = self.resume_speed_cap
        prof = prof0 if cap is None else capped_profile(prof0, float(cap))
        s = resume_law(R, v0, prof, travelled, PHYSICS_DT)
        new_pos, new_vel = retime(pos, lam, s, R, PHYSICS_DT)
        info = getattr(task, "episode_info", None)
        if isinstance(info, dict):
            info.setdefault("via_resumes", []).append(
                {"remaining": round(R, 4), "travelled": round(travelled, 4), "v0": round(v0, 4),
                 "kept_via": len(via), "of": len(via0), "duration": round(len(new_pos) * PHYSICS_DT, 3),
                 "speed_cap": cap})
        print(f"[risk_tick] plan #{self._n_plan} {arm} via path RESUMED: {len(via)} of {len(via0)} waypoints ahead, "
              f"{R:.3f} m left of {float(prof0['length']):.3f}, from {v0:.2f} m/s, {len(new_pos) * PHYSICS_DT:.2f}s"
              + (f", capped at {float(cap):.2f} m/s" if cap is not None else ""))
        return {"status": "Success", "position": new_pos, "velocity": new_vel, "_item_profiled": True}


def controller_for(cfg):
    """The tick controller of a rollout: ResumeViaController when the config asks for it, else RiskTickController."""
    return ResumeViaController if bool(getattr(cfg, "resume_via_replan", False)) else RiskTickController
