"""Non-simulator checks of the tray_slide pieces and of the T1 path staying as it was (run from the repo root):

    OMP_NUM_THREADS=1 python -m pytest -q residual/tests/test_tray_nonsim.py
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from residual.config import ResidualConfig, apply_sets

T1_EVAL_KEYS = {
    "n", "success", "truncated", "grasp_ee_off_median", "grasp_joint_off_median",
    *[f"{g}/{k}" for g in ("in_way", "out_of_way")
      for k in ("n", "broken", "success", "safe_success", "correction_rms", "replans", "named_held", "broken_avoid",
                "broken_stay")],
}
T1_TRAIN_GROUP_KEYS = {f"{g}/{k}" for g in ("in_way", "out_of_way") for k in ("share", "broken", "success",
                                                                               "correction_rms")}


# ------------------------------------------------------------------------------------------------- config ----
def test_config_tray_fields_and_t1_defaults():
    d = ResidualConfig()
    assert d.task == "selective_avoid" and d.profiled_replan == "resync" and d.tray_behaviours == ("fast",)
    cfg = apply_sets(ResidualConfig(), ['task="tray_slide"', 'tray_behaviours=["fast","medium"]', "eval_pairs=3"])
    assert cfg.task == "tray_slide" and cfg.tray_behaviours == ("fast", "medium") and cfg.eval_pairs == 3
    assert ResidualConfig.from_dict(cfg.to_dict()) == cfg
    # a run saved before these fields keeps loading, with today's defaults
    old = {k: v for k, v in ResidualConfig().to_dict().items()
           if k not in ("profiled_replan", "resync_threshold", "tray_behaviours", "eval_pairs")}
    assert ResidualConfig.from_dict(old) == ResidualConfig()
    for bad in (['task="push_zone"'], ['tray_behaviours=["warp"]'], ["resync_threshold=0.3"],
                ['profiled_replan="maybe"']):
        with pytest.raises(ValueError):
            apply_sets(ResidualConfig(), bad)


# ------------------------------------------------------------------------------------- controller: resync ----
def _ctrl(cfg):
    from residual.controller import ResidualTickController

    c = object.__new__(ResidualTickController)   # no Env: only the bookkeeping the tested methods touch
    c.cfg = cfg
    c.stats = {"policy_ticks": 0, "replans": 0, "replans_fixed": 0, "replans_deviation": 0, "replan_failures": 0,
               "replan_time": 0.0, "max_dev": 0.0, "resyncs": 0, "resync_rows": 0}
    c.replan_log = []
    c.anchor = {"left": np.zeros(6), "right": np.zeros(6)}
    return c


def _line_plan(n=200, dof=6, profiled=True):
    pos = np.linspace(0.0, 1.0, n)[:, None] * np.ones((1, dof))   # joint j moves 0 -> 1 rad along the plan
    plan = {"status": "Success", "position": pos, "velocity": np.zeros_like(pos), "_cehj_target_pose": [0] * 7}
    if profiled:
        plan["_item_profiled"] = True
    return plan


def test_nearest_row():
    from residual.controller import nearest_row

    pos = np.linspace(0.0, 1.0, 101)[:, None] * np.ones((1, 3))
    j, d = nearest_row(pos, np.full(3, 0.305), ref=80)
    assert j in (30, 31) and d <= 0.005 + 1e-12
    # a plateau (rows that do not move): the row nearest the reference wins
    pos2 = np.concatenate([np.zeros((10, 2)), np.ones((10, 2))])
    assert nearest_row(pos2, np.ones(2), ref=0)[0] == 10
    assert nearest_row(pos2, np.ones(2), ref=19)[0] == 19


def test_resync_moves_the_cursor_back_and_never_replans(monkeypatch):
    cfg = ResidualConfig()
    c = _ctrl(cfg)
    plan = _line_plan()
    pos = plan["position"]
    q_lag = pos[99].copy()                     # the arm sits at row 99 while the cursor says 150
    monkeypatch.setattr(type(c), "_arm_qpos", lambda self: {"left": q_lag, "right": np.zeros(6)})
    monkeypatch.setattr(type(c), "_replan_arm", lambda self, a, p: pytest.fail("a via-path plan must not replan"))
    plans = {"left": plan, "right": None}
    cursor = {"left": 150, "right": 0}
    since = {"left": 3, "right": 0}
    c._maybe_replan(plans, cursor, since)
    assert cursor["left"] == 100 and c.stats["resyncs"] == 1 and c.stats["resync_rows"] == 50
    assert plans["left"] is plan and c.stats["replans"] == 0 and since["left"] == 0
    # within the threshold: nothing happens
    cursor["left"] = 101
    c._maybe_replan(plans, cursor, since)
    assert cursor["left"] == 101 and c.stats["resyncs"] == 1


def test_resync_never_jumps_forward_or_on_a_sideways_offset(monkeypatch):
    """Smoke 11-13: exploration noise made the cursor random-walk, mostly forward (drag at 1.7 m/s). Only a real lag
    along the path moves the cursor, and only back."""
    c = _ctrl(ResidualConfig())
    plan = _line_plan()
    pos = plan["position"]
    state = {"q": pos[170].copy()}               # the arm AHEAD of the cursor (row 149)
    monkeypatch.setattr(type(c), "_arm_qpos", lambda self: {"left": state["q"], "right": np.zeros(6)})
    cursor = {"left": 150, "right": 0}
    c._maybe_replan({"left": plan, "right": None}, cursor, {"left": 0, "right": 0})
    assert cursor["left"] == 150 and c.stats["resyncs"] == 0
    state["q"] = pos[149] + 0.05 * np.array([1, -1, 1, -1, 1, -1])   # off the path sideways, on time
    c._maybe_replan({"left": plan, "right": None}, cursor, {"left": 0, "right": 0})
    assert cursor["left"] == 150 and c.stats["resyncs"] == 0
    state["q"] = pos[120] + 0.05 * np.array([1, -1, 1, -1, 1, -1])   # sideways AND lagging: the lag is removed
    c._maybe_replan({"left": plan, "right": None}, cursor, {"left": 0, "right": 0})
    assert cursor["left"] == 121 and c.stats["resyncs"] == 1


def test_resync_past_the_end_keeps_the_move_going(monkeypatch):
    c = _ctrl(ResidualConfig())
    plan = _line_plan()
    q = plan["position"][180].copy()           # the cursor has passed the end, the arm has not
    monkeypatch.setattr(type(c), "_arm_qpos", lambda self: {"left": q, "right": np.zeros(6)})
    cursor = {"left": 205, "right": 0}
    c._maybe_replan({"left": plan, "right": None}, cursor, {"left": 0, "right": 0})
    assert cursor["left"] == 181
    # at the end within the threshold: the cursor stays past the end, the move finishes
    q[:] = plan["position"][-1]
    cursor["left"] = 205
    c._maybe_replan({"left": plan, "right": None}, cursor, {"left": 0, "right": 0})
    assert cursor["left"] == 205 and c.stats["resyncs"] == 1


def test_t1_plans_still_replan(monkeypatch):
    """A plan without _item_profiled (every T1 plan) goes through _replan_arm exactly as before."""
    c = _ctrl(ResidualConfig())
    plan = _line_plan(profiled=False)
    new = _line_plan(n=120, profiled=False)
    q = plan["position"][99].copy()
    calls = []
    monkeypatch.setattr(type(c), "_arm_qpos", lambda self: {"left": q, "right": np.zeros(6)})
    monkeypatch.setattr(type(c), "_replan_arm", lambda self, a, p: calls.append(a) or new)
    plans = {"left": plan, "right": None}
    cursor = {"left": 150, "right": 0}
    c._maybe_replan(plans, cursor, {"left": 0, "right": 0})
    assert calls == ["left"] and plans["left"] is new and cursor["left"] == 0 and c.stats["resyncs"] == 0
    # and a finished T1 plan is never looked at
    calls.clear()
    c._maybe_replan({"left": plan, "right": None}, {"left": 205, "right": 0}, {"left": 0, "right": 0})
    assert calls == []


def test_profiled_replan_switch_uses_the_old_replan(monkeypatch):
    c = _ctrl(ResidualConfig(profiled_replan="replan"))
    plan = _line_plan()
    q = plan["position"][99].copy()
    calls = []
    monkeypatch.setattr(type(c), "_arm_qpos", lambda self: {"left": q, "right": np.zeros(6)})
    monkeypatch.setattr(type(c), "_replan_arm", lambda self, a, p: calls.append(a) or _line_plan(n=50))
    c._maybe_replan({"left": plan, "right": None}, {"left": 150, "right": 0}, {"left": 0, "right": 0})
    assert calls == ["left"]


def test_view_progress_and_unchanged_fields():
    cfg = ResidualConfig()
    c = _ctrl(cfg)
    plan = _line_plan(n=200)
    q = {"left": plan["position"][40].copy(), "right": np.full(6, 0.3)}
    v = c._view({"left": plan, "right": None}, {"left": 50, "right": 0}, q)
    pos = plan["position"]
    assert np.allclose(v["a_nom"][:6], pos[59] - q["left"]) and np.allclose(v["a_nom"][6:], -0.3)
    assert np.allclose(v["future"][0, :6], pos[min(50 + 50 + 9, 199)] - q["left"])
    assert np.allclose(v["active"], [1.0, 0.0]) and np.allclose(v["progress"], [0.25, 0.0])


# ------------------------------------------------------------------------------------------------- state ----
def _fake_tray_probe(named_id="glass"):
    from residual.tray import ITEM_IDS, TrayProbe

    p = object.__new__(TrayProbe)
    p.named = types.SimpleNamespace(item_id=named_id)
    p.other_id = next(i for i in ITEM_IDS if i != named_id)
    p.task = types.SimpleNamespace(carry_arm="right", target_xy=np.array([0.0, -0.02]))
    p.design = {"glass": {"h": 0.2, "base_r": 0.04, "a_tip": 3.3}, "can": {"h": 0.115, "base_r": 0.024, "a_tip": 4.2}}
    return p


def _item(x, tilt):
    T = np.eye(4)
    T[:3, 3] = [x, 0.0, 0.76]
    return {"T": T, "vel": np.arange(6) * 0.1, "tilt": tilt, "margin": np.radians(45.0) - tilt}


def _rob():
    return {"q_left": np.zeros(6), "q_right": np.ones(6), "qd_left": np.zeros(6), "qd_right": np.zeros(6),
            "grip": np.ones(2), "ee_left": np.array([-0.3, 0, 1.0, 1, 0, 0, 0]),
            "ee_right": np.array([0.3, -0.2, 0.9, 1, 0, 0, 0])}


def test_tray_obs_layout():
    from residual.tray import N_ITEM_FEAT, build_obs, obs_dim

    cfg = ResidualConfig(task="tray_slide")
    T_tray = np.eye(4)
    T_tray[:3, 3] = [0.33, -0.02, 0.741]
    m = {"named": _item(0.35, 0.1), "other": _item(0.31, 0.0), "tray_T": T_tray, "tray_vel": np.zeros(6),
         "tray_acc": np.array([-2.0, 0, 0]), "drag": True}
    view = {"a_nom": np.zeros(12), "future": np.zeros((len(cfg.future_ticks), 12)), "active": np.array([0.0, 1.0]),
            "progress": np.array([0.0, 0.5])}
    obs_g = build_obs(cfg, _fake_tray_probe("glass"), m, _rob(), view, 1.5)
    assert obs_g.shape == (obs_dim(cfg),) == (186,) and obs_g.dtype == np.float32 and np.all(np.isfinite(obs_g))
    assert obs_g[-1] == 1.5 and list(obs_g[-3:-1]) == [1.0, 0.0]          # w, then the named one-hot (glass)
    items = obs_g[-3 - 2 * N_ITEM_FEAT:-3].reshape(2, N_ITEM_FEAT)
    assert np.isclose(items[0, 0], 0.02) and np.isclose(items[0, 15], 0.1)   # the named item comes first
    assert np.isclose(items[0, 17], 0.2) and np.isclose(items[1, 17], 0.115)  # heights: glass, then can
    obs_c = build_obs(cfg, _fake_tray_probe("can"), m, _rob(), view, 1.5)
    assert list(obs_c[-3:-1]) == [0.0, 1.0]
    assert np.isclose(obs_c[-3 - 2 * N_ITEM_FEAT + 17], 0.115)               # named first: now the can's height
    # carry arm one-hot right after the hands, then the planner part
    assert list(obs_g[44:46]) == [0.0, 1.0]
    # an infinite margin (no rule for the other item) never reaches the network as inf
    m["other"]["margin"] = float("inf")
    assert np.all(np.isfinite(build_obs(cfg, _fake_tray_probe("glass"), m, _rob(), view, 1.0)))


def test_tray_probe_measure_with_fake_entities(monkeypatch):
    """Margins from the runner's own one_margin (45 deg minus tilt), the tray acceleration by finite difference."""
    from residual.tray import TrayNamed, TrayProbe  # first: sets up the main.item_envs path

    import main.item_envs.runner as runner_mod
    from main.item_envs.rules import Consequence

    class Spec:
        def __init__(self, iid, up=(0.0, 1.0, 0.0)):
            self.item_id, self.up_axis_local = iid, np.array(up)
            self.T = np.eye(4)
            self.v = np.zeros(6)
            self.obb_center_local = np.zeros(3)
            self.half_extents = np.ones(3) * 0.02
            self.category = iid

    def rot_x(a):
        T = np.eye(4)
        T[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        return T

    tray, glass, can = Spec("tray", (0, 0, 1)), Spec("glass"), Spec("can")
    y_up = rot_x(np.pi / 2)   # model y -> world z: upright
    glass.T = y_up.copy()
    can.T = rot_x(np.pi / 2 - np.radians(30.0))    # tilted 30 deg
    monkeypatch.setattr(runner_mod, "_world_T", lambda s: s.T)
    monkeypatch.setattr(runner_mod, "_velocity6", lambda s: s.v)
    rule = lambda iid: types.SimpleNamespace(rule_id=f"upright|{iid}|-", consequence=Consequence.UPRIGHT,
                                             params={"theta_max": np.radians(45.0)})
    runner = types.SimpleNamespace(items=[tray, glass, can], rules=[rule("glass"), rule("can")], _stay_ref={},
                                   env=types.SimpleNamespace(task=types.SimpleNamespace(
                                       standing=[], _time_scale_risky=2.33, _time_scale_safe=0.6,
                                       active_time_scale=2.33)))
    clock = {"n": 0}
    named = TrayNamed(item_id="glass", kind="upright", in_way=True, reason="fast", rule_index=0, item_index=1)
    p = TrayProbe(runner, named, ResidualConfig(task="tray_slide"), clock=lambda: clock["n"])
    m = p.measure()
    assert np.isclose(m["margin"], np.radians(45.0)) and np.isclose(m["other"]["margin"], np.radians(15.0))
    assert m["drag"] and np.isclose(m["tilt"], 0.0, atol=1e-9)
    tray.v = np.array([-0.5, 0, 0, 0, 0, 0])
    clock["n"] = 10
    m = p.measure()
    assert np.allclose(m["tray_acc"], [-0.5 / 0.04, 0, 0])
    assert np.isclose(p.tray_v_max, 0.5) and np.isclose(np.degrees(p.other_max_tilt), 30.0)
    runner.env.task.active_time_scale = 0.6
    assert not p.drag_active()


# --------------------------------------------------------------------------------------- scenes and jobs ----
def test_item_balancer_alternates(monkeypatch):
    import residual.tray as tray
    from residual.episode import Job

    monkeypatch.setattr(tray, "make_named", lambda runner, iid: types.SimpleNamespace(item_id=iid))
    bal = tray.ItemBalancer(ResidualConfig(task="tray_slide"))
    rng = np.random.default_rng(0)
    got = [bal(None, rng, Job()).item_id for _ in range(100)]
    assert got.count("glass") == 50 and all(got[2 * k] != got[2 * k + 1] for k in range(50))
    assert bal(None, rng, Job(item_id="can")).item_id == "can"


def test_tray_eval_scenes_paired():
    from residual.tray import select_eval_scenes

    scans = [{"seed": 9_000_000 + s, "n_candidates": 2, "categories": {"glass": "wine_glass", "can": "can"}}
             for s in range(8)]
    scans[1]["error"] = "RuntimeError: boom"
    sel = select_eval_scenes(scans, 3, ("fast", "medium"))
    assert [c["seed"] for c in sel] == [9_000_000, 9_000_000, 9_000_002, 9_000_002, 9_000_003, 9_000_003]
    assert [c["item_id"] for c in sel] == ["glass", "can"] * 3
    assert [c["behaviour"] for c in sel] == ["fast", "fast", "medium", "medium", "fast", "fast"]
    assert [c["in_way"] for c in sel[:4]] == [True, True, True, False]   # medium tips the glass only


def test_tray_hooks_jobs():
    from residual.episode import Job
    from residual.tasks import task_hooks

    h = task_hooks("tray_slide")
    cfg = ResidualConfig(task="tray_slide", tray_behaviours=("fast", "slow"))
    rng = np.random.default_rng(3)
    drawn = set()
    for _ in range(40):
        j = Job()
        h.prepare_job(cfg, j, rng)
        drawn.add(j.extra["behaviour"])
    assert drawn == {"fast", "slow"}
    j = Job(extra={"behaviour": "medium"})
    h.prepare_job(cfg, j, rng)
    assert j.extra["behaviour"] == "medium"
    assert h.scene_job_extra({"behaviour": "fast"}) == {"behaviour": "fast"}
    t1 = task_hooks("selective_avoid")
    j = Job()
    t1.prepare_job(ResidualConfig(), j, rng)
    assert j.extra == {} and t1.scene_job_extra({"behaviour": "x"}) == {}
    with pytest.raises(ValueError):
        task_hooks("granular_pour")


# ------------------------------------------------------------------------------------- learner summaries ----
def _learner(task):
    from residual.tasks import task_hooks
    from residual.train import Learner

    L = object.__new__(Learner)
    L.cfg = ResidualConfig(task=task)
    L.hooks = task_hooks(task)
    L.model = types.SimpleNamespace(std=lambda: types.SimpleNamespace(mean=lambda: 0.1))
    return L


def _ep(**kw):
    e = {"in_way": True, "item_id": "x", "rule_kind": "avoid", "violated": False, "success": True, "corr_rms": 0.1,
         "replans": 1, "named_held": False, "end": "finished", "grip_events": [], "end_reason": "settled",
         "rew_mean": 0.9, "policy_ticks": 100, "clip_frac": 0.0, "sim_ticks": 150, "wall_episode": 10.0,
         "traj": {"rew": np.ones(100)}}
    e.update(kw)
    return e


def test_t1_summary_keys_unchanged():
    L = _learner("selective_avoid")
    eps = [_ep(), _ep(in_way=False, rule_kind="stay"), _ep(violated=True, end="violation")]
    s = L.eval_summary(eps)
    assert set(s) == T1_EVAL_KEYS
    assert s["in_way/broken"] == 0.5 and s["out_of_way/broken_stay"] == 0.0
    t = L.train_stats(eps)
    assert T1_TRAIN_GROUP_KEYS <= set(t) and not any(k.startswith(("glass/", "can/")) for k in t)
    assert t["in_way/broken"] == 0.5 and "named_held" in t
    assert L.hooks.baseline_keys == ("in_way/broken", "in_way/success", "out_of_way/success", "success")


def test_tray_summaries_split_by_item():
    L = _learner("tray_slide")
    eps = [_ep(item_id="glass", rule_kind="upright", violated=True, end="violation", other_broken=True, resyncs=4),
           _ep(item_id="glass", rule_kind="upright", other_broken=False, resyncs=2),
           _ep(item_id="can", rule_kind="upright", other_broken=True, resyncs=0)]
    s = L.eval_summary(eps)
    assert s["glass/n"] == 2 and s["glass/broken"] == 0.5 and s["can/broken"] == 0.0
    assert s["glass/other_broken"] == 0.5 and s["glass/resyncs"] == 3.0 and s["can/other_broken"] == 1.0
    assert not any(k.startswith(("in_way/", "out_of_way/")) for k in s)
    t = L.train_stats(eps)
    assert t["glass/broken"] == 0.5 and t["can/share"] == pytest.approx(1 / 3)
    assert "broken glass 0.50 can 0.00" in L.hooks.update_line(t)
    assert "broken glass 0.50 can 0.00" in L.hooks.eval_line(s)
