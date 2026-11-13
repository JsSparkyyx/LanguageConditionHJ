"""Per-task hooks: everything in the episode, the worker and the learner that differs between the tasks.

``task_hooks(cfg.task)`` returns one object with: building a scene, naming a rule (the chooser), the measuring
probe and the state vector, the evaluation scenes, and the groups that the training and evaluation numbers are split
by. ``T1Hooks`` calls the code that existed before (scene.py, state.py, worker.SideBalancer, train.select_eval_scenes)
unchanged, so T1 runs exactly as before; ``TrayHooks`` is the sliding tray (tray.py, PLAN section 8).
"""

from __future__ import annotations

import numpy as np


def _rate(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


class T1Hooks:
    """T1 ``selective_avoid``: an extra object in or out of the planner's way, its avoid or stay rule."""

    name = "selective_avoid"
    groups = (("in_way", lambda e: bool(e["in_way"])), ("out_of_way", lambda e: not e["in_way"]))
    rule_kinds = ("avoid", "stay")
    baseline_keys = ("in_way/broken", "in_way/success", "out_of_way/success", "success")
    video_prefix = {"in_way": "in", "out_of_way": "out"}

    # ------------------------------------------------------------------ episodes ----
    def prepare_job(self, cfg, job, rng) -> None:
        return None

    def build_runner(self, cfg, seed: int, out_root: str, job=None):
        from .scene import build_runner

        return build_runner(cfg, seed, out_root)

    def make_chooser(self, cfg):
        from .worker import SideBalancer

        return SideBalancer(cfg)

    def make_probe(self, runner, named, cfg, clock=None):
        from .state import NamedProbe

        return NamedProbe(runner, named, cfg)

    def build_obs(self, cfg, probe, m, rob, view, w):
        from .state import build_obs

        return build_obs(cfg, probe, m, rob, view, w)

    def obs_dim(self, cfg) -> int:
        from .state import obs_dim

        return obs_dim(cfg)

    def margin_unit(self, named) -> float:
        return 0.05 if named.kind == "avoid" else 0.03

    def describe(self, named) -> list[str]:
        return [f"don't {'touch' if named.kind == 'avoid' else 'move'} the {named.category}",
                f"({'in the way' if named.in_way else 'out of the way'}: {named.reason})"]

    def margin_text(self, margin: float) -> str:
        return f"margin {margin * 100:+5.1f} cm"

    def episode_extra(self, ep) -> dict:
        return {}

    # ---------------------------------------------------------------- evaluation ----
    def scan_record(self, runner) -> dict:
        from .scene import scan_record

        return scan_record(runner)

    def eval_need(self, cfg) -> int:
        return int(cfg.eval_in_way + cfg.eval_out_way)

    def select_eval_scenes(self, scans, cfg) -> list[dict]:
        from .train import select_eval_scenes

        return select_eval_scenes(scans, cfg.eval_in_way, cfg.eval_out_way)

    def eval_enough(self, chosen, cfg) -> tuple[bool, str]:
        n_in = sum(c["in_way"] for c in chosen)
        n_out = len(chosen) - n_in
        return (n_in >= cfg.eval_in_way and n_out >= cfg.eval_out_way,
                f"{n_in} in the way, {n_out} out of the way")

    def scene_group(self, sc: dict) -> str:
        return "in_way" if sc["in_way"] else "out_of_way"

    def scene_job_extra(self, sc: dict) -> dict:
        return {}

    def group_extra(self, sub: list[dict], eval_round: bool) -> dict:
        if eval_round:
            return {"named_held": _rate(e["named_held"] for e in sub)}
        return {}

    def update_line(self, t: dict) -> str:
        return (f"in-way broken {t['in_way/broken']:.2f} success {t['success']:.2f} "
                f"corr in {t['in_way/correction_rms']:.3f} out {t['out_of_way/correction_rms']:.3f}")

    def eval_line(self, s: dict) -> str:
        return (f"in-way broken {s['in_way/broken']:.2f} success {s['in_way/success']:.2f} | out-of-way corr "
                f"{s['out_of_way/correction_rms']:.3f} success {s['out_of_way/success']:.2f}")


class TrayHooks(T1Hooks):
    """T8 ``tray_slide``: the glass or the can named (upright), the careless fast drag as the nominal."""

    name = "tray_slide"
    groups = (("glass", lambda e: e["item_id"] == "glass"), ("can", lambda e: e["item_id"] == "can"))
    rule_kinds = ()
    baseline_keys = ("glass/broken", "glass/success", "can/broken", "can/success", "success")
    video_prefix = {"glass": "glass", "can": "can"}

    def prepare_job(self, cfg, job, rng) -> None:
        if not job.extra.get("behaviour"):
            bs = list(cfg.tray_behaviours)
            job.extra["behaviour"] = bs[0] if len(bs) == 1 else bs[int(rng.integers(len(bs)))]

    def build_runner(self, cfg, seed: int, out_root: str, job=None):
        from .scene import build_runner

        b = (job.extra.get("behaviour") if job is not None else None) or cfg.tray_behaviours[0]
        return build_runner(cfg, seed, out_root, risk={"behaviour": str(b)})

    def make_chooser(self, cfg):
        from .tray import ItemBalancer

        return ItemBalancer(cfg)

    def make_probe(self, runner, named, cfg, clock=None):
        from .tray import TrayProbe

        return TrayProbe(runner, named, cfg, clock=clock)

    def build_obs(self, cfg, probe, m, rob, view, w):
        from .tray import build_obs

        return build_obs(cfg, probe, m, rob, view, w)

    def obs_dim(self, cfg) -> int:
        from .tray import obs_dim

        return obs_dim(cfg)

    def margin_unit(self, named) -> float:
        return float(np.radians(45.0))

    def describe(self, named) -> list[str]:
        return [f"keep the {'wine glass' if named.item_id == 'glass' else 'can'} standing",
                f"(drag {named.behaviour}: {'at risk' if named.in_way else 'not at risk'})"]

    def margin_text(self, margin: float) -> str:
        return f"margin {np.degrees(margin):+5.1f} deg"

    def episode_extra(self, ep) -> dict:
        p = ep.probe
        task = ep.task
        out = {"behaviour": ep.named.behaviour, "other_item": p.other_id,
               "other_broken": bool(p.other_min_margin < 0.0), "other_min_margin": float(p.other_min_margin),
               "named_max_tilt_deg": float(np.degrees(p.named_max_tilt)),
               "other_max_tilt_deg": float(np.degrees(p.other_max_tilt)),
               "tray_v_max": float(p.tray_v_max), "tray_acc_max": float(p.tray_acc_max),
               "referent": getattr(task, "referent_id", None), "carry_arm": getattr(task, "carry_arm", None)}
        st = dict(getattr(ep.ctrl, "stats", {}) or {})
        out["resyncs"] = int(st.get("resyncs", 0))
        out["resync_rows"] = int(st.get("resync_rows", 0))
        info = getattr(task, "episode_info", None) or {}
        out["drag_profile"] = info.get("drag_profile")
        out["drag_fallback"] = bool(info.get("drag_fallback", False))
        try:
            from main.item_envs.runner import _item_position

            xy = _item_position(p.tray)[:2]
            out["pad_dist"] = float(np.linalg.norm(xy - np.asarray(task.target_xy, dtype=np.float64)))
        except Exception:
            out["pad_dist"] = None
        return out

    def scan_record(self, runner) -> dict:
        from .tray import scan_record

        return scan_record(runner)

    def eval_need(self, cfg) -> int:
        return int(cfg.eval_pairs)

    def select_eval_scenes(self, scans, cfg) -> list[dict]:
        from .tray import select_eval_scenes

        return select_eval_scenes(scans, cfg.eval_pairs, cfg.tray_behaviours)

    def eval_enough(self, chosen, cfg) -> tuple[bool, str]:
        n = len({c["seed"] for c in chosen})
        return n >= cfg.eval_pairs, f"{n} paired scenes (glass and can named)"

    def scene_group(self, sc: dict) -> str:
        return sc["item_id"]

    def scene_job_extra(self, sc: dict) -> dict:
        return {"behaviour": sc.get("behaviour")}

    def group_extra(self, sub: list[dict], eval_round: bool) -> dict:
        return {"other_broken": _rate(e.get("other_broken", False) for e in sub),
                "resyncs": _rate(e.get("resyncs", 0) for e in sub),
                "max_tilt_deg": _rate(e.get("named_max_tilt_deg", 0.0) for e in sub),
                "tray_v_max": _rate(e.get("tray_v_max", 0.0) for e in sub)}

    def update_line(self, t: dict) -> str:
        return (f"broken glass {t['glass/broken']:.2f} can {t['can/broken']:.2f} success {t['success']:.2f} "
                f"corr glass {t['glass/correction_rms']:.3f} can {t['can/correction_rms']:.3f}")

    def eval_line(self, s: dict) -> str:
        return (f"broken glass {s['glass/broken']:.2f} can {s['can/broken']:.2f} | success glass "
                f"{s['glass/success']:.2f} can {s['can/success']:.2f} | corr glass {s['glass/correction_rms']:.3f} "
                f"can {s['can/correction_rms']:.3f}")


_HOOKS = {"selective_avoid": T1Hooks, "tray_slide": TrayHooks}


def task_hooks(task: str):
    if task not in _HOOKS:
        raise ValueError(f"no residual hooks for task {task!r} (known: {sorted(_HOOKS)})")
    return _HOOKS[task]()
