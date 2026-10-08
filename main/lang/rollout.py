"""LangRolloutController (VANILLA_PLAN 7.5): the CrossSafe RolloutController
with small injected pieces instead of copied methods.

- ``run()``: captures s_0 before the first physics step outside collect mode
  (collect mode already does), records ``trace["status"]`` in {ok, timeout,
  exception} (+ the exception text) by wrapping ``env.task.play_once`` (the
  base class catches these and only prints them), and per-tick timing:
  rendering (``env.get_encoder_obs``), encoding (``_encode_obs``), h
  (``h_fn``), physics (``scene.step``).
- ``LangPolicyShim`` (``self.policy_enc``): CrossSafe's 7-argument encoder
  call; keeps the frame history (pushed only during a capture) and assembles
  frames [t, t-k, 0]; runs the LangPolicyEncoder with the episode's text.
- ``set_instruction(sentence, cf_sentence=None)``: text store lookup + text
  adapter once per episode (no grad).
- ``_tick_hook``: with a counterfactual sentence, Q_cf = qmin(enc_cf, a_nom).
- per-tick trace: h_all, h_named, h_other_min, action_source, the named
  item's robot_item_dist, held (from the capture's diag).
- ``wire_item_eval(ro, rule_id)``: the h_fn for evaluation, no buffer.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import ensure_item_path

ensure_item_path()

from main.train.rollout import RolloutController, RolloutTimeout  # noqa: E402

__all__ = ["LangRolloutController", "LangPolicyShim", "wire_item_eval", "single_rule"]


def single_rule(rule) -> bool:
    """A rule that can be named alone, as language.episode_record decides: not a set rule (several items, or a
    zone inside / outside rule over several regions) and not ambiguous. A bead rule over two containers (T2 /
    T6's spill rule, bowl_src_interior + target_interior) is single (review M2)."""
    cons = rule.consequence.value if hasattr(rule.consequence, "value") else str(rule.consequence)
    is_set = len(rule.item_ids) > 1 or (cons in ("inside", "outside") and len(rule.region_ids) > 1)
    return not is_set and not getattr(rule, "ambiguous", False)


class _Timer:
    """Accumulating wall-clock timer around a callable."""

    def __init__(self, fn, acc: dict, key: str):
        self.fn, self.acc, self.key = fn, acc, key

    def __call__(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return self.fn(*a, **k)
        finally:
            self.acc[self.key] = self.acc.get(self.key, 0.0) + time.perf_counter() - t0


class LangPolicyShim:
    """CrossSafe's 7-argument policy_enc call -> LangEncoded with frames [t, t-k, 0].

    A frame is pushed into the history only during a capture (``ro._in_capture``):
    the first pushed frame is frame 0, a deque holds the last k + 1 captured
    frames, and t - k is frame 0 until k + 1 frames exist (the sampler's
    max(t - k, 0)). On the base class's cache-miss path the fresh frame is
    used as t without being pushed.
    """

    def __init__(self, ro, policy_enc, k: int = 2):
        self.ro, self.pe, self.k = ro, policy_enc, int(k)
        self.hist = deque(maxlen=self.k + 1)
        self.frame0 = None
        self.cent_hist = deque(maxlen=self.k + 1)   # V3 monitor: {rule_id: centre or None} per captured frame
        self.cent0 = None
        self.enc_cf = None
        self.n_pushed = 0
        self.n_calls = 0

    @staticmethod
    def _flat(tokens, pos):
        # [1, n_cam, 400, 256] -> [1, 1200, 256] (cameras flattened within the frame)
        return tokens.flatten(1, 2), pos.flatten(1, 2)

    def _centres_now(self) -> dict:
        """V3 monitor: world centre of each named rule's object at this capture (the tick's item poses, from the
        diag the capture computed before encoding, + the item spec's OBB centre offset)."""
        from .data import _quat_rotate

        ro = self.ro
        runner = ro.env._item_runner
        pose = (ro._last_diag or {}).get("item_pose")
        items = [sp.item_id for sp in runner.items]
        out = {}
        for rid in (ro.named_rule_id, ro.cf_rule_id):
            rule = next((r for r in runner.rules if r.rule_id == rid), None)
            out[rid] = None
            if rule is None or pose is None or len(rule.item_ids) != 1 or rule.item_ids[0] not in items:
                continue
            if rule.consequence.value not in ("avoid", "upright", "stay"):
                continue
            k = items.index(rule.item_ids[0])
            pk = np.asarray(pose[k], np.float64)
            out[rid] = pk[:3] + _quat_rotate(pk[3:7], np.asarray(runner.items[k].obb_center_local, np.float64))
        return out

    def centres(self, rid):
        """[1, 3, 3] centres and [1, 3] validity of rule rid's object in frames [t, t-k, 0] (V3 monitor)."""
        import torch

        cur = self.cent_hist[-1] if self.cent_hist else {}
        prev = self.cent_hist[0] if len(self.cent_hist) == self.k + 1 else (self.cent0 or cur)
        f0 = self.cent0 or cur
        c, v = np.zeros((1, 3, 3), np.float32), np.zeros((1, 3), bool)
        for f, fr in enumerate((cur, prev, f0)):
            x = fr.get(rid) if fr else None
            if x is not None:
                c[0, f], v[0, f] = x, True
        return torch.from_numpy(c).cuda(), torch.from_numpy(v).cuda()

    def frames(self, tok, pos):
        if self.ro._in_capture:
            if self.frame0 is None:
                self.frame0 = (tok, pos)
            self.hist.append((tok, pos))
            self.n_pushed += 1
            cur = self.hist[-1]
            if hasattr(self.pe, "e_tag"):
                cn = self._centres_now()
                if self.cent0 is None:
                    self.cent0 = cn
                self.cent_hist.append(cn)
        else:
            cur = (tok, pos)
        prev = self.hist[0] if len(self.hist) == self.k + 1 else self.frame0
        if prev is None:   # nothing captured yet: frame 0 = the fresh frame
            prev = cur
        f0 = self.frame0 if self.frame0 is not None else cur
        import torch

        st = torch.stack([cur[0], prev[0], f0[0]], dim=1)          # [1, 3, 1200, 256]
        sp = torch.stack([cur[1], prev[1], f0[1]], dim=1)
        return st, sp

    def __call__(self, state, body_tokens, link_pos, body_mask, joint_index, tokens, pos_mean):
        import torch

        self.n_calls += 1
        tok, pos = self._flat(tokens, pos_mean)
        st, sp = self.frames(tok, pos)
        ro = self.ro
        tag = hasattr(self.pe, "e_tag")   # V3 (monitor only): each sentence's tag follows its own named object
        kw_a = dict(zip(("ref_center", "ref_valid"), self.centres(ro.named_rule_id))) if tag else {}
        kw_b = dict(zip(("ref_center", "ref_valid"), self.centres(ro.cf_rule_id))) if tag else {}
        with torch.no_grad():
            enc = self.pe(state, body_tokens, link_pos, body_mask, joint_index, st, sp, text_out=ro.text_out, **kw_a)
            self.enc_cf = (self.pe(state, body_tokens, link_pos, body_mask, joint_index, st, sp,
                                   text_out=ro.cf_text_out, **kw_b) if ro.cf_text_out is not None else None)
        return enc


class LangRolloutController(RolloutController):
    def __init__(self, *args, text_store=None, monitor_only: bool = False, **kwargs):
        cfg0 = args[0] if args else kwargs.get("cfg")
        if getattr(cfg0, "oracle_ref_tag", False) and not monitor_only:
            raise SystemExit("oracle_ref_tag reads the named object's position from the simulator: an offline "
                             "measuring tool (review round 3, E2), never run in closed loop (monitor_only: Q is "
                             "recorded, the filter can never engage)")
        # monitor_only (round 3, E0a / E3): Q of both sentences every tick, the filter never engages (nominal motion)
        self.monitor_only = bool(monitor_only)
        self.cf_rule_id = None
        t0 = time.perf_counter()
        self._in_capture = False
        self.text_out = self.cf_text_out = None
        self.text_store = text_store
        self.named_rule_id = None
        super().__init__(*args, **kwargs)
        self.timing = {"build": time.perf_counter() - t0}
        self.trace["status"] = "ok"
        for k in ("h_all", "h_named", "h_other_min", "action_source", "named_dist", "held", "Q_cf", "Q_cf_t"):
            self.trace[k] = []
        env = self.env
        # timing seams (instance attributes only; restored by the env's close)
        env.get_encoder_obs = _Timer(env.get_encoder_obs, self.timing, "render")
        self._encode_obs = _Timer(self._encode_obs, self.timing, "encode")
        if self.policy_enc is not None and not isinstance(self.policy_enc, LangPolicyShim):
            from .network import LangPolicyEncoder

            if isinstance(self.policy_enc, LangPolicyEncoder):
                self.policy_enc = LangPolicyShim(self, self.policy_enc, int(getattr(self.cfg, "k_prev", 2)))
        self._h_unit_margins()

    def _h_unit_margins(self) -> None:
        """The base class scales filter_margin / filter_release_margin by h_scale (CrossSafe's metres -> h x 20
        units). This critic's Q is in h units, as are LangConfig's margins: undo the scale (review O5; at the
        default 0 it made no difference, at 0.05 the filter compared Q with 1.0 and never released)."""
        flt = getattr(self, "flt", None)
        if flt is not None:
            flt.margin = float(getattr(self.cfg, "filter_margin", 0.0))
            flt.release_margin = float(getattr(self.cfg, "filter_release_margin", 0.0))
            if getattr(self, "monitor_only", False):   # no Q engages: the motion stays nominal
                flt.margin = flt.release_margin = -1e9

    # ------------------------------------------------------------------ text --
    def set_instruction(self, sentence: str, cf_sentence: str | None = None, rule_id: str | None = None,
                        cf_rule_id: str | None = None) -> None:
        """Text store lookup + text adapter, once per episode (no grad)."""
        import torch

        shim = self.policy_enc
        if not isinstance(shim, LangPolicyShim):
            raise ValueError("set_instruction needs a LangPolicyEncoder policy (LangPolicyShim)")
        with torch.no_grad():
            w, m = self.text_store.lookup([sentence])
            self.text_out = shim.pe.encode_text(w, m)
            if cf_sentence is not None:
                w2, m2 = self.text_store.lookup([cf_sentence])
                self.cf_text_out = shim.pe.encode_text(w2, m2)
        self.sentence, self.cf_sentence = sentence, cf_sentence
        if rule_id is not None:
            self.named_rule_id = rule_id
        self.cf_rule_id = cf_rule_id   # the counterfactual sentence's rule (the V3 tag follows its object)

    # --------------------------------------------------------------- capture --
    def _capture_tick(self):
        t0 = time.perf_counter()
        writer = getattr(self, "writer", None)   # main.lang.locks.WriterLock of the data root (collection)
        if writer is not None:
            # every capture, in memory (the lock files every 2 s); collect.guard_writes also checks right before
            # each row write, so a stall anywhere inside the capture is caught too (review round 3, C1)
            writer.check_fast()
        self._in_capture = True
        try:
            return super()._capture_tick()
        finally:
            self._in_capture = False
            self._trace_rules()
            self.timing["capture"] = self.timing.get("capture", 0.0) + time.perf_counter() - t0
            self.timing["n_capture"] = self.timing.get("n_capture", 0) + 1

    def _trace_rules(self):
        diag = self._last_diag
        runner = getattr(self.env, "_item_runner", None)
        if diag is None or runner is None or "margins" not in diag:
            return
        from main.item_envs.margins import rule_h

        rules = list(runner.rules)
        m = np.asarray(diag["margins"], dtype=np.float64).reshape(-1)
        counts = diag.get("violation_counts")
        counts = [None] * len(rules) if counts is None else list(np.asarray(counts).reshape(-1))
        h_all = [float(rule_h(r, v, c)) for r, v, c in zip(rules, m, counts)]
        tr = self.trace
        tr["h_all"].append(h_all)
        nid = self.named_rule_id
        ids = [r.rule_id for r in rules]
        j = ids.index(nid) if nid in ids else None
        tr["h_named"].append(h_all[j] if j is not None else float("nan"))
        single = [k for k, r in enumerate(rules) if single_rule(r) and k != j]
        tr["h_other_min"].append(min((h_all[k] for k in single), default=float("nan")))
        tr["action_source"].append(int(self._tick_source))
        dist = float("nan")
        if j is not None and rules[j].consequence.value in ("avoid", "upright", "stay") and len(rules[j].item_ids) == 1:
            items = [s.item_id for s in runner.items]
            d = np.asarray(diag.get("robot_item_dist", []), dtype=np.float64).reshape(-1)
            k = items.index(rules[j].item_ids[0]) if rules[j].item_ids[0] in items else None
            if k is not None and k < d.size:
                dist = float(d[k])
        tr["named_dist"].append(dist)
        held = diag.get("held")
        tr["held"].append([int(x) for x in np.asarray(held).reshape(-1)] if held is not None else [-1, -1])

    # ------------------------------------------------------------------ hook --
    def _tick_hook(self, ctx) -> bool:
        if self.cf_text_out is not None and self.flt is not None and self.filter_active:
            import torch

            body, enc = self._policy_inputs_cached()
            shim = self.policy_enc
            if shim.enc_cf is not None:
                a = self._anom(ctx)
                with torch.no_grad():
                    qcf = self.flt.critics.qmin(shim.enc_cf, a, body["Jlin"][None].cuda(), body["Jang"][None].cuda(),
                                                self.flt.joint_index)
                self.trace["Q_cf"].append(float(qcf))
                self.trace["Q_cf_t"].append(self._n_physics / self.env.PHYSICS_FREQ)
        return super()._tick_hook(ctx)

    def run(self, max_steps: int | None = None):
        env = self.env
        orig_play = env.task.play_once
        trace = self.trace

        def play_wrapped(*a, **k):
            trace["play_started"] = True
            try:
                return orig_play(*a, **k)
            except RolloutTimeout:
                trace["status"] = "timeout"
                raise
            except Exception as exc:
                trace["status"] = "exception"
                trace["exception"] = f"{type(exc).__name__}: {exc}"[:800]
                raise

        env.task.play_once = play_wrapped
        # physics timing: run() hooks scene.step -> self._orig_step, so wrap first
        env.task.scene.step = _Timer(env.task.scene.step, self.timing, "physics")
        if self.h_fn is not None:
            self.h_fn = _Timer(self.h_fn, self.timing, "h")
        if self.mode != "collect":
            # frame 0 must be s_0 at deployment too (training uses the s_0 row)
            self._capture_tick()
        t0 = time.perf_counter()
        out = super().run(max_steps)
        self.timing["run"] = time.perf_counter() - t0
        out["timing"] = dict(self.timing)
        out["n_ticks"] = len(out.get("t", []))
        if not trace.get("play_started") and trace.get("status") == "ok":
            # the base run catches an error in the first capture (outside play_once, so play_wrapped never ran)
            # and returns: whether or not that capture had appended its row, the episode never ran
            # (review round 2: zero ticks; round 3, C7: one row appended before the error)
            trace["status"] = out["status"] = "exception"
            trace.setdefault("exception", f"play_once never started ({out['n_ticks']} tick(s) captured; "
                                          "an error in the first capture, see the log)")
            out.setdefault("exception", trace["exception"])
        return out


def wire_item_eval(ro, rule_id: str | None) -> None:
    """Evaluation wiring (no buffer): the h_fn tracks ``rule_id``, without the
    per-tick segmentation masks (evaluation never reads them), and the T1
    risky-skill labels reach the controller as in wire_item_episode."""
    from main.item_envs.margin_hook import ItemMarginHFn

    runner = getattr(ro.env, "_item_runner", None)
    if runner is None:
        raise ValueError("wire_item_eval: env was not built by build_item_training_env")
    runner.active_rule_id = rule_id
    h_scale = float(getattr(ro.cfg, "h_scale", 1.0))
    ro.h_fn = ItemMarginHFn(rule_id, 1.0 / h_scale, ctx=runner, row_sink=None, with_masks=False,
                            episode_id=int(getattr(ro, "episode_id", 0)))
    labels = tuple(runner.task_cfg.get("risky_skill_labels", ()) or ())
    if labels and hasattr(ro.ctrl, "risky_skill_labels"):
        ro.ctrl.risky_skill_labels = labels
