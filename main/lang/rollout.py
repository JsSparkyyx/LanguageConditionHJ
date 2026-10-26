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
- V5 (fusion_impl.md 4.6): ``fusion`` (a FusionStage) and ``keep_neck``. With
  either, ``_encode_obs`` makes one backbone pass that also gives the frame's
  neck dict (``_neck_now``) and still returns the UNFUSED tokens, so the
  StepBuffer row is today's (FUSION_PLAN R3). ``fusion``: the shim fuses the
  neck with each sentence into that sentence's own frame history.
  ``keep_neck`` (version-2 collection): every captured row carries its
  neck_buffer/ row (``_pending["_neck"]``, popped by NeckLockstepWriter).
- V6 (qwen_impl.md 5): ``qwen`` (a HoloBrainQwenEncoder). Frames render at
  308 x 252 (obs.qwen_obs) and encode through HoloBrain-Qwen (297 scene
  tokens, the prefix keys / values ``_kv_now``); the robot state stays GD's.
  The shim runs each sentence against the frame's prefix every call (the
  words read the frame). ``keep_kv`` / ``keep_frames``: every captured row
  carries its kv_buffer/ and frames/ rows (``_pending["_kv"]`` /
  ``["_frames"]``, popped by SidecarLockstepWriter).
"""

from __future__ import annotations

import time
from collections import deque
from functools import partial

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

    With a fusion stage on the controller (``ro.fusion``, V5) the tokens it is
    handed are ignored: the frame's neck (``ro._neck_now``) is fused with the
    named sentence, and with the counterfactual one when set, and each
    sentence keeps its own history (``hist`` / ``frame0``, ``hist_cf`` /
    ``frame0_cf``): a fused frame is valid for its sentence only (R4).
    """

    def __init__(self, ro, policy_enc, k: int = 2):
        self.ro, self.pe, self.k = ro, policy_enc, int(k)
        self.hist = deque(maxlen=self.k + 1)
        self.frame0 = None
        self.hist_cf = deque(maxlen=self.k + 1)     # fusion: the counterfactual sentence's fused frames
        self.frame0_cf = None
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
        return self._stack(self.hist, self.frame0, cur)

    def frames_cf(self, tok, pos):
        """frames() for the counterfactual sentence's own fused history (fusion only; pushed at the same captures
        as the named one, so both hold the same frames, each fused with its sentence)."""
        if self.ro._in_capture:
            if self.frame0_cf is None:
                self.frame0_cf = (tok, pos)
            self.hist_cf.append((tok, pos))
            cur = self.hist_cf[-1]
        else:
            cur = (tok, pos)
        return self._stack(self.hist_cf, self.frame0_cf, cur)

    def _stack(self, hist, frame0, cur):
        prev = hist[0] if len(hist) == self.k + 1 else frame0
        if prev is None:   # nothing captured yet: frame 0 = the fresh frame
            prev = cur
        f0 = frame0 if frame0 is not None else cur
        import torch

        st = torch.stack([cur[0], prev[0], f0[0]], dim=1)          # [1, 3, 1200, 256]
        sp = torch.stack([cur[1], prev[1], f0[1]], dim=1)
        return st, sp

    def __call__(self, state, body_tokens, link_pos, body_mask, joint_index, tokens, pos_mean):
        import torch

        self.n_calls += 1
        if getattr(self.ro, "fusion", None) is not None:
            return self._call_fused(state, body_tokens, link_pos, body_mask, joint_index)
        if getattr(self.ro, "qwen", None) is not None:
            self._qwen_text()
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

    def _qwen_text(self):
        """V6: the episode's sentence (and the counterfactual one) against this frame's prefix keys / values, one
        batch, then the text adapter: text_out / cf_text_out for this call. With qwen_words_prefix = "blank" the
        words read the black-image prefix instead (the words-only control)."""
        import torch

        ro = self.ro
        if ro.sentence is None:
            raise RuntimeError("V6: no sentence; call set_instruction before run()")
        t0 = time.perf_counter()
        blank = getattr(getattr(ro, "cfg", None), "qwen_words_prefix", "frame") == "blank"
        kv = ro.qwen.blank_kv() if blank else ro._kv_now
        if kv is None:
            raise RuntimeError("V6: no encoded frame yet (the prefix keys / values come from _encode_obs)")
        cf = ro.cf_sentence is not None
        words, wmask = ro.qwen.encode_text(kv, [ro.sentence] + ([ro.cf_sentence] if cf else []))
        with torch.no_grad():
            ro.text_out = self.pe.encode_text(words[:1], wmask[:1])
            ro.cf_text_out = self.pe.encode_text(words[1:2], wmask[1:2]) if cf else None
        timing = getattr(ro, "timing", None)
        if timing is not None:
            timing["text"] = timing.get("text", 0.0) + time.perf_counter() - t0

    def _call_fused(self, state, body_tokens, link_pos, body_mask, joint_index):
        """V5: this frame's neck fused with each sentence (one fuse call, the neck twice when a counterfactual is
        set), pushed into that sentence's history during a capture, then the encoder on each sentence's frames
        [t, t-k, 0]. With fused_words the text adapter reads the fused words of frame t, so text_out is recomputed
        every call (``ro.text_out`` / ``ro.cf_text_out`` hold the last one; FUSION_PLAN 3.4)."""
        import torch

        ro = self.ro
        if ro.text_words is None:
            raise RuntimeError("scene_fusion: no sentence before the first frame; call set_instruction before run() "
                               "(frame 0 would be fused without the episode's sentence, FUSION_PLAN C5)")
        cf = ro.cf_text_words is not None
        neck, (words, wmask) = ro._neck_now, ro.text_words
        if cf:
            neck = {key: v.expand(2, *v.shape[1:]) for key, v in neck.items()}
            words = torch.cat([words, ro.cf_text_words[0]])
            wmask = torch.cat([wmask, ro.cf_text_words[1]])
        t0 = time.perf_counter()
        tok, pos, w_fused = ro.fusion.fuse(neck, words, wmask)        # [F, 1200, 256], [F, 1200, 3], [F, 24, 256]
        timing = getattr(ro, "timing", None)
        if timing is not None:
            timing["fuse"] = timing.get("fuse", 0.0) + time.perf_counter() - t0
        st, sp = self.frames(tok[:1], pos[:1])
        st_cf, sp_cf = self.frames_cf(tok[1:2], pos[1:2]) if cf else (None, None)
        tag = hasattr(self.pe, "e_tag")
        kw_a = dict(zip(("ref_center", "ref_valid"), self.centres(ro.named_rule_id))) if tag else {}
        kw_b = dict(zip(("ref_center", "ref_valid"), self.centres(ro.cf_rule_id))) if tag else {}
        with torch.no_grad():
            if bool(getattr(getattr(ro, "cfg", None), "fused_words", False)):
                ro.text_out = self.pe.encode_text(w_fused[:1], ro.text_words[1])
                if cf:
                    ro.cf_text_out = self.pe.encode_text(w_fused[1:2], ro.cf_text_words[1])
            enc = self.pe(state, body_tokens, link_pos, body_mask, joint_index, st, sp, text_out=ro.text_out, **kw_a)
            self.enc_cf = (self.pe(state, body_tokens, link_pos, body_mask, joint_index, st_cf, sp_cf,
                                   text_out=ro.cf_text_out, **kw_b) if cf else None)
        return enc


class LangRolloutController(RolloutController):
    def __init__(self, *args, text_store=None, monitor_only: bool = False, fusion=None, keep_neck: bool = False,
                 qwen=None, keep_kv: bool = False, keep_frames: bool = False, **kwargs):
        cfg0 = args[0] if args else kwargs.get("cfg")
        cfg_qwen = bool(getattr(cfg0, "qwen", False))
        if (qwen is not None) != cfg_qwen:
            raise ValueError(f"encoder={getattr(cfg0, 'encoder', 'holobrain_gd')} in the config but qwen="
                             f"{'an encoder' if qwen is not None else 'None'}: pass a HoloBrainQwenEncoder exactly "
                             "when the config's encoder is holobrain_qwen")
        if qwen is not None and (fusion is not None or keep_neck):
            raise ValueError("V6 (qwen) has no GD neck: fusion / keep_neck are refused")
        if (keep_kv or keep_frames) and qwen is None:
            raise ValueError("keep_kv / keep_frames are the V6 sidecars: they need qwen")
        if getattr(cfg0, "oracle_ref_tag", False) and not monitor_only:
            raise SystemExit("oracle_ref_tag reads the named object's position from the simulator: an offline "
                             "measuring tool (review round 3, E2), never run in closed loop (monitor_only: Q is "
                             "recorded, the filter can never engage)")
        if fusion is not None or getattr(cfg0, "scene_fusion", False):
            # checked before the env is built: a scene_fusion model on unfused tokens (or a stage nothing calls)
            # would run silently on inputs it was not trained on
            from .network import LangPolicyEncoder

            pe0 = kwargs.get("policy_enc", args[5] if len(args) > 5 else None)
            if isinstance(pe0, LangPolicyEncoder) and (fusion is not None) != bool(getattr(cfg0, "scene_fusion", False)):
                raise ValueError(f"scene_fusion={getattr(cfg0, 'scene_fusion', False)} in the config but fusion="
                                 f"{'a FusionStage' if fusion is not None else 'None'}: pass the stage exactly when the "
                                 "model was trained with scene_fusion")
            if fusion is not None and not isinstance(pe0, LangPolicyEncoder):
                raise ValueError("fusion needs a LangPolicyEncoder policy (the LangPolicyShim runs the stage)")
        # monitor_only (round 3, E0a / E3): Q of both sentences every tick, the filter never engages (nominal motion)
        self.monitor_only = bool(monitor_only)
        self.cf_rule_id = None
        t0 = time.perf_counter()
        self._in_capture = False
        self.text_out = self.cf_text_out = None
        self.text_store = text_store
        self.named_rule_id = None
        # V5 (fusion_impl.md 4.6): the frozen fusion stage (the shim fuses the kept neck with the sentence) and
        # version-2 collection (each captured row carries its neck row)
        self.fusion = fusion
        self.keep_neck = bool(keep_neck)
        self._neck_now = None       # the neck dict of the last encoded frame (captures and cache misses)
        self._neck_capture = None   # the neck of this capture's frame, until _collect_tick takes it
        self.text_words = self.cf_text_words = None   # fusion: (words, mask) of each sentence, from the text store
        # V6 (qwen_impl.md 5): the frozen HoloBrain-Qwen front end and the sidecar rows of each capture
        self.qwen = qwen
        self.keep_kv, self.keep_frames = bool(keep_kv), bool(keep_frames)
        self._kv_now = None          # (K, V) of the last encoded frame (captures and cache misses)
        self._kv_capture = self._frames_capture = None   # this capture's rows, until _collect_tick takes them
        self.sentence = self.cf_sentence = None
        super().__init__(*args, **kwargs)
        self.timing = {"build": time.perf_counter() - t0}
        self.trace["status"] = "ok"
        for k in ("h_all", "h_named", "h_other_min", "action_source", "named_dist", "held", "Q_cf", "Q_cf_t"):
            self.trace[k] = []
        env = self.env
        # timing seams (instance attributes only; restored by the env's close)
        if self.qwen is not None:   # the base capture and the cache-miss path both render through it
            from .obs import qwen_obs

            env.get_encoder_obs = _Timer(partial(qwen_obs, env), self.timing, "render")
        else:
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
        """Text store lookup + text adapter, once per episode (no grad).

        With fusion the words and masks are kept for the shim's fuse calls, and the call must come before the
        first captured frame (the frame histories are fused with the sentence, R4). With fused_words the text
        adapter runs in the shim, on each frame's fused words (text_out stays None until the first frame)."""
        import torch

        shim = self.policy_enc
        if not isinstance(shim, LangPolicyShim):
            raise ValueError("set_instruction needs a LangPolicyEncoder policy (LangPolicyShim)")
        if getattr(self, "qwen", None) is not None:
            # V6: the words depend on the frame, so the shim runs the text pass every call; check the token budget now
            for s in (sentence, cf_sentence):
                if s is not None:
                    self.qwen.instruction_ids(s)
            self.text_out = self.cf_text_out = None
            self.sentence, self.cf_sentence = sentence, cf_sentence
            if rule_id is not None:
                self.named_rule_id = rule_id
            self.cf_rule_id = cf_rule_id
            return
        fusion = getattr(self, "fusion", None)
        if fusion is not None and shim.n_pushed:
            raise RuntimeError(f"set_instruction after {shim.n_pushed} captured frame(s): with scene_fusion the frame "
                               "history is fused with the episode's first sentence (FUSION_PLAN R4)")
        per_frame = fusion is not None and bool(getattr(self.cfg, "fused_words", False))
        with torch.no_grad():
            w, m = self.text_store.lookup([sentence])
            self.text_out = None if per_frame else shim.pe.encode_text(w, m)
            w2 = m2 = None
            if cf_sentence is not None:
                w2, m2 = self.text_store.lookup([cf_sentence])
                self.cf_text_out = None if per_frame else shim.pe.encode_text(w2, m2)
        if fusion is not None:
            self.text_words = (w, m)
            self.cf_text_words = (w2, m2) if cf_sentence is not None else None
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
            out = super()._capture_tick()
        finally:
            self._in_capture = False
            self._trace_rules()
            self.timing["capture"] = self.timing.get("capture", 0.0) + time.perf_counter() - t0
            self.timing["n_capture"] = self.timing.get("n_capture", 0) + 1
        if getattr(self, "agent_view", None) is not None:
            self._agent_view_tick()
        return out

    def _agent_view_tick(self) -> None:
        """VLM failure function (agent_view.py): the agent-view frame of this capture's row, at the ticks the writer
        wants. The tick is the item h_fn's row counter (one row per capture, t_in_ep), read after the capture."""
        fn = getattr(self.h_fn, "fn", self.h_fn)   # run() wraps the h_fn in a _Timer
        n = getattr(fn, "_tick", None)
        if n is None or n == getattr(self, "_av_rows", None):   # no new row this capture
            return
        self._av_rows = n
        tick = int(n) - 1
        if self.agent_view.wants(tick):
            from main.envs.record import observer_rgb

            t0 = time.perf_counter()
            self.agent_view.save(tick, observer_rgb(self.env.task))
            self.timing["agent_view"] = self.timing.get("agent_view", 0.0) + time.perf_counter() - t0

    def _encode_obs(self, obs):
        """The base encoding (tokens, pos_mean, state, body), always UNFUSED: the StepBuffer row and the V0 readers
        keep today's tokens (FUSION_PLAN R3). With fusion or keep_neck one backbone pass also gives the frame's neck
        dict (fusion.encode_obs_with_neck: tokens and positions bit-identical to the base path), kept as _neck_now
        for the shim and, inside a capture only, as the capture's neck for _collect_tick (the base class's
        cache-miss encoding runs outside a capture and never reaches a buffer row)."""
        if getattr(self, "qwen", None) is not None:
            return self._encode_qwen(obs)
        if self.fusion is None and not self.keep_neck:
            return super()._encode_obs(obs)
        import torch

        from .fusion import encode_obs_with_neck

        with torch.no_grad():
            (tokens, pos_mean), neck = encode_obs_with_neck(self.encoder, obs)
            state = self.encoder.encode_joint_angles(
                obs["joint_state"], self.kin, embodiedment_mat=obs["T_base2ego"],
            ).squeeze(2)                          # [B, N, D]
            body = self.extractor.extract()
        self._neck_now = neck
        if self._in_capture:
            self._neck_capture = neck
        return tokens, pos_mean, state, body

    def _encode_qwen(self, obs):
        """V6: HoloBrain-Qwen tokens [1, 3, 99, 256] / positions, GD's robot-state encoder and the body features.
        Keeps the frame's prefix keys / values (_kv_now) and, inside a capture, the sidecar rows' sources."""
        import torch

        from .obs import encode_qwen_obs

        if not obs.get("qwen"):
            raise RuntimeError("V6: a GD observation bundle reached the Qwen encoder (get_encoder_obs not swapped?)")
        with torch.no_grad():
            tokens, pos_mean, kv = encode_qwen_obs(self.qwen, obs)
            state = self.encoder.encode_joint_angles(
                obs["joint_state"], self.kin, embodiedment_mat=obs["T_base2ego"],
            ).squeeze(2)                          # [B, N, D]
            body = self.extractor.extract()
        self._kv_now = kv
        if self._in_capture:
            if self.keep_kv:
                self._kv_capture = kv
            if self.keep_frames:
                self._frames_capture = (obs["native"], obs["T_base2ego"])
        return tokens, pos_mean, state, body

    def _collect_tick(self, obs, h, diag, encoded):
        """keep_neck (version 2): the capture's neck row rides in the entry the base class stashes as _pending and
        writes one capture later (the last one in _flush_pending); NeckLockstepWriter pops it inside the same
        append. The row is made before the base call, so a failed conversion (fp16 overflow) leaves the previous
        entry pending, as any tick error does (the episode is then rolled back at its end). V6 keep_kv /
        keep_frames: the same for the kv_buffer/ and frames/ rows (SidecarLockstepWriter pops them)."""
        rows = {}
        if getattr(self, "qwen", None) is not None:
            import torch

            tok = encoded[0]
            if not bool(torch.isfinite(tok).all()) or float(tok.abs().max()) > 6.0e4:
                raise RuntimeError("V6: scene tokens outside the fp16 range (the StepBuffer stores them as fp16)")
            if self.keep_kv:
                kv, self._kv_capture = self._kv_capture, None
                if kv is None:
                    raise RuntimeError("keep_kv: this capture's encoding left no prefix keys / values")
                from .kv_buffer import kv_to_row

                rows["_kv"] = kv_to_row(*kv)
            if self.keep_frames:
                fr, self._frames_capture = self._frames_capture, None
                if fr is None:
                    raise RuntimeError("keep_frames: this capture's encoding left no native frames")
                from .frames_buffer import frames_to_row

                rows["_frames"] = frames_to_row(*fr)
        row = None
        if self.keep_neck:
            neck, self._neck_capture = self._neck_capture, None
            if neck is None:
                raise RuntimeError("keep_neck: this capture's encoding left no neck (only _encode_obs inside a "
                                   "capture provides one)")
            from .fusion import neck_to_row

            row = neck_to_row(neck)
        super()._collect_tick(obs, h, diag, encoded)
        if row is not None:
            self._pending["_neck"] = row   # after super(): _pending is this capture's entry
        self._pending.update(rows)

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
