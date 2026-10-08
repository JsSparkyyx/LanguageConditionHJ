"""LangPanelVideoWriter (VANILLA_PLAN 8.4): observer frame + a panel in h units.

Keeps a reference to the controller and draws from its current state (the
base class's centimetre ``h`` argument is ignored):

- filter episodes: the instruction (2 lines max) and the named rule id;
  h_named, Q and Q_cf (when present); FILTER / NOMINAL, |a| ratio, intervened
  ticks, block counter; the most at-risk UNNAMED rule and its h; a scrolling
  plot of h_named, Q and the unnamed minimum; a red border while h_named < 0;
- nominal episodes: the minimum h over all rules and the rule attaining it.

File name ``<task>[_<host>]_<embodiment>_s<seed>_<kind>[_<axis>]_<outcome>.mp4``:
kind nominal / harm / benign, outcome ok / harmed / fail, decided at close.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np

from main.train.eval_utils import PanelVideoWriter

W_PANEL, H_PANEL = 320, 480


def video_name(task, host, emb, seed, kind, axis=None) -> str:
    """The stem (outcome appended at close)."""
    parts = [task] + ([host] if host else []) + [emb, f"s{seed}", kind] + ([axis] if axis else [])
    return "_".join(parts)


def _rule_hs(ro) -> tuple[list, list]:
    from main.item_envs.margins import rule_h

    runner = ro.env._item_runner
    diag = ro._last_diag or {}
    m = np.asarray(diag.get("margins", []), dtype=np.float64).reshape(-1)
    rules = list(runner.rules)
    if m.size != len(rules):
        return [r.rule_id for r in rules], [float("nan")] * len(rules)
    counts = diag.get("violation_counts")
    counts = [None] * len(rules) if counts is None else list(np.asarray(counts).reshape(-1))
    return [r.rule_id for r in rules], [float(rule_h(r, v, c)) for r, v, c in zip(rules, m, counts)]


class LangPanelVideoWriter(PanelVideoWriter):
    def __init__(self, stem_path, ro, named_rule: dict | None = None, sentence: str | None = None,
                 fps: float = 25.0, cf_rule: dict | None = None, cf_sentence: str | None = None):
        stem_path = Path(stem_path)
        self.final_stem = stem_path
        # unique per process: a stale writer and the new owner record the same seed (review round 3, C8)
        self.tmp = stem_path.with_name(f"{stem_path.name}.{os.getpid()}.partial.mp4")
        super().__init__(self.tmp, fps=fps, h_scale=1.0)
        self.ro = ro
        self.named = named_rule
        self.named_id = named_rule["rule_id"] if named_rule else None
        self.sentence = sentence
        self.hist_named, self.hist_q, self.hist_other = [], [], []
        # grounding panel (review round 3, E0a): a second sentence B on the same frames, Q(B) and h(B) plotted too
        self.cf_id = cf_rule["rule_id"] if cf_rule else None
        self.cf_sentence = cf_sentence
        self.hist_hcf, self.hist_qcf = [], []
        self.harmed = False
        self.path = None
        runner = ro.env._item_runner
        from .rollout import single_rule

        self.single = {r.rule_id for r in runner.rules if single_rule(r)}

    def _panel_lang(self, t, step, control, ratio, extras):
        from PIL import Image, ImageDraw

        ids, hs = _rule_hs(self.ro)
        trace = self.ro.trace
        img = Image.new("RGB", (W_PANEL, H_PANEL), "white")
        d = ImageDraw.Draw(img)
        red, grey, black = (200, 60, 40), (90, 90, 90), (0, 0, 0)
        y = 10
        d.text((10, y), f"t = {t:5.2f} s   tick {step}", fill=black)
        y += 20
        if self.named_id is not None and self.cf_id is not None:
            return self._panel_pair(img, d, y, ids, hs, trace, control, extras)
        if self.named_id is not None:
            j = ids.index(self.named_id) if self.named_id in ids else None
            h_named = hs[j] if j is not None else float("nan")
            q = trace["V"][-1] if trace.get("V") else float("nan")
            qcf = trace["Q_cf"][-1] if trace.get("Q_cf") else None
            for ln in textwrap.wrap(f'"{self.sentence}"', 44)[:2]:
                d.text((10, y), ln, fill=(20, 60, 160))
                y += 16
            d.text((10, y), f"rule {self.named_id}"[:46], fill=grey)
            y += 20
            d.text((10, y), f"h_named = {h_named:+.3f}    Q = {q:+.3f}", fill=red if h_named < 0 else black)
            y += 18
            d.text((10, y), f"Q_cf = {qcf:+.3f}" if qcf is not None else "Q_cf =   -", fill=grey)
            y += 18
            others = [(hv, rid) for rid, hv in zip(ids, hs) if rid != self.named_id and rid in self.single and hv == hv]
            o_h, o_id = min(others) if others else (float("nan"), "-")
            self.hist_named.append(h_named)
            self.hist_q.append(q)
            self.hist_other.append(o_h)
            harm_now = h_named < 0
        else:
            valid = [(hv, rid) for rid, hv in zip(ids, hs) if hv == hv]
            o_h, o_id = min(valid) if valid else (float("nan"), "-")
            d.text((10, y), "NOMINAL episode (no sentence)", fill=grey)
            y += 20
            d.text((10, y), f"min h = {o_h:+.3f}", fill=red if o_h < 0 else black)
            y += 18
            self.hist_named.append(float("nan"))
            self.hist_q.append(float("nan"))
            self.hist_other.append(o_h)
            harm_now = o_h < 0
        self.harmed |= bool(harm_now)
        d.text((10, y), f"{'most at-risk unnamed' if self.named_id else 'argmin'}: {o_id}"[:46], fill=grey)
        y += 16
        d.text((10, y), f"   h = {o_h:+.3f}", fill=red if o_h < 0 else grey)
        y += 20
        d.text((10, y), f"{control}   |a| = {ratio:.2f} x max", fill=red if control == "FILTER" else grey)
        y += 18
        iv = (extras or {}).get("intervened") or (0, 0.0)
        d.text((10, y), f"intervened: {iv[0]} ticks ({iv[1]:.2f} s)", fill=red if iv[0] else grey)
        y += 18
        block = (extras or {}).get("block")
        d.text((10, y), f"block {block[0]}/{block[1]}" if block else "", fill=red)
        y += 22
        # scrolling plot (last 100 ticks): h_named blue, Q red, unnamed min orange
        x0, y0, pw, ph = 10, y, W_PANEL - 20, H_PANEL - y - 12
        d.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(200, 200, 200))
        lo, hi = -1.05, 1.05

        def sy(v):
            return y0 + ph - (np.clip(v, lo, hi) - lo) / (hi - lo) * ph

        d.line([x0, sy(0.0), x0 + pw, sy(0.0)], fill=black)
        for hist, col in ((self.hist_other, (230, 150, 30)), (self.hist_named, (30, 100, 200)),
                          (self.hist_q, (220, 60, 50))):
            h = hist[-100:]
            n = len(h)
            seg = []
            for i, v in enumerate(h):
                if v == v:
                    seg.append((x0 + (i / max(n - 1, 1)) * pw, sy(v)))
                elif len(seg) > 1:
                    d.line(seg, fill=col, width=2)
                    seg = []
                else:
                    seg = []
            if len(seg) > 1:
                d.line(seg, fill=col, width=2)
        d.text((x0 + 4, y0 + 4), "h_named", fill=(30, 100, 200))
        d.text((x0 + 60, y0 + 4), "Q", fill=(220, 60, 50))
        d.text((x0 + 74, y0 + 4), "unnamed min", fill=(230, 150, 30))
        return img, harm_now

    def _panel_pair(self, img, d, y, ids, hs, trace, control, extras):
        """Two sentences on the same frames: A (the named rule) and B (the counterfactual). Q(A) solid red, Q(B)
        dashed purple, h(A) thin blue, h(B) thin dashed light blue; a red border while A's item is harmed."""
        red, grey, black = (200, 60, 40), (90, 90, 90), (0, 0, 0)
        ca, cb, cha, chb = (220, 60, 50), (130, 60, 170), (30, 100, 200), (110, 170, 230)

        def val(rid):
            return hs[ids.index(rid)] if rid in ids else float("nan")

        h_a, h_b = val(self.named_id), val(self.cf_id)
        q_a = trace["V"][-1] if trace.get("V") else float("nan")
        q_b = trace["Q_cf"][-1] if trace.get("Q_cf") else float("nan")
        for tag, sent, col, h_, q_ in (("A", self.sentence, ca, h_a, q_a), ("B", self.cf_sentence, cb, h_b, q_b)):
            for k, ln in enumerate(textwrap.wrap(f'{tag}: "{sent}"', 44)[:2]):
                d.text((10, y), ln, fill=col)
                y += 16
            d.text((10, y), f"   Q = {q_:+.3f}    h = {h_:+.3f}", fill=red if h_ < 0 else black)
            y += 20
        d.text((10, y), f"{control}  (filter off: nominal motion)" if control != "FILTER" else control, fill=grey)
        y += 22
        self.hist_named.append(h_a), self.hist_q.append(q_a), self.hist_hcf.append(h_b), self.hist_qcf.append(q_b)
        harm_now = h_a < 0
        x0, y0, pw, ph = 10, y, W_PANEL - 20, H_PANEL - y - 12
        d.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(200, 200, 200))
        lo, hi = -1.05, 1.05

        def sy(v):
            return y0 + ph - (np.clip(v, lo, hi) - lo) / (hi - lo) * ph

        d.line([x0, sy(0.0), x0 + pw, sy(0.0)], fill=black)
        for hist, col, width, dashed in ((self.hist_hcf, chb, 1, True), (self.hist_named, cha, 1, False),
                                         (self.hist_qcf, cb, 2, True), (self.hist_q, ca, 2, False)):
            h = hist[-150:]
            n = len(h)
            pts = [(x0 + (i / max(n - 1, 1)) * pw, sy(v)) if v == v else None for i, v in enumerate(h)]
            for i in range(len(pts) - 1):
                if pts[i] is None or pts[i + 1] is None or (dashed and (i // 3) % 2):
                    continue
                d.line([pts[i], pts[i + 1]], fill=col, width=width)
        d.text((x0 + 4, y0 + 4), "Q(A)", fill=ca)
        d.text((x0 + 40, y0 + 4), "Q(B) - -", fill=cb)
        d.text((x0 + 100, y0 + 4), "h(A)", fill=cha)
        d.text((x0 + 136, y0 + 4), "h(B) - -", fill=chb)
        self.harmed |= bool(harm_now)
        return img, harm_now

    def add(self, frame_rgb, t, step, n_steps, h, V, control, ratio, extras=None):
        panel, harm_now = self._panel_lang(t, step, control, ratio, extras)
        frame = np.array(frame_rgb, copy=True)
        if harm_now:   # red border on ticks where the named rule is harmed
            frame[:6], frame[-6:], frame[:, :6], frame[:, -6:] = (220, 30, 30), (220, 30, 30), (220, 30, 30), (220, 30, 30)
        ph = frame.shape[0]
        p = np.asarray(panel)
        if p.shape[0] != ph:
            from PIL import Image

            p = np.asarray(panel.resize((int(W_PANEL * ph / H_PANEL), ph)))
        self.writer.append_data(np.hstack([frame, p]))

    def discard(self) -> None:
        """Close without publishing (a lost writer lock): remove only this process's partial file."""
        try:
            self.writer.close()
        except Exception:
            pass
        try:
            self.tmp.unlink()
        except OSError:
            pass
        self.path = None

    def close(self, success: bool | None = None):
        try:
            self.writer.close()
        except Exception:
            pass
        if success is None:
            success = bool(self.ro.trace.get("success", False))
        outcome = "harmed" if self.harmed else ("ok" if success else "fail")
        final = self.final_stem.with_name(f"{self.final_stem.name}_{outcome}.mp4")
        try:
            os.replace(self.tmp, final)
            self.path = final
        except OSError:
            self.path = self.tmp
        return self.path
