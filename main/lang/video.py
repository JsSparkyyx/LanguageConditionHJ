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
    return [r.rule_id for r in rules], [float(rule_h(r, v)) for r, v in zip(rules, m)]


class LangPanelVideoWriter(PanelVideoWriter):
    def __init__(self, stem_path, ro, named_rule: dict | None = None, sentence: str | None = None,
                 fps: float = 25.0):
        stem_path = Path(stem_path)
        self.final_stem = stem_path
        self.tmp = stem_path.with_name(stem_path.name + ".partial.mp4")
        super().__init__(self.tmp, fps=fps, h_scale=1.0)
        self.ro = ro
        self.named = named_rule
        self.named_id = named_rule["rule_id"] if named_rule else None
        self.sentence = sentence
        self.hist_named, self.hist_q, self.hist_other = [], [], []
        self.harmed = False
        self.path = None
        runner = ro.env._item_runner
        self.single = {r.rule_id for r in runner.rules
                       if len(r.item_ids) <= 1 and len(r.region_ids) <= 1 and not getattr(r, "ambiguous", False)}

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
