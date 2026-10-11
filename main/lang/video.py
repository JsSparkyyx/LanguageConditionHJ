"""LangPanelVideoWriter (VANILLA_PLAN 8.4): observer frame + a panel in h units.

Keeps a reference to the controller and draws from its current state (the
base class's centimetre ``h`` argument is ignored):

- filter episodes: the sentence and its rule in plain words ("don't touch: red mug"); the named object's h
  (+1 safe, 0 harm, -1 badly harmed) and the filter's Q for the sentence (it acts when Q < 0); the other
  sentence the run also scores but never acts on (Q_cf, a rule of the opposite designation) with its Q; the
  most endangered OTHER object (lowest h among the rules the sentence does not name: harm the filter is not
  asked to prevent); FILTER / NOMINAL, |a| ratio, intervened ticks, block counter; a scrolling plot of those
  values with a legend; a red border while the named object's h < 0 (Yuxuan 2026-10-11: readable labels);
- nominal episodes: the most endangered object and its h.

File name ``<task>[_<host>]_<embodiment>_s<seed>_<kind>[_<axis>]_<outcome>.mp4``:
kind nominal / harm / benign, outcome ok / harmed / fail, decided at close.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np

from main.train.eval_utils import PanelVideoWriter

W_PANEL, H_PANEL = 400, 480
FONT_PX = 14
WRAP = 46   # characters per line at FONT_PX in W_PANEL

# plain words for the record's consequence names (the rule id is "<consequence>|<item>|<region or ->")
CONS_WORDS = {
    "avoid": "don't touch", "upright": "keep upright", "stay": "don't move", "zone_exclude": "keep out of",
    "zone_include": "keep on", "contain_particles": "don't spill", "exclude_particles": "no beads in",
}


def _font(px: int = FONT_PX):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=px)   # Pillow >= 10.1: a scalable font
    except TypeError:
        return ImageFont.load_default()


def _thing(entry: dict | None) -> str:
    """An object or region of the episode record in words: size, colour, category ("tall blue block")."""
    if not entry:
        return "?"
    name = entry.get("category") or (entry.get("names") or [entry.get("region_id") or entry.get("item_id")])[0]
    return " ".join(str(w) for w in (entry.get("size"), entry.get("color"), name) if w)


def rule_words(rule: dict | None, record: dict | None) -> str:
    """'don't touch: red mug' for a rule dict of the episode record; the bare rule id without a record."""
    if rule is None:
        return "-"
    if record is None:
        return str(rule.get("rule_id"))
    subj, ref = rule.get("subject") or {}, rule.get("reference") or {}
    objs, regs = record.get("objects") or [], record.get("regions") or []
    if "object" in subj:
        what = _thing(objs[subj["object"]])
    elif "particles" in subj:
        what = _thing(objs[subj["particles"]])
    elif "region" in ref:
        what = _thing(regs[ref["region"]])
    else:
        what = str(rule.get("rule_id"))
    return f"{CONS_WORDS.get(rule.get('consequence'), rule.get('consequence'))}: {what}"


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
                 fps: float = 25.0, cf_rule: dict | None = None, cf_sentence: str | None = None,
                 record: dict | None = None, alt_rule: dict | None = None, alt_sentence: str | None = None):
        """cf_rule / cf_sentence: the two-sentence grounding panel (A and B plotted side by side).
        record: the episode record (rules and objects in words). alt_rule / alt_sentence: the counterfactual
        sentence a filter episode also scores every tick (trace Q_cf) and never acts on, shown by name."""
        stem_path = Path(stem_path)
        self.final_stem = stem_path
        # unique per process: a stale writer and the new owner record the same seed (review round 3, C8)
        self.tmp = stem_path.with_name(f"{stem_path.name}.{os.getpid()}.partial.mp4")
        super().__init__(self.tmp, fps=fps, h_scale=1.0)
        self.ro = ro
        self.named = named_rule
        self.named_id = named_rule["rule_id"] if named_rule else None
        self.sentence = sentence
        self.hist_named, self.hist_q, self.hist_other, self.hist_qalt = [], [], [], []
        # grounding panel (review round 3, E0a): a second sentence B on the same frames, Q(B) and h(B) plotted too
        self.cf_id = cf_rule["rule_id"] if cf_rule else None
        self.cf_sentence = cf_sentence
        self.hist_hcf, self.hist_qcf = [], []
        self.record = record
        self.by_id = {r["rule_id"]: r for r in (record or {}).get("rules", [])}
        self.alt_rule, self.alt_sentence = alt_rule, alt_sentence
        self.font, self.small = _font(FONT_PX), _font(FONT_PX - 2)
        self.harmed = False
        self.path = None
        runner = ro.env._item_runner
        from .rollout import single_rule

        self.single = {r.rule_id for r in runner.rules if single_rule(r)}

    def _words(self, rid) -> str:
        return rule_words(self.by_id.get(rid), self.record) if rid in self.by_id else str(rid)

    def _panel_lang(self, t, step, control, ratio, extras):
        from PIL import Image, ImageDraw

        ids, hs = _rule_hs(self.ro)
        trace = self.ro.trace
        img = Image.new("RGB", (W_PANEL, H_PANEL), "white")
        d = ImageDraw.Draw(img)
        red, grey, black, blue, purple = (200, 60, 40), (90, 90, 90), (0, 0, 0), (20, 60, 160), (130, 60, 170)
        F, S, LH = self.font, self.small, FONT_PX + 4

        def line(y, text, fill=black, font=None):
            d.text((10, y), text, fill=fill, font=font or F)
            return y + LH

        y = line(8, f"t = {t:5.2f} s   tick {step}", grey, S)
        if self.named_id is not None and self.cf_id is not None:
            return self._panel_pair(img, d, y, ids, hs, trace, control, extras)
        if self.named_id is not None:
            j = ids.index(self.named_id) if self.named_id in ids else None
            h_named = hs[j] if j is not None else float("nan")
            q = trace["V"][-1] if trace.get("V") else float("nan")
            q_alt = trace["Q_cf"][-1] if trace.get("Q_cf") else float("nan")
            y = line(y, "Sentence:", grey, S)
            for ln in textwrap.wrap(f'"{self.sentence}"', WRAP)[:2]:
                y = line(y, ln, blue)
            y = line(y, f"  rule  {self._words(self.named_id)}"[:WRAP + 4], grey, S)
            y = line(y + 2, f"named object  h = {h_named:+.2f}", red if h_named < 0 else black)
            y = line(y, f"filter        Q = {q:+.2f}   (acts when Q < 0)", red if q < 0 else black)
            if self.alt_sentence:
                y = line(y + 4, "Other sentence (scored, never acted on):", grey, S)
                for ln in textwrap.wrap(f'"{self.alt_sentence}"', WRAP + 6)[:2]:
                    y = line(y, ln, purple, S)
                y = line(y, f"              Q_cf = {q_alt:+.2f}", purple)
            others = [(hv, rid) for rid, hv in zip(ids, hs) if rid != self.named_id and rid in self.single and hv == hv]
            o_h, o_id = min(others) if others else (float("nan"), "-")
            y = line(y + 4, "Most endangered other object (not named):", grey, S)
            y = line(y, f"  {self._words(o_id)}"[:WRAP], grey)
            y = line(y, f"              h = {o_h:+.2f}", red if o_h < 0 else grey)
            self.hist_named.append(h_named)
            self.hist_q.append(q)
            self.hist_qalt.append(q_alt)
            self.hist_other.append(o_h)
            harm_now = h_named < 0
        else:
            valid = [(hv, rid) for rid, hv in zip(ids, hs) if hv == hv]
            o_h, o_id = min(valid) if valid else (float("nan"), "-")
            y = line(y, "Nominal run: no sentence, filter off", grey)
            y = line(y + 4, "Most endangered object:", grey, S)
            y = line(y, f"  {self._words(o_id)}"[:WRAP], black)
            y = line(y, f"              h = {o_h:+.2f}", red if o_h < 0 else black)
            self.hist_named.append(float("nan"))
            self.hist_q.append(float("nan"))
            self.hist_qalt.append(float("nan"))
            self.hist_other.append(o_h)
            harm_now = o_h < 0
        self.harmed |= bool(harm_now)
        iv = (extras or {}).get("intervened") or (0, 0.0)
        block = (extras or {}).get("block")
        y = line(y + 4, f"{control}   |a| = {ratio:.2f} x max" + (f"   block {block[0]}/{block[1]}" if block else ""),
                 red if control == "FILTER" else grey, S)
        y = line(y, f"filter took over: {iv[0]} ticks ({iv[1]:.2f} s)", red if iv[0] else grey, S)
        # scrolling plot (last 100 ticks), h scale: +1 safe, 0 harm, -1 badly harmed
        x0, y0, pw, ph = 34, y + 18, W_PANEL - 44, H_PANEL - y - 40
        d.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(200, 200, 200))
        lo, hi = -1.05, 1.05

        def sy(v):
            return y0 + ph - (np.clip(v, lo, hi) - lo) / (hi - lo) * ph

        d.line([x0, sy(0.0), x0 + pw, sy(0.0)], fill=black)
        for v, lab in ((1.0, "+1"), (0.0, " 0"), (-1.0, "-1")):
            d.text((8, sy(v) - 7), lab, fill=grey, font=S)
        series = ((self.hist_other, (230, 150, 30), False), (self.hist_named, (30, 100, 200), False),
                  (self.hist_qalt, purple, True), (self.hist_q, (220, 60, 50), False))
        for hist, col, dashed in series:
            h = hist[-100:]
            n = len(h)
            pts = [(x0 + (i / max(n - 1, 1)) * pw, sy(v)) if v == v else None for i, v in enumerate(h)]
            for i in range(len(pts) - 1):
                if pts[i] is None or pts[i + 1] is None or (dashed and (i // 3) % 2):
                    continue
                d.line([pts[i], pts[i + 1]], fill=col, width=2)
        if self.named_id is None:
            legend = [("h, most endangered object", (230, 150, 30))]
        else:
            legend = [("Q", (220, 60, 50)), ("h named", (30, 100, 200)), ("h other object", (230, 150, 30))]
            if self.alt_sentence:
                legend.insert(1, ("Q_cf", purple))
        lx = x0
        for lab, col in legend:
            d.text((lx, y0 - 16), lab, fill=col, font=S)
            lx += 10 + 8 * len(lab)
        d.text((x0, y0 + ph + 4), "h: +1 safe, 0 harmed, -1 badly harmed", fill=grey, font=S)
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
            for k, ln in enumerate(textwrap.wrap(f'{tag}: "{sent}"', WRAP)[:2]):
                d.text((10, y), ln, fill=col, font=self.font)
                y += FONT_PX + 4
            rid = self.named_id if tag == "A" else self.cf_id
            d.text((10, y), f"   rule  {self._words(rid)}"[:WRAP + 4], fill=grey, font=self.small)
            y += FONT_PX + 2
            d.text((10, y), f"   Q = {q_:+.2f}    h = {h_:+.2f}", fill=red if h_ < 0 else black, font=self.font)
            y += FONT_PX + 8
        d.text((10, y), f"{control}  (filter off: nominal motion)" if control != "FILTER" else control, fill=grey,
               font=self.small)
        y += FONT_PX + 8
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
        lx = x0 + 4
        for lab, col in (("Q(A)", ca), ("Q(B) - -", cb), ("h(A)", cha), ("h(B) - -", chb)):
            d.text((lx, y0 + 4), lab, fill=col, font=self.small)
            lx += 14 + 8 * len(lab)
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
