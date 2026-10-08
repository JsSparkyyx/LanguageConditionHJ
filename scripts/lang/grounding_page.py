"""Build the coauthor page "Which object does the sentence name?" (review round 3, Part II.1) from the results.

    python scripts/lang/grounding_page.py --out runs/lang/grounding_1008/page

Inputs (all under runs/): the official checkpoint's E0 outputs (eval_offline/grounding_e0: seeing_vs_selecting,
word_attention), the E0a / E3 traces (runs/lang/grounding_1008/e0a, e3) and videos, and the E2 / V4 summary
(runs/lang/grounding_1008/summary.json from grounding_summary.py). Writes <out>/index.html and copies the videos
it shows into <out>/videos/. The style is the "Named-Object Safety Tasks" page's (artifact PA41QH8buYwXauFEuhPcLR),
copied verbatim, plus a few rules for the trace figures.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
G = REPO / "runs/lang/grounding_1008"
E0 = REPO / "runs/lang/lang_full_paired_off_1007/eval_offline/grounding_e0"
TASK_PAGE = "https://claude.ai/artifact/PA41QH8buYwXauFEuhPcLR"

STYLE = """
/* layout: a catalogue — intro + legend, a summary table, then one section per task with its two videos side by side */
:root {
  --bg: #f3f5f8;
  --surface: #ffffff;
  --fg: #1b2230;
  --muted: #586374;
  --line: #d9dfe7;
  --accent: #2b57c9;
  --safe: #1d8a4b;
  --unsafe: #c4342a;
  --chip: #e8edf5;
  --font-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --font-body: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #11151b; --surface: #181e26; --fg: #e5eaf0; --muted: #9ca8b6; --line: #2a333e;
    --accent: #84a2ff; --safe: #4cc47d; --unsafe: #ff6e62; --chip: #222b36; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #11151b; --surface: #181e26; --fg: #e5eaf0; --muted: #9ca8b6; --line: #2a333e;
  --accent: #84a2ff; --safe: #4cc47d; --unsafe: #ff6e62; --chip: #222b36; color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); font-family: var(--font-body);
  font-size: 1rem; line-height: 1.6; padding-inline: 16px; padding-block: 40px 72px;
}
.wrap { max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 56px; }
h1, h2, h3 { font-family: var(--font-display); text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(2rem, 4.5vw, 2.9rem); font-weight: 800; letter-spacing: -0.01em; }
h2 { font-size: clamp(1.45rem, 3vw, 1.9rem); font-weight: 700; }
h3 { font-size: 1.05rem; font-weight: 700; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font-family: var(--font-display); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent); }
.lede { font-size: 1.13rem; color: var(--fg); }
.muted { color: var(--muted); }
header { display: flex; flex-direction: column; gap: 16px; }
.points { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 8px 0 0; padding: 0; list-style: none; }
.points li { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; min-width: 0; }
.points b { display: block; font-family: var(--font-display); margin-bottom: 4px; }
.legend { display: flex; flex-direction: column; gap: 12px; }
.banner { font-family: var(--font-mono); font-size: 0.86rem; background: #1d232c; color: #f2f4f7; border-radius: 8px; padding: 12px 16px; line-height: 1.7; overflow-x: auto; }
.banner .ok { color: #66d98c; } .banner .bad { color: #ff7d72; }
.tablebox { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; min-width: 720px; font-size: 0.94rem; }
th, td { text-align: left; padding: 11px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-family: var(--font-display); font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); font-weight: 700; }
tr:last-child td { border-bottom: 0; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
td a { color: var(--accent); text-decoration: none; font-weight: 600; }
td a:hover { text-decoration: underline; }
.task { display: flex; flex-direction: column; gap: 18px; padding-top: 8px; border-top: 2px solid var(--line); scroll-margin-top: 16px; }
.task-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 16px; }
.tag { font-family: var(--font-mono); font-size: 0.8rem; background: var(--chip); color: var(--muted); padding: 2px 8px; border-radius: 6px; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px 28px; }
.facts div { min-width: 0; }
.facts h3 { font-size: 0.8rem; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }
.quote { font-family: var(--font-mono); font-size: 0.9rem; }
.tests { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 8px; padding: 12px 16px; }
.tests b { font-family: var(--font-display); }
.measure { display: flex; flex-direction: column; gap: 4px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; }
.measure b { font-family: var(--font-display); }
.scale { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.scale div { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; min-width: 0; }
.scale b { display: block; font-family: var(--font-mono); font-size: 1.15rem; margin-bottom: 2px; }
.scale .neg b { color: var(--unsafe); } .scale .zero b { color: var(--muted); } .scale .pos b { color: var(--safe); }
.videos { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
figure { margin: 0; display: flex; flex-direction: column; gap: 8px; min-width: 0; }
video { width: 100%; max-width: 100%; border-radius: 8px; background: #000; aspect-ratio: 4 / 3; display: block; }
.videos.wide { grid-template-columns: 1fr; }
.videos.wide video { aspect-ratio: 1120 / 480; }
figcaption { font-size: 0.9rem; color: var(--muted); }
figcaption .pill { display: inline-block; font-family: var(--font-display); font-weight: 700; font-size: 0.74rem; letter-spacing: 0.05em; text-transform: uppercase; padding: 1px 8px; border-radius: 999px; margin-right: 6px; color: #fff; }
.pill.bad { background: var(--unsafe); } .pill.ok { background: var(--safe); }
footer { color: var(--muted); font-size: 0.88rem; }
a:focus-visible, video:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
"""

EXTRA_STYLE = """
/* this page: sections with prose + one table or figure each; trace figures with a hover readout */
section { display: flex; flex-direction: column; gap: 14px; }
section ul { margin: 0; padding-left: 1.2em; max-width: 72ch; display: flex; flex-direction: column; gap: 6px; }
a { color: var(--accent); }
.trace { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; position: relative; }
.trace svg { width: 100%; height: auto; display: block; overflow: visible; }
.trace .axis text, .trace .lab { fill: var(--muted); font-family: var(--font-mono); font-size: 11px; }
.trace .grid { stroke: var(--line); stroke-width: 1; }
.trace .zero { stroke: var(--muted); stroke-width: 1; }
.trace .band { fill: var(--unsafe); fill-opacity: 0.12; }
.trace .qa { stroke: var(--unsafe); stroke-width: 2; fill: none; }
.trace .qb { stroke: var(--accent); stroke-width: 2; fill: none; stroke-dasharray: 6 4; }
.trace .ha { stroke: var(--unsafe); stroke-width: 1; fill: none; stroke-opacity: 0.55; }
.trace .hb { stroke: var(--accent); stroke-width: 1; fill: none; stroke-opacity: 0.55; stroke-dasharray: 3 3; }
.trace .cross { stroke: var(--muted); stroke-width: 1; }
.trace .keys { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.86rem; color: var(--muted); margin-bottom: 6px; }
.trace .keys span { display: inline-flex; align-items: center; gap: 6px; }
.trace .keys svg { width: 26px; height: 10px; }
.trace .tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
  padding: 6px 9px; font-family: var(--font-mono); font-size: 0.76rem; color: var(--fg); white-space: nowrap; }
.diagram { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; overflow-x: auto; }
.diagram svg { min-width: 680px; width: 100%; height: auto; display: block; }
.diagram .box { fill: var(--chip); stroke: var(--line); }
.diagram .new { fill: var(--surface); stroke: var(--unsafe); stroke-width: 2; }
.diagram text { fill: var(--fg); font-family: var(--font-body); font-size: 12px; }
.diagram .row { fill: var(--muted); font-family: var(--font-display); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; }
.diagram .arrow { stroke: var(--muted); stroke-width: 1.5; fill: none; marker-end: url(#ah); }
.diagram .arrow.hot { stroke: var(--unsafe); stroke-width: 2.5; marker-end: url(#ahr); }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Public+Sans:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">')


def e(s) -> str:
    return html.escape(str(s))


def ci(t, nd=2) -> str:
    """(mean, lo, hi) -> '0.53 [0.45, 0.61]'."""
    m, lo, hi = t
    if m != m:
        return "–"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" if lo == lo else f"{m:.{nd}f}"


# ---------------------------------------------------------------------------- trace figure
def trace_svg(d: dict, fig_id: str) -> str:
    """Q(A), Q(B), h(A), h(B) over one nominal episode, the harm band of A shaded; hover readout via JS."""
    t, ha, hb = np.array(d["t"]), np.array(d["h_a"]), np.array(d["h_b"])
    qt, qa = np.array(d["q_t"]), np.array(d["q_a"])
    qb = np.interp(qt, np.array(d["q_b_t"]), np.array(d["q_b"])) if d["q_b_t"] else np.full_like(qt, np.nan)
    W, H, L, R, T, B = 720, 250, 44, 14, 10, 30
    x0, x1 = 0.0, float(max(t.max(), qt.max()))
    sx = lambda v: L + (v - x0) / max(x1 - x0, 1e-9) * (W - L - R)  # noqa: E731
    sy = lambda v: T + (1.05 - np.clip(v, -1.05, 1.05)) / 2.1 * (H - T - B)  # noqa: E731

    def path(xs, ys):
        pts = [f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, ys) if b == b]
        return "M" + " L".join(pts) if pts else ""

    band = []
    harm = ha < 0
    i = 0
    while i < len(t):
        if harm[i]:
            j = i
            while j + 1 < len(t) and harm[j + 1]:
                j += 1
            band.append(f'<rect class="band" x="{sx(t[i]):.1f}" y="{T}" width="{max(sx(t[j]) - sx(t[i]), 2):.1f}" height="{H - T - B}"/>')
            i = j + 1
        else:
            i += 1
    grid = "".join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>' for v in (-1, 1))
    yt = "".join(f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:+.0f}' + "</text>" if v else
                 f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">0</text>' for v in (-1, 0, 1))
    step = 2 if x1 <= 16 else 4
    xt = "".join(f'<text x="{sx(v):.1f}" y="{H - 10}" text-anchor="middle">{v:.0f} s</text>' for v in np.arange(0, x1 + 1e-9, step))
    data = {"t": [round(float(x), 3) for x in qt], "qa": [round(float(x), 3) for x in qa], "qb": [round(float(x), 3) for x in qb],
            "ha": [round(float(x), 3) for x in np.interp(qt, t, ha)], "hb": [round(float(x), 3) for x in np.interp(qt, t, hb)],
            "x0": x0, "x1": x1, "W": W, "L": L, "R": R}
    keys = ('<div class="keys">'
            '<span><svg viewBox="0 0 26 10"><line x1="0" y1="5" x2="26" y2="5" class="qa" style="stroke:var(--unsafe);stroke-width:2"/></svg>Q, sentence A (object at risk)</span>'
            '<span><svg viewBox="0 0 26 10"><line x1="0" y1="5" x2="26" y2="5" style="stroke:var(--accent);stroke-width:2;stroke-dasharray:6 4"/></svg>Q, sentence B (safe object)</span>'
            '<span><svg viewBox="0 0 26 10"><line x1="0" y1="5" x2="26" y2="5" style="stroke:var(--unsafe);stroke-width:1;stroke-opacity:.55"/></svg>h of A\'s object</span>'
            '<span><svg viewBox="0 0 26 10"><line x1="0" y1="5" x2="26" y2="5" style="stroke:var(--accent);stroke-width:1;stroke-opacity:.55;stroke-dasharray:3 3"/></svg>h of B\'s object</span>'
            '<span><svg viewBox="0 0 26 10"><rect x="0" y="0" width="26" height="10" style="fill:var(--unsafe);fill-opacity:.12"/></svg>A\'s object harmed</span></div>')
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Q and h of two sentences over one episode">'
           f'{"".join(band)}{grid}<line class="zero" x1="{L}" x2="{W - R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>'
           f'<g class="axis">{yt}{xt}</g>'
           f'<path class="hb" d="{path(t, hb)}"/><path class="ha" d="{path(t, ha)}"/>'
           f'<path class="qb" d="{path(qt, qb)}"/><path class="qa" d="{path(qt, qa)}"/>'
           f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{H - B}" visibility="hidden"/></svg>')
    return (f'<div class="trace" id="{fig_id}" data-trace=\'{e(json.dumps(data))}\'>{keys}{svg}'
            f'<div class="tip" hidden></div></div>')


TRACE_JS = """
<script>
(function () {
  document.querySelectorAll(".trace").forEach(function (box) {
    var d; try { d = JSON.parse(box.getAttribute("data-trace")); } catch (err) { return; }
    var svg = box.querySelector("svg"), cross = box.querySelector(".cross"), tip = box.querySelector(".tip");
    function f(v) { return (v >= 0 ? "+" : "") + v.toFixed(2); }
    function show(ev) {
      var r = svg.getBoundingClientRect(), xv = (ev.clientX - r.left) / r.width * d.W;
      var tv = d.x0 + (xv - d.L) / (d.W - d.L - d.R) * (d.x1 - d.x0), k = 0, best = 1e9;
      for (var i = 0; i < d.t.length; i++) { var g = Math.abs(d.t[i] - tv); if (g < best) { best = g; k = i; } }
      var px = d.L + (d.t[k] - d.x0) / (d.x1 - d.x0) * (d.W - d.L - d.R);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      tip.textContent = d.t[k].toFixed(1) + " s   Q(A) " + f(d.qa[k]) + "  Q(B) " + f(d.qb[k]) + "   h(A) " + f(d.ha[k]) + "  h(B) " + f(d.hb[k]);
      tip.hidden = false;
      var bx = box.getBoundingClientRect(), left = ev.clientX - bx.left + 12;
      tip.style.left = Math.min(left, bx.width - tip.offsetWidth - 8) + "px"; tip.style.top = "40px";
    }
    svg.addEventListener("mousemove", show);
    svg.addEventListener("mouseleave", function () { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
  });
})();
</script>
"""


def separation(d: dict) -> tuple[float | None, float, float]:
    """(first harm time of A, mean Q(A) - Q(B) in the last second before it, the same after it)."""
    t, ha = np.array(d["t"]), np.array(d["h_a"])
    qt, qa = np.array(d["q_t"]), np.array(d["q_a"])
    qb = np.interp(qt, np.array(d["q_b_t"]), np.array(d["q_b"])) if d["q_b_t"] else np.full_like(qt, np.nan)
    h = np.flatnonzero(ha < 0)
    th = float(t[h[0]]) if h.size else None
    pre = (qt < th) & (qt >= th - 1.0) if th is not None else np.zeros_like(qt, bool)
    post = qt >= th if th is not None else np.zeros_like(qt, bool)
    sep = qa - qb
    return th, float(sep[pre].mean()) if pre.any() else float("nan"), float(sep[post].mean()) if post.any() else float("nan")


def video_fig(src: str, d: dict, label: str) -> str:
    th, pre, post = separation(d)
    if pre == pre and pre < -0.1:
        verdict = f"Q(A) drops below Q(B) before the harm (by {abs(pre):.2f} over the last second): the sentence about the object at risk reads lower."
        pill = '<span class="pill ok">Separates</span>'
    elif pre == pre and pre > 0.1:
        verdict = f"Q(A) stays above Q(B) before the harm (by {pre:.2f}): the order is wrong."
        pill = '<span class="pill bad">Wrong order</span>'
    else:
        verdict = "The two Q lines do not separate before the harm."
        pill = '<span class="pill bad">No separation</span>'
    return (f'<figure><video src="{e(src)}" controls muted playsinline preload="metadata"></video><figcaption>{pill}'
            f'{e(label)}. A: <span class="quote">"{e(d["sentence_a"])}"</span> (its object is harmed at {th:.1f} s); '
            f'B: <span class="quote">"{e(d["sentence_b"])}"</span>. The panel plots Q(A) solid red and Q(B) dashed purple. '
            f'{e(verdict)}</figcaption></figure>')


# ---------------------------------------------------------------------------- page
def build(out: Path) -> str:
    svs = json.loads((E0 / "seeing_vs_selecting.json").read_text())
    wa = json.loads((E0 / "word_attention.json").read_text())
    summ = json.loads((G / "summary.json").read_text()) if (G / "summary.json").exists() else None
    traces = {p.stem: json.loads(p.read_text()) for p in sorted((G / "e0a").glob("v0_*.json"))}
    e3 = {}
    for p in sorted((G / "e3").glob("*.json")) if (G / "e3").exists() else []:
        if p.name != "selection.json":
            e3[p.stem] = json.loads(p.read_text())
    (out / "videos").mkdir(parents=True, exist_ok=True)

    def vid(path: str) -> str:
        p = Path(path)
        if p.exists():
            shutil.copy2(p, out / "videos" / p.name)
        return f"videos/{p.name}"

    rho = lambda task, k: svs[f"{task}/all"][k]  # noqa: E731
    rhoc = lambda task, k: ci((svs[f"{task}/all"][k], *svs[f"{task}/all"][k + "_ci"]))  # noqa: E731
    sp = wa.get("special_tokens", {})
    P = []
    P.append(f"""<header>
<span class="eyebrow">Language-conditioned safety filter · first results</span>
<h1>Which object does the sentence name?</h1>
<p class="lede">We trained the first version of the language-conditioned safety filter. It learned to read what kind of rule a sentence gives ("don't touch" vs "don't move"). But on the tasks with several similar objects, it barely tells which object the sentence names. This page shows the evidence, a few small experiments that test why, and the design change we propose.</p>
<ul class="points">
<li><b>What it reads</b>The kind of rule: 0.70 when two sentences name the same object with different rules.</li>
<li><b>What it misses</b>The object: 0.53 when two sentences name different objects with the same rule (0.5 is a coin flip).</li>
<li><b>What we propose</b>{{PROPOSE}}</li>
</ul>
<p class="muted">The tasks, the careless controller and the safety numbers are described on the <a href="{TASK_PAGE}">Named-Object Safety Tasks</a> page.</p>
</header>""")
    P.append("""<section aria-labelledby="test-h">
<h2 id="test-h">1. How we test reading, without running the robot</h2>
<p>Take one moment of a recorded episode and two sentences. One names an object that is being harmed, or will be within the next second. The other names an object that stays safe. Both sentences go to the filter with the same camera images, the same arm state and the same next move, so only the sentence differs. The filter outputs a safety value Q for each. It reads the sentences correctly if Q is lower for the harmed object.</p>
<ul>
<li><b>Score:</b> the fraction of such pairs it gets right. 0.5 is a coin flip, 1.0 is perfect.</li>
<li><b>Two kinds of moments:</b> <i>Already harmed</i>: the named object has already been touched, moved or tipped, so the answer is visible in the images now. <i>Not yet</i>: the harm comes later in the next second.</li>
<li><b>Two references:</b> the <i>simulator reference</i> ranks the pair by the true safety number h of each named object at that moment. It knows which object each sentence names and sees its present state, but it cannot see the future. On "already harmed" moments it is right by construction (1.0). The <i>sentence-only reference</i> is a small classifier that guesses from the words alone, with no images, trained on the training data.</li>
</ul>
</section>""")
    P.append("""<section aria-labelledby="found-h">
<h2 id="found-h">2. What we found</h2>
<p>The model was trained for 20,000 steps on 420 recorded episodes of the careless controller, then scored on 41 separate validation episodes. Brackets are 95% intervals from resampling whole episodes. They are optimistic with 7 to 10 episodes per task.</p>
<div class="tablebox"><table>
<thead><tr><th>Task</th><th>Already harmed (model / simulator)</th><th>Not yet (model / simulator)</th><th>Sentence only (all moments)</th></tr></thead>
<tbody>
<tr><td>1 · Don't touch it</td><td class="num">0.69 / 1.00 (1272 pairs)</td><td class="num">0.48 / 0.71 (771)</td><td class="num">0.64</td></tr>
<tr><td>7 · Don't disturb it</td><td class="num">0.54 / 1.00 (602)</td><td class="num">0.55 / 0.59 (183)</td><td class="num">0.68</td></tr>
<tr><td>3 · Push carefully</td><td class="num">0.87 / 1.00 (75)</td><td class="num">0.68 / 0.23 (79)</td><td class="num">0.56</td></tr>
<tr><td>5 · Don't drop the block</td><td class="num">0.91 / 1.00 (130)</td><td class="num">0.94 / 0.61 (48)</td><td class="num">0.82</td></tr>
</tbody></table></div>
<ul>
<li><b>Same object, different rule:</b> 0.70 [0.54, 0.83] over tasks 1, 3 and 7. <b>Different objects, same rule:</b> 0.53 [0.45, 0.61].</li>
<li><b>Task 1, not yet harmed.</b> For "don't touch", the simulator's number is essentially the arm's distance to the named object, and ranking by it scores 0.71. The model scores 0.48. It does not use the distance to the object the sentence names.</li>
<li><b>Task 1, already harmed.</b> 0.69 against 1.00. In a third of these pairs the model rates "don't touch the red mug" as safer than "don't touch the blue box", even though the mug has just been hit.</li>
<li><b>Task 7:</b> a coin flip on both kinds of moments.</li>
<li><b>Tasks 3 and 5 look better, but each has a shortcut.</b> In task 5 the block at risk is always the tall one: 0.94 when a sentence contains "tall" or "short", 0.77 [0.44, 0.93] when it does not. In task 3 the safe rule is often "keep the box on the table", which never breaks.</li>
<li><b>Overall, the model is not yet distinguishable from the sentence-only reference:</b> −0.04 [−0.14, 0.07].</li>
</ul>
</section>""")
    # section 3
    t1 = [k for k in traces if "selective_avoid" in k]
    t7 = [k for k in traces if "gap_place" in k]

    def best(keys):   # the trace that separates best before harm, and one that does not
        s = sorted(keys, key=lambda k: separation(traces[k])[1])
        return s[0], s[-1]
    fig_t1, fig_t7 = best(t1)[0], best(t7)[0]
    n_sep = sum(1 for k in traces if separation(traces[k])[1] < -0.1)
    reading = (f"Q follows the most endangered object more closely than the named one on both tasks "
               f"(task 1: +{rho('selective_avoid', 'any_minus_named'):.2f} {rhoc('selective_avoid', 'any_minus_named').split(' ', 1)[1]}, "
               f"task 7: +{rho('gap_place', 'any_minus_named'):.2f} {rhoc('gap_place', 'any_minus_named').split(' ', 1)[1]} as paired differences), "
               f"but every correlation is weak (at most {max(rho(t, 'rho_any') for t in ('selective_avoid', 'gap_place')):.2f}) and the two intervals overlap: "
               f"the model sees the danger only weakly, and ties it to the named object even less.")
    P.append(f"""<section aria-labelledby="see-h">
<h2 id="see-h">3. Seeing versus selecting</h2>
<p>The model's backbone is CrossSafe, built to measure the distance between the arm and everything around it. So the question is whether it sees the danger but ties it to the wrong object.</p>
<p class="tests"><b>Example:</b> the arm carries a plate past a red mug, 3 cm from its path, and a blue box 25 cm away. A filter that reads the sentence gives "don't touch the red mug" a low Q and "don't touch the blue box" a high one. A filter that sees danger but not the name gives both the same low Q.</p>
<p>Over task 1 and task 7 moments (23 and 25 ur5 validation episodes), how well Q follows the named object's safety number: {rhoc('selective_avoid', 'rho_named')} (task 1), {rhoc('gap_place', 'rho_named')} (task 7); and the most endangered object's number, whichever is named: {rhoc('selective_avoid', 'rho_any')}, {rhoc('gap_place', 'rho_any')}. (Rank correlations, from −1 to 1.)</p>
<p>{e(reading)}</p>
<h3>One episode per task, filter off</h3>
<p class="muted">Q for the sentence naming the object at risk (A), and for the sentence naming a safe object (B), on the same nominal run; thin lines are the two objects' safety numbers. Hover for values. Over the six recorded scenes, Q(A) drops clearly below Q(B) in the second before the harm in {n_sep} of 6.</p>
{trace_svg(traces[fig_t1], "trace-t1")}
{trace_svg(traces[fig_t7], "trace-t7")}
<div class="videos">{"".join(video_fig(vid(traces[k]["video"]), traces[k], ("Task 1" if "selective" in k else "Task 7") + " · UR5 robot") for k in (best(t1) + best(t7)))}</div>
</section>""")
    tok = (f" Measured: the frozen start token's feature is the same vector for every sentence (cosine to the mean "
           f"{sp.get('cls', {}).get('mean_cos_to_mean', float('nan')):.4f} over {sp.get('n_sentences', 0)} sentences)."
           if sp else "")
    P.append(f"""<section aria-labelledby="why-h">
<h2 id="why-h">4. Three possible reasons</h2>
<div class="measure"><b>1. The word features enter too large.</b><p>The frozen text encoder's word features are about 9 times larger than everything else, with a few channels up to 150. They enter the model's word attention unnormalized, so attention can lock onto whichever words are large rather than the object's name.</p></div>
<div class="measure"><b>2. Attention lands on padding-like tokens.</b><p>In the first block, 45% of the attention on the sentence goes to the two special tokens that start and end every sentence, about twice their share. If the encoder computes the start token from itself alone, it is the same for every sentence and carries no name.{e(tok)}</p></div>
<div class="measure"><b>3. Nothing ties a phrase to one object.</b><p>The images are 3,600 small patches, and an object is spread over dozens of them. The model has to learn, from the safety loss alone, which patches "the red mug" refers to. Its only teacher is pairs of sentences on the same moment.</p></div>
<p>Reasons 1 and 2 are cheap to fix. Reason 3 needs a design change.</p>
</section>""")
    # section 5
    P.append(section5(summ, e3, vid))
    P.append(section6(summ))
    P.append("""<section aria-labelledby="cav-h">
<h2 id="cav-h">7. Caveats</h2>
<ul>
<li>One checkpoint, and short runs for the experiments.</li>
<li>Validation episodes are few: 23 for task 1 and 25 for task 7 in the experiments (20 new per task; two task-1 episodes failed at their first tick when the host task found no grasp).</li>
<li>Offline scores, not closed loop. The filter's own corrective moves are not tested here.</li>
<li>Task 1 "don't touch" is the case where a geometric controller could also work once the object is known. Its value here is as the cleanest test of reading the name.</li>
</ul>
</section>""")
    body = "\n".join(P).replace("{PROPOSE}", propose(summ)).replace("{TASK_PAGE}", TASK_PAGE)
    page = (f"<title>Which object does the sentence name?</title>\n{FONTS}\n<style>{STYLE}{EXTRA_STYLE}</style>\n"
            f'<div class="wrap">\n{body}\n<footer><p>RoboTwin simulation, ur5-wsg robot. Code and numbers: '
            f'docs/lang_filter/grounding_1008.md in the repository.</p></footer>\n</div>\n{TRACE_JS}')
    return page


def outcome(summ) -> str:
    return (summ or {}).get("_e4", {}).get("outcome", "")


def propose(summ) -> str:
    o = outcome(summ)
    if o.startswith("a fixes") or o.startswith("b fixes"):
        return "Keep the model and adopt the change that fixed it: " + e(o.split(": ", 1)[1]) + "."
    if "slots plus a pointer" in o:
        return "Neither cheap fix reaches the bar, but marking the named object does: add an explicit link from the sentence to one object, object slots plus a pointer."
    if o:
        return "The small experiments do not point at object reading alone: " + e(o) + "."
    return "If the small experiments confirm it, add an explicit link from the sentence to one object: object slots plus a pointer."


ROWS = [("v0", "nothing (short-run baseline)"), ("v1", "word features normalized"),
        ("v2", "V1 + the two special tokens hidden from the word attention"),
        ("v4", "an extra training signal: one readout query predicts where the named object is (simulator labels, training only)"),
        ("v3", "oracle: the named object's patches are marked for the model (simulator only, a ceiling)")]


def section5(summ, e3, vid) -> str:
    L = ['<section aria-labelledby="exp-h">', '<h2 id="exp-h">5. Small experiments</h2>',
         "<p>Short training runs on tasks 1 and 7 only (6,000 steps, two seeds each, the UR5 robot), scored on the same "
         "48 validation episodes. Cells: the mean of the two seeds [95% interval from resampling episodes], each seed after it.</p>"]
    if not summ:
        L.append("<p>Running.</p></section>")
        return "\n".join(L)
    L.append('<div class="tablebox"><table><thead><tr><th>Run</th><th>What changes</th>'
             '<th>Different objects, already harmed (T1 + T7 · T1 / T7)</th><th>Not yet (T1 + T7)</th>'
             '<th>Same object, different rule (already harmed)</th></tr></thead><tbody>')
    for v, what in ROWS:
        r = summ.get(v)
        if not r:
            continue
        g = lambda k: r.get(k, {})  # noqa: E731
        a = g("referent/already_harmed/T1 + T7")
        seeds = " / ".join(f"{x:.2f}" for x in a.get("per_seed", {}).values())
        cell = (f'{ci(a["mean"])} ({seeds}) · {g("referent/already_harmed/T1")["mean"][0]:.2f} / '
                f'{g("referent/already_harmed/T7")["mean"][0]:.2f}')
        L.append(f'<tr><td>{v.upper()}</td><td>{e(what)}</td><td class="num">{cell}</td>'
                 f'<td class="num">{ci(g("referent/not_yet/T1 + T7")["mean"])}</td>'
                 f'<td class="num">{ci(g("consequence/already_harmed/T1 + T7")["mean"])}</td></tr>')
    pr = summ.get("v0", {}).get("referent/already_harmed/T1 + T7", {}).get("prior")
    if pr:
        L.append(f'<tr><td>–</td><td>sentence-only reference, same pairs</td><td class="num">{ci(pr)}</td><td></td><td></td></tr>')
    L.append("</tbody></table></div>")
    aux = summ.get("_aux")
    if aux:
        L.append(f"<p>V4's prediction of where the named object is: off by {aux}.</p>")
    L.append(f"<p><b>Decision rule</b> (agreed beforehand): if one of the text fixes (V1, V2) or the extra training signal (V4) "
             f"reaches 0.85 on different-object pairs at already-harmed moments, tasks 1 and 7 pooled, we keep the model "
             f"with that change; only if neither does but the oracle (V3) does, we propose object slots plus a pointer. "
             f"<b>Outcome:</b> {e(outcome(summ))}.</p>")
    if e3:
        L.append('<div class="videos">' + "".join(
            video_fig(vid(d["video"]), d, f'{d["variant"].upper()} · ' + ("Task 1" if d["task"] == "selective_avoid" else "Task 7"))
            for d in e3.values()) + "</div>")
    L.append("</section>")
    return "\n".join(L)


DIAGRAM = """<div class="diagram"><svg viewBox="0 0 980 300" role="img" aria-label="Today's model and the proposed one">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker>
<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--unsafe)"/></marker></defs>
<text class="row" x="10" y="22">Today</text>
<rect class="box" x="10" y="34" width="120" height="44" rx="6"/><text x="70" y="61" text-anchor="middle">sentence</text>
<rect class="box" x="170" y="34" width="130" height="44" rx="6"/><text x="235" y="61" text-anchor="middle">text adapter → words</text>
<rect class="box" x="340" y="34" width="330" height="44" rx="6"/><text x="505" y="54" text-anchor="middle">arm links and 4 readout queries attend</text><text x="505" y="70" text-anchor="middle">over 3,600 image patches</text>
<rect class="box" x="710" y="34" width="80" height="44" rx="6"/><text x="750" y="61" text-anchor="middle">Q</text>
<path class="arrow" d="M130,56 L168,56"/><path class="arrow" d="M300,56 L338,56"/><path class="arrow" d="M670,56 L708,56"/>
<text class="row" x="10" y="132">Proposed</text>
<rect class="box" x="10" y="146" width="150" height="44" rx="6"/><text x="85" y="166" text-anchor="middle">image patches</text><text x="85" y="182" text-anchor="middle">(frames t, t−k, 0)</text>
<rect class="box" x="200" y="146" width="190" height="44" rx="6"/><text x="295" y="166" text-anchor="middle">object slots: one per object,</text><text x="295" y="182" text-anchor="middle">with position and motion</text>
<rect class="box" x="10" y="232" width="150" height="44" rx="6"/><text x="85" y="259" text-anchor="middle">sentence</text>
<rect class="box" x="200" y="232" width="190" height="44" rx="6"/><text x="295" y="259" text-anchor="middle">entity queries</text>
<rect class="new" x="430" y="188" width="170" height="48" rx="6"/><text x="515" y="208" text-anchor="middle">pointer: "is this the</text><text x="515" y="224" text-anchor="middle">named object?" → tags</text>
<rect class="box" x="640" y="170" width="200" height="84" rx="6"/><text x="740" y="196" text-anchor="middle">tags steer the arm links' attention;</text><text x="740" y="212" text-anchor="middle">tagged slots join the reasoning;</text><text x="740" y="228" text-anchor="middle">Q = minimum over arm links</text><text x="740" y="244" text-anchor="middle">and tagged slots</text>
<rect class="box" x="880" y="190" width="80" height="44" rx="6"/><text x="920" y="217" text-anchor="middle">Q</text>
<path class="arrow" d="M160,168 L198,168"/><path class="arrow" d="M160,254 L198,254"/>
<path class="arrow hot" d="M390,170 L428,200"/><path class="arrow hot" d="M390,252 L428,224"/>
<path class="arrow" d="M600,212 L638,212"/><path class="arrow" d="M840,212 L878,212"/>
<text x="409" y="286" style="fill:var(--unsafe);font-weight:600">the one new link: phrase → slot</text>
</svg></div>"""


def section6(summ) -> str:
    o = outcome(summ)
    head = '<section aria-labelledby="prop-h">\n<h2 id="prop-h">6. Proposed change: object slots and a pointer</h2>'
    if "slots plus a pointer" in o:
        return (head + DIAGRAM + """
<ul>
<li><b>What stays:</b> no rule types or task types anywhere, the kind of rule is still read from the sentence by the safety loss, and the same data and relabeling.</li>
<li><b>What is new:</b> two training signals that use the simulator's labels during training only: which patches belong to which object (slot grouping), and which slot the sentence names (pointer).</li>
<li>V3 is the ceiling of this design: it is given the pointer's answer.</li>
</ul></section>""")
    if not o:
        return head + "<p>Written once the experiments are in.</p></section>"
    return (head.replace("Proposed change: object slots and a pointer", "What the experiments point to") +
            f"<p>{e(o)}. Slots plus a pointer are not proposed: the decision rule calls for them only when neither cheap "
            "fix reaches 0.85 and the oracle does.</p></section>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(G / "page"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(build(out))
    print(f"[grounding_page] {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
