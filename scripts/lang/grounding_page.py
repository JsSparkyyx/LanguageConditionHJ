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
details > summary { cursor: pointer; color: var(--accent); font-weight: 600; padding-block: 6px; }
details[open] > summary { margin-bottom: 12px; }
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
E0_DIR = REPO / "runs/lang/lang_full_paired_off_1007/eval_offline"
OTHER_SETS = ("val20_e0", "checkpoint_final_round3")   # T3 / T5 scores: the ur5 20-per-task set if scored, else the older one
TASK_NAMES = {"selective_avoid": "1 · Don't touch it", "gap_place": "7 · Don't disturb it",
              "push_zone": "3 · Push carefully", "tray_transport": "5 · Don't drop the block"}


def _metrics(name: str) -> tuple[dict, dict] | None:
    f = E0_DIR / name / "metrics.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d.get("metrics", {}), d.get("data", {})


def _g(M: dict, key: str) -> dict:
    return M.get(f"train/symmetric/{key}") or {}


def _t(m: dict, k: str) -> tuple:
    """(value, lo, hi) of metric k in group m (nan when absent)."""
    v = m.get(k)
    lo, hi = (m.get(k + "_ci") or (float("nan"), float("nan")))
    return (float("nan") if v is None else float(v), float(lo), float(hi))


def found_numbers() -> dict:
    """Section 2 and the header, from the official checkpoint's offline scores: T1 / T7 on the grounding set
    (eval_offline/grounding_e0), T3 / T5 on val20_e0 if scored, else on the older round-3 set (stated)."""
    M, D = _metrics("grounding_e0")
    rc = _g(M, "rc_pool")
    out = {"sets": [], "rows": [], "cons": _t(rc, "rank_acc_consequence"), "ref": _t(rc, "rank_acc_referent"),
           "ref_prior": _t(rc, "rank_acc_referent_text"), "pool_minus_prior": _t(_g(M, "pooled"), "model_minus_text"),
           "M": M}
    out["sets"].append(("tasks 1 and 7", "grounding_e0", D.get("embodiments"),
                        {t: _g(M, f"task/{t}").get("n_episodes") for t in ("selective_avoid", "gap_place")}))
    for t in ("selective_avoid", "gap_place"):
        out["rows"].append((t, _g(M, f"task_state/{t}/violated"), _g(M, f"task_state/{t}/not_yet"), _g(M, f"task/{t}")))
    for name in OTHER_SETS:
        r = _metrics(name)
        if r and _g(r[0], "task/push_zone"):
            M2, D2 = r
            out["sets"].append(("tasks 3 and 5", name, D2.get("embodiments"),
                                {t: _g(M2, f"task/{t}").get("n_episodes") for t in ("push_zone", "tray_transport")}))
            for t in ("push_zone", "tray_transport"):
                out["rows"].append((t, _g(M2, f"task_state/{t}/violated"), _g(M2, f"task_state/{t}/not_yet"),
                                    _g(M2, f"task/{t}")))
            out["M2"] = M2
            break
    return out


def lede(f: dict) -> str:
    c, r = f["cons"], f["ref"]
    if c[1] > 0.5 and not r[1] > f["ref_prior"][2]:
        return ("It reads which kind of rule a sentence gives better than which object the sentence names: on the "
                "object question it is not distinguishable from reading the words alone.")
    if r[1] > f["ref_prior"][2]:
        return "It tells apart both the kind of rule and the named object better than the words alone."
    return "Neither the kind of rule nor the named object is read clearly better than chance."


def section2(f: dict) -> str:
    def cell(m, k="rank_acc", ref="rank_acc_h"):
        if not m:
            return "–"
        return f"{m[k]:.2f} / {m.get(ref, float('nan')):.2f} ({m.get('n_pairs', 0)})"
    rows = "\n".join(f'<tr><td>{e(TASK_NAMES[t])}</td><td class="num">{cell(v)}</td><td class="num">{cell(n)}</td>'
                     f'<td class="num">{ci(_t(a, "rank_acc_text"))}</td></tr>' for t, v, n, a in f["rows"])
    def one_set(what, name, embs, eps):
        robots = ", ".join(e(x) for x in (embs or ["all embodiments"]))
        counts = ", ".join(f"task {TASK_NAMES[t].split(' ')[0]}: {n}" for t, n in eps.items())
        return f"{what}: {robots} validation episodes ({counts} with pairs), <code>{e(name)}</code>"
    sets = "; ".join(one_set(*x) for x in f["sets"])
    M = f["M"]
    t1v, t1n = _g(M, "task_state/selective_avoid/violated"), _g(M, "task_state/selective_avoid/not_yet")
    t7v, t7n = _g(M, "task_state/gap_place/violated"), _g(M, "task_state/gap_place/not_yet")
    L = ['<section aria-labelledby="found-h">', '<h2 id="found-h">2. What we found</h2>',
         "<p>The official model (20,000 steps on 420 recorded episodes of the careless controller) on separate "
         "validation episodes. Cells: model / simulator reference (pairs). Brackets are 95% intervals from "
         "resampling whole episodes; they are optimistic with this few episodes.</p>",
         '<div class="tablebox"><table>',
         "<thead><tr><th>Task</th><th>Already harmed (model / simulator)</th><th>Not yet (model / simulator)</th>"
         "<th>Sentence only (all moments)</th></tr></thead>", f"<tbody>\n{rows}\n</tbody></table></div>",
         f'<p class="muted">Sets: {sets}.</p>', "<ul>",
         f"<li><b>Same object, different rule:</b> {ci(f['cons'])}. <b>Different objects, same rule:</b> "
         f"{ci(f['ref'])}; the sentence-only reference on the same pairs {ci(f['ref_prior'])} (tasks 1 and 7).</li>"]
    if t1n:
        gap = t1n["rank_acc_h"] - t1n["rank_acc"]
        L.append(f"<li><b>Task 1, not yet harmed.</b> For \"don't touch\", the simulator's number is essentially the "
                 f"arm's distance to the named object, and ranking by it scores {t1n['rank_acc_h']:.2f}. The model "
                 f"scores {ci(_t(t1n, 'rank_acc'))}" + (", so it does not use the distance to the object the sentence "
                 "names as well as that number does.</li>" if gap > 0.05 else ".</li>"))
    if t1v:
        r = _t(t1v, "rank_acc_referent")
        L.append(f"<li><b>Task 1, already harmed, different objects with the same rule.</b> {ci(r)} against 1.00 for the "
                 f"simulator ({t1v.get('n_referent', 0)} pairs): in {1 - r[0]:.0%} of them the model rates the sentence "
                 "naming the object just hit as the safer one.</li>")
    if t7v and t7n:
        coin = all(_t(m, "rank_acc")[1] <= 0.5 for m in (t7v, t7n))
        L.append(f"<li><b>Task 7:</b> {ci(_t(t7v, 'rank_acc'))} already harmed, {ci(_t(t7n, 'rank_acc'))} not yet"
                 + (", not distinguishable from a coin flip on either kind of moment.</li>" if coin else ".</li>"))
    M2 = f.get("M2")
    if M2:
        sw, nsw = _g(M2, "t5/size_word"), _g(M2, "t5/no_size_word")
        zi = [(k, v) for k, v in M2.items() if k.startswith("train/symmetric/t3_benign/")]
        n_zi = sum(v.get("n_pairs", 0) for k, v in zi if k.endswith("/zone_include"))
        n_t3 = sum(v.get("n_pairs", 0) for k, v in zi)
        if sw and nsw:
            L.append(f"<li><b>Tasks 3 and 5 each have a shortcut.</b> In task 5 the block at risk is almost always "
                     f"the tall one: {ci(_t(sw, 'rank_acc'))} when a sentence contains \"tall\" or \"short\", "
                     f"{ci(_t(nsw, 'rank_acc'))} when it does not. In task 3 the safe rule is \"keep the box on the "
                     f"table\" in {n_zi} of {n_t3} pairs, a rule that never breaks.</li>")
    L.append(f"<li><b>Model minus the sentence-only reference</b> (tasks 1 and 7, all pairs, paired): "
             f"{ci(f['pool_minus_prior'])}.</li>")
    L += ["</ul>", "</section>"]
    return "\n".join(L)


def seeing_reading(svs: dict) -> str:
    """The E0b reading, from the state-paired difference (any - named) and its interval (round 4, item 4)."""
    parts = []
    for t, name in (("selective_avoid", "task 1"), ("gap_place", "task 7")):
        a = svs[f"{t}/all"]
        d, (lo, hi) = a["any_minus_named"], a["any_minus_named_ci"]
        strata = []
        for st, sname in (("already_harmed", "already harmed"), ("not_yet", "harm coming"),
                          ("neither", "no harm near")):
            b = svs.get(f"{t}/{st}")
            if b and b.get("any_minus_named_ci"):
                blo, bhi = b["any_minus_named_ci"]
                if blo > 0 or bhi < 0:
                    strata.append(f"{sname} {b['any_minus_named']:+.2f} [{blo:+.2f}, {bhi:+.2f}]")
        tail = ("; by kind of moment: " + ", ".join(strata)) if strata else "; inconclusive per kind of moment"
        if lo > 0:
            parts.append(f"on {name} Q follows the most endangered object more closely than the named one "
                         f"(paired difference {d:+.2f} [{lo:+.2f}, {hi:+.2f}]{tail})")
        elif hi < 0:
            parts.append(f"on {name} Q follows the named object more closely than the most endangered one "
                         f"({d:+.2f} [{lo:+.2f}, {hi:+.2f}]{tail})")
        else:
            parts.append(f"on {name} there is no clear difference ({d:+.2f} [{lo:+.2f}, {hi:+.2f}])")
    rmax = max(max(abs(svs[f"{t}/all"]["rho_named"]), abs(svs[f"{t}/all"]["rho_any"]))
               for t in ("selective_avoid", "gap_place"))
    s = "Judged by the paired difference: " + "; ".join(parts) + ". "
    s += (f"Every correlation is weak (at most {rmax:.2f}): the model sees the danger only weakly, and ties it to the "
          "named object even less." if rmax < 0.4 else f"The correlations reach {rmax:.2f}.")
    return s


def special_reading(wa: dict) -> str:
    """Reason 2 from the E0c numbers (word_attention.json), stating only what was measured (audit 2026-10-08):
    the start token's attention mass and its multiple of its share per word-reading layer, both critics' last
    blocks, the key sizes with the counterexample, and the per-head entropies."""
    L = wa["layers"]["pooled"]
    names = list(L)
    first = next(n for n in names if "encoder block 1" in n)
    crit_last = {}
    for n in names:
        if "critic" in n:
            crit_last[n.split(" block")[0]] = n          # the last block of each critic, in list order
    f = L[first]
    ent = [L[n]["entropy_head_query"] for n in names]
    uni = f["entropy_uniform"]
    words_k = float(np.mean([f[k]["key_rms"] for k in ("referent", "other")]))
    crit = "; ".join(f"{n.split(', ')[-1]}: [CLS] {L[n]['cls']['mass']:.0%} ({L[n]['cls']['enrichment']:.1f}x), "
                     f"[SEP] {L[n]['sep']['mass']:.0%} ({L[n]['sep']['enrichment']:.1f}x)" for n in crit_last.values())
    srw = L.get("scene reads words")
    counter = (f" Large keys do not always mean more attention: where the scene reads the words, [CLS] has its largest "
               f"key ({srw['cls']['key_rms']:.0f}) but gets {srw['cls']['enrichment']:.1f}x its share." if srw else "")
    return (f"In the body's first word-reading layer, the start token [CLS] gets {f['cls']['mass']:.0%} of the attention "
            f"on the sentence, {f['cls']['enrichment']:.1f}x its share of the tokens, and the end token [SEP] "
            f"{f['sep']['mass']:.0%} ({f['sep']['enrichment']:.1f}x). In the two critics' last blocks: {crit}. Their keys "
            f"are larger than the words' (RMS {f['cls']['key_rms']:.0f} and {f['sep']['key_rms']:.0f} vs {words_k:.0f}).{counter} "
            f"Each attention head is sharp: entropy {min(ent):.2f}-{max(ent):.2f} nats per head and query, against "
            f"{uni:.1f} for uniform, so each head reads essentially one token.")



def _ref_ah(name: str, run: Path | None = None) -> tuple | None:
    """(referent rank accuracy at already-harmed moments, T1 + T7, its interval, pairs) from an eval_offline
    output (rc_state/violated: the same pairs as the primary metric)."""
    f = (run or REPO / "runs/lang/lang_full_paired_off_1007") / "eval_offline" / name / "metrics.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())["metrics"].get("train/symmetric/rc_state/violated") or {}
    if m.get("rank_acc_referent") is None:
        return None
    return (*_t(m, "rank_acc_referent"), m.get("n_referent", 0))


def why_bullet() -> str:
    """The header's 'why' bullet: the official model on its training scenes vs new scenes (section 4)."""
    tr, va = _ref_ah("grounding_e0_train"), _ref_ah("grounding_e0")
    if not (tr and va):
        return ""
    return (f"<li><b>Why</b>It memorizes its training scenes: on different-object pairs at already-harmed moments it scores "
            f"{tr[0]:.2f} on the episodes it trained on and {va[0]:.2f} on new ones.</li>")


def section_memorize(summ) -> str:
    """Section 4 (audit 2026-10-08): the same score on the scenes the model trained on and on new scenes; harm-
    coming pairs on training scenes against the simulator's present-state number; the validation learning
    curve; the two-robot data-scale run. Every number is read from the eval outputs."""
    L = ['<section aria-labelledby="mem-h">', '<h2 id="mem-h">4. Training scenes versus new scenes</h2>']
    off_val, off_tr = _ref_ah("grounding_e0"), _ref_ah("grounding_e0_train")
    ex, st = (summ or {}).get("_extra", {}), (summ or {}).get("_settings", {})
    runs = [(n, x["train"]) for n, x in sorted(ex.items()) if x.get("train")]
    if not off_tr and not runs:
        L.append("<p>Running.</p></section>")
        return "\n".join(L)
    L.append("<p>The score of section 2 for different objects at already-harmed moments, on the episodes each model "
             "trained on and on new episodes of the same tasks. A model that has learned to find the named object scores "
             "about the same on both.</p>")
    rows = []
    if off_val and off_tr:
        rows.append(("Official model (20,000 steps, all tasks, both robots)", off_tr, off_val, None))
    for n, tr in runs:
        v, sd = n.rsplit("_", 2)[-2:]
        val = (summ.get(v, {}).get("referent/already_harmed/T1 + T7", {}).get("per_seed", {}) or {}).get(sd.lstrip("s"))
        rows.append((f"{v.upper()} seed {sd.lstrip('s')} (6,000 steps, tasks 1 and 7)", (tr[0], None, None, tr[1]),
                     (val, None, None, None), st.get(n, {}).get("train_pair_acc")))
    L.append('<div class="tablebox"><table><thead><tr><th>Model</th><th>Training scenes</th><th>New scenes</th>'
             '</tr></thead><tbody>')
    fmt = lambda t: "–" if t is None or t[0] is None else (ci(t[:3]) if t[1] is not None and t[1] == t[1] else f"{t[0]:.2f}")  # noqa: E731
    for name, tr, va, ib in rows:
        L.append(f'<tr><td>{e(name)}</td><td class="num">{fmt(tr)}</td><td class="num">{fmt(va)}</td></tr>')
    L.append("</tbody></table></div>")
    # harm-coming pairs on training scenes vs the simulator's present number
    run0 = next((REPO / "runs/lang/grounding_1008" / n for n, _ in runs if "_v0_" in n), None)
    f = run0 / "eval_offline/grounding_train/metrics.json" if run0 else None
    if f and f.exists():
        M = json.loads(f.read_text())["metrics"]
        cells = []
        for t, name in (("selective_avoid", "task 1"), ("gap_place", "task 7")):
            m = M.get(f"train/symmetric/task_state/{t}/not_yet") or {}
            if m:
                cells.append(f"{name} {m['rank_acc']:.2f} against {m['rank_acc_h']:.2f}")
        if cells:
            L.append("<p><b>It remembers what happens next in its training episodes.</b> On training scenes, before the "
                     "harm, V0 ranks the pairs better than the simulator's present safety number does (" +
                     "; ".join(cells) + "). The present number cannot see which object will be hit a second later; a model "
                     "can know that only from having seen the episode.</p>")
    curves = {n: x["curve"] for n, x in ex.items() if x.get("curve")}
    if curves:
        n, c = sorted(curves.items())[0]
        v, sd = n.rsplit("_", 2)[-2:]
        fin = (summ.get(v, {}).get("referent/already_harmed/T1 + T7", {}).get("per_seed", {}) or {}).get(sd.lstrip("s"))
        pts = ", ".join(f"{int(k):,} steps {a[0]:.2f}" for k, a in sorted(c.items(), key=lambda kv: int(kv[0])))
        run_c = REPO / "runs/lang/grounding_1008" / n
        trc = [(k, _ref_ah(f"grounding_train_step{k}", run_c)) for k in (1000, 2000, 3000, 4000, 5000)]
        trc = [(k, x) for k, x in trc if x]
        tr_txt = ("; on its training scenes: " + ", ".join(f"{k:,} steps {x[0]:.2f}" for k, x in trc)) if trc else ""
        L.append(f"<p><b>More training does not help on new scenes.</b> {e(v.upper())} seed {e(sd.lstrip('s'))} on new "
                 f"scenes by checkpoint: {pts}" + (f", 6,000 steps {fin:.2f}" if fin is not None else "") + tr_txt + ".</p>")
    d2 = [(_ref_ah("grounding", REPO / f"runs/lang/grounding_1008/g1008_d2_s{k}"), k) for k in (0, 1)]
    d2 = [(x, k) for x, k in d2 if x]
    if d2:
        L.append("<p><b>Twice the scenes.</b> V0 trained on tasks 1 and 7 from both robots (120 training episodes "
                 "instead of 60), scored on the same new ur5 scenes: " +
                 "; ".join(f"seed {k} {ci(x[:3])}" for x, k in d2) + ".</p>")
    L.append("</section>")
    return "\n".join(L)



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
    found = found_numbers()
    P = []
    P.append(f"""<header>
<span class="eyebrow">Language-conditioned safety filter · first results</span>
<h1>Which object does the sentence name?</h1>
<p class="lede">We trained the first version of the language-conditioned safety filter and tested whether it reads the sentence. {lede(found)} This page shows the evidence, a few small experiments that test why, and what they point to.</p>
<ul class="points">
<li><b>The kind of rule</b>{ci(found["cons"])} when two sentences name the same object with different rules (tasks 1 and 7; 0.5 is a coin flip).</li>
<li><b>The object</b>{ci(found["ref"])} when two sentences name different objects with the same rule; words alone give {ci(found["ref_prior"])}.</li>
{why_bullet()}
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
    P.append(section2(found))
    # section 3
    t1 = [k for k in traces if "selective_avoid" in k]
    t7 = [k for k in traces if "gap_place" in k]

    def best(keys):   # the trace that separates best before harm, and one that does not
        s = sorted(keys, key=lambda k: separation(traces[k])[1])
        return s[0], s[-1]
    fig_t1, fig_t7 = best(t1)[0], best(t7)[0]
    n_sep = sum(1 for k in traces if separation(traces[k])[1] < -0.1)
    reading = seeing_reading(svs)
    P.append(f"""<section aria-labelledby="see-h">
<h2 id="see-h">3. Seeing versus selecting</h2>
<p>The model's backbone is CrossSafe, built to measure the distance between the arm and everything around it. So the question is whether it sees the danger but ties it to the wrong object.</p>
<p class="tests"><b>Example:</b> the arm carries a plate past a red mug, 3 cm from its path, and a blue box 25 cm away. A filter that reads the sentence gives "don't touch the red mug" a low Q and "don't touch the blue box" a high one. A filter that sees danger but not the name gives both the same low Q.</p>
<p>Over task 1 and task 7 moments ({svs['selective_avoid/all']['n_episodes']} and {svs['gap_place/all']['n_episodes']} ur5 validation episodes), how well Q follows the named object's safety number: {rhoc('selective_avoid', 'rho_named')} (task 1), {rhoc('gap_place', 'rho_named')} (task 7); and the most endangered object's number, whichever is named: {rhoc('selective_avoid', 'rho_any')}, {rhoc('gap_place', 'rho_any')}. (Rank correlations, from −1 to 1.)</p>
<p>{e(reading)}</p>
<h3>One episode per task, filter off</h3>
<p class="muted">Q for the sentence naming the object at risk (A), and for the sentence naming a safe object (B), on the same nominal run; thin lines are the two objects' safety numbers. Hover for values. Over the six recorded scenes, Q(A) drops clearly below Q(B) in the second before the harm in {n_sep} of 6.</p>
{trace_svg(traces[fig_t1], "trace-t1")}
{trace_svg(traces[fig_t7], "trace-t7")}
<div class="videos">{"".join(video_fig(vid(traces[k]["video"]), traces[k], ("Task 1" if "selective" in k else "Task 7") + " · UR5 robot") for k in (best(t1) + best(t7)))}</div>
</section>""")
    tok = (f" The frozen text encoder computes the start token from itself alone: its feature is the same vector for "
           f"every sentence (cosine to the mean {sp.get('cls', {}).get('mean_cos_to_mean', float('nan')):.4f} over "
           f"{sp.get('n_sentences', 0)} sentences), so it carries no name. The end token is computed from the whole "
           f"sentence (relative spread {sp.get('sep', {}).get('rel_std', float('nan')):.2f}, against "
           f"{sp.get('cls', {}).get('rel_std', float('nan')):.4f} for the start token), so it can carry the name."
           if sp else "")
    P.append(section_memorize(summ))
    P.append(f"""<section aria-labelledby="why-h">
<h2 id="why-h">5. Three other possible reasons</h2>
<div class="measure"><b>1. The word features enter too large.</b><p>The frozen text encoder's word features are about 9 times larger than everything else, with a few channels up to 150. They enter the model's word attention unnormalized, so attention can lock onto whichever words are large rather than the object's name.</p></div>
<div class="measure"><b>2. Attention lands on the sentence's start token.</b><p>{e(special_reading(wa))}{e(tok)}</p></div>
<div class="measure"><b>3. Nothing ties a phrase to one object.</b><p>The images are 3,600 small patches, and an object is spread over dozens of them. The model has to learn, from the safety loss alone, which patches "the red mug" refers to. Its only teacher is pairs of sentences on the same moment.</p></div>
<p>Reasons 1 and 2 are cheap to fix. Reason 3 needs a design change. Section 6 tests them.</p>
</section>""")
    # section 5
    P.append(section5(summ, e3, vid))
    P.append(section6(summ))
    P.append(f"""<section aria-labelledby="cav-h">
<h2 id="cav-h">8. Caveats</h2>
<ul>
<li>One checkpoint, and short runs for the experiments.</li>
<li>Validation episodes are few: {svs['selective_avoid/all']['n_episodes']} for task 1 and {svs['gap_place/all']['n_episodes']} for task 7 in the experiments (20 new per task plus the earlier ur5 ones; two new task-1 episodes failed at their first tick when the host task found no grasp).</li>
<li>Offline scores, not closed loop. The filter's own corrective moves are not tested here.</li>
<li>Some task-1 sentences name an object by a word another object in the scene also fits ("the box" next to a "coffee box", "the bottle" next to a "pill bottle"), about one in eight task-1 pairs. The model scores about the same on those pairs as on the rest, so they do not explain the low score. Fixing it changes the sentence generator, which needs a decision first.</li>
<li>Task 1 "don't touch" is the case where a geometric controller could also work once the object is known. Its value here is as the cleanest test of reading the name.</li>
</ul>
</section>""")
    body = "\n".join(P).replace("{PROPOSE}", propose(summ)).replace("{TASK_PAGE}", TASK_PAGE)
    page = (f"<title>Which object does the sentence name?</title>\n{FONTS}\n<style>{STYLE}{EXTRA_STYLE}</style>\n"
            f'<div class="wrap">\n{body}\n<footer><p>RoboTwin simulation. The official model was trained on the '
            f'ur5-wsg and franka-panda robots; tasks 1 / 7 and the small runs use ur5-wsg; the sets under the section 2 '
            f'table say which robots each row covers. Code and numbers: docs/lang_filter/grounding_1008.md in the '
            f'repository.</p></footer>\n</div>\n{TRACE_JS}')
    return page


def outcome(summ) -> str:
    return (summ or {}).get("_e4", {}).get("outcome", "")


def status(summ) -> str:
    return (summ or {}).get("_e4", {}).get("status", "")


def propose(summ) -> str:
    """The header's last bullet, from the summary's decision status (audit 2026-10-08: an interim outcome is shown
    as pending, never as a conclusion)."""
    o, st = outcome(summ), status(summ)
    if st in ("final_a", "final_b"):
        return "Keep the model and adopt the change that fixed it: " + e(o.split(": ", 1)[1]) + "."
    if st == "final_slots":
        return ("Neither the text fixes nor V4 reach the bar, but marking the named object does: add an explicit link "
                "from the sentence to one object, object slots plus a pointer.")
    if st == "inconclusive_oracle":
        return ("No fix reaches the bar, and the model ignores even the oracle mark on the named object, so the agreed "
                "rule cannot decide yet: see sections 4, 6 and 7.")
    if st == "final_other":
        return "The small experiments do not point at object reading alone: " + e(o) + "."
    if st == "pending":
        return "Pending: " + e(pending_text(summ)) + "."
    return "Written once the experiments are in."


def pending_text(summ) -> str:
    """The pending decision in plain words (the summary's outcome string is for the results file)."""
    pr = (summ or {}).get("_e4", {}).get("primary", {})
    got = lambda v: pr.get(v) and pr[v][0] == pr[v][0]  # noqa: E731
    parts = []
    if got("v1") or got("v2"):
        parts.append("the text fixes V1 and V2 do not reach the bar (" + ", ".join(
            f"{v.upper()} {pr[v][0]:.2f}" for v in ("v1", "v2") if got(v)) + " against 0.85)")
    if not got("v4"):
        parts.append("V4, which trains one readout query to point at the named object, is still training")
    elif not got("v4b"):
        parts.append("V4b, which pulls that query's attention toward the named object, is still training")
    return "; ".join(parts) if parts else outcome(summ)


ROWS = [("v0", "nothing (short-run baseline)"), ("v1", "word features normalized"),
        ("v2", "V1 + the two special tokens hidden from the word attention"),
        ("v4", "an extra training signal: one readout query predicts where the named object is (simulator labels, training only)"),
        ("v4b", "an extra training signal: one readout query's attention on the image patches is pulled toward the named object"),
        ("v3", "oracle: the named object's patches are marked for the model (simulator only, a ceiling)")]


def train_val(summ) -> str:
    """Whether V3 reads its mark (audit 2026-10-08), from the summary; the training-vs-new-scene table is section 4."""
    tags = [(n, x["tag_use"]) for n, x in sorted((summ or {}).get("_extra", {}).items()) if x.get("tag_use")]
    if not tags:
        return ""
    return ("<p><b>Does V3 read its mark?</b> The same V3 model scored with the mark on the named object removed: " +
            "; ".join(f"seed {e(n.rsplit('_s', 1)[1])}: {t['flip_frac']:.0%} of the {t['n_pairs']} pairs change, "
                      f"{t['acc_tag']:.2f} with the mark vs {t['acc_no_tag']:.2f} without" for n, t in tags) +
            ". The model fits its training scenes without the mark (section 4), so nothing pushes it to use it.</p>")


def e3_tally(e3: dict) -> str:
    """How the two Q lines behave in the second before the harm, per variant, on the same recorded scenes."""
    from collections import Counter

    by = {}
    official = [json.loads(f.read_text()) for f in sorted((G / "e0a").glob("v0_*.json"))]
    for d in official + list(e3.values()):
        d = dict(d, variant="official" if d in official else d["variant"])
        pre = separation(d)[1]
        k = "sep" if pre == pre and pre < -0.1 else "wrong" if pre == pre and pre > 0.1 else "none"
        by.setdefault(d["variant"], Counter())[k] += 1
    order = lambda v: (v != "official", v)  # noqa: E731
    rows = "".join(f'<tr><td>{e("Official model" if v == "official" else v.upper())}</td><td class="num">{c["sep"]}</td><td class="num">{c["none"]}</td>'
                   f'<td class="num">{c["wrong"]}</td><td class="num">{sum(c.values())}</td></tr>' for v, c in sorted(by.items(), key=lambda kv: order(kv[0])))
    return ("<h3>The same scenes, run with each model</h3><p>Each model watches the same nominal runs (task 1 and task 7, three "
            "scenes each) with the two sentences of section 3. In the second before the harm, does Q for the sentence about the "
            "object at risk drop below Q for the safe one?</p>"
            '<div class="tablebox"><table><thead><tr><th>Model</th><th>Separates</th><th>No separation</th><th>Wrong order</th>'
            f"<th>Scenes</th></tr></thead><tbody>{rows}</tbody></table></div>")


def section5(summ, e3, vid) -> str:
    L = ['<section aria-labelledby="exp-h">', '<h2 id="exp-h">6. Small experiments</h2>']
    if not summ:
        L.append("<p>Running.</p></section>")
        return "\n".join(L)
    st0 = next((x for n, x in sorted(summ.get("_settings", {}).items()) if "_v0_" in n), {})
    p0 = summ.get("v0", {}).get("referent/already_harmed/T1 + T7", {})
    seeds = ", ".join(f"{v.upper()} {len(summ[v]['seeds'])}" for v, _ in ROWS if summ.get(v))
    L.append(f"<p>Short training runs on tasks 1 and 7 only ({st0.get('steps', '?')} steps, the UR5 robot), scored "
             f"on the same validation pairs ({p0.get('n_pairs', '?')} primary pairs in {p0.get('n_episodes', '?')} "
             f"episodes). Seeds scored: {seeds}. Cells: the mean over the scored seeds [95% interval from resampling "
             "episodes], each seed after it.</p>")
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
    L.append(train_val(summ))
    st = summ.get("_settings", {})
    v0s = next((x for n, x in sorted(st.items()) if "_v0_" in n), None)
    if v0s:
        L.append(f"<p class=\"muted\">Held equal across runs: gamma {v0s['gamma']}, actor head "
                 f"{'linear' if not v0s['actor_hidden'] else 'MLP ' + str(v0s['actor_hidden'])}, TF32 "
                 f"{'on' if v0s['tf32_train'] else 'off'}, {v0s['steps']} steps, tasks {', '.join(v0s['tasks'])}, "
                 f"robot {', '.join(v0s['embodiments'])}.</p>")
    ratios = [(n, x.get("aux_ratio_500"), x.get("att_ratio_500")) for n, x in sorted(st.items())
              if x.get("aux_ratio_500") is not None or x.get("att_ratio_500") is not None]
    if ratios:
        L.append("<p>Size of the extra training signal, as a fraction of the safety loss at about 500 steps "
                 "(it counts as tried at 0.05 or more): " + "; ".join(
                     f"{e(n.split('_', 1)[1])} {r if r is not None else a:.2f}" for n, r, a in ratios) + ".</p>")
    if summ.get("_refused"):
        L.append("<p class=\"muted\">Left out (not comparable): " + "; ".join(
            f"{e(n)} ({e('; '.join(w))})" for n, w in summ["_refused"].items()) + ".</p>")
    L.append(f"<p><b>Decision rule</b> (agreed beforehand): if one of the text fixes (V1, V2) or the extra training signal (V4) "
             f"reaches 0.85 on different-object pairs at already-harmed moments, tasks 1 and 7 pooled, we keep the model "
             f"with that change; only if neither the text fixes nor V4 do but the oracle (V3) does, we propose object "
             f"slots plus a pointer. "
             f"<b>Outcome:</b> {e(pending_text(summ) if status(summ) == 'pending' else outcome(summ))}.</p>")
    if e3:
        L.append(e3_tally(e3))
        L.append('<details><summary>All scene videos of the small runs</summary><div class="videos">' + "".join(
            video_fig(vid(d["video"]), d, f'{d["variant"].upper()} · ' + ("Task 1" if d["task"] == "selective_avoid" else "Task 7"))
            for d in sorted(e3.values(), key=lambda d: (d["variant"], d["task"], d.get("seed", 0)))) + "</div></details>")
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
    o, st = outcome(summ), status(summ)
    head = '<section aria-labelledby="prop-h">\n<h2 id="prop-h">7. {title}</h2>'
    if st == "final_slots":
        return (head.format(title="Proposed change: object slots and a pointer") + DIAGRAM + """
<ul>
<li><b>What stays:</b> no rule types or task types anywhere, the kind of rule is still read from the sentence by the safety loss, and the same data and relabeling.</li>
<li><b>What is new:</b> two training signals that use the simulator's labels during training only: which patches belong to which object (slot grouping), and which slot the sentence names (pointer).</li>
<li>V3 is the ceiling of this design: it is given the pointer's answer.</li>
</ul></section>""")
    if st in ("final_a", "final_b"):
        return head.format(title="What the experiments point to") + f"<p>{e(o)}.</p></section>"
    if st == "final_other":
        return (head.format(title="What the experiments point to") +
                f"<p>{e(o)}. Slots plus a pointer are not proposed: the decision rule calls for them only when neither "
                "the text fixes nor V4 reach 0.85 and the oracle does.</p></section>")
    if st == "inconclusive_oracle":
        return (head.format(title="What the experiments point to") + f"<p>{e(o)}.</p>"
                "<p>An oracle the model does not read sets no ceiling, so it says nothing either way about slots plus a "
                "pointer. A pointer comes with its own supervised loss, which the model cannot ignore the way it "
                "ignores an added tag.</p></section>")
    if st == "pending":
        return head.format(title="Status") + f"<p>Pending: {e(pending_text(summ))}. The decision follows the agreed rule once the remaining runs are scored.</p></section>"
    return head.format(title="Status") + "<p>Written once the experiments are in.</p></section>"


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
