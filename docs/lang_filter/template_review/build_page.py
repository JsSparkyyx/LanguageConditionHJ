"""Build the constraint-template review page (2026-10-20) from the bank itself, my notes (flags.py) and real filled
sentences (examples.json, from recorded validation episodes with today's object words).

    python docs/lang_filter/template_review/build_page.py    # -> docs/lang_filter/template_review/review.html
"""

import html
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import flags  # noqa: E402

BANK = yaml.safe_load((REPO / "envs/main/item_envs/language/paraphrases.yaml").read_text())
OBJECTS = yaml.safe_load((REPO / "envs/main/item_envs/configs/objects.yaml").read_text())
EX = json.loads((HERE / "examples.json").read_text())
ORDER = ["avoid", "stay", "upright", "zone_exclude", "zone_include", "contain_particles", "exclude_particles"]
TITLE = {"avoid": "Avoid (don't touch)", "stay": "Stay (don't move)", "upright": "Upright (don't tip over)",
         "zone_exclude": "Keep out of a mat", "zone_include": "Keep on the table",
         "contain_particles": "Don't spill the beads", "exclude_particles": "Keep beads out of a bin or plate"}


def tpl(t: str) -> str:
    """A template with its slots marked."""
    out = html.escape(t)
    return re.sub(r"\{(obj|region|material)\}", lambda m: f'<span class="slot">{m.group(1)}</span>', out)


def count(cons, level, pred=lambda t: True):
    return sum(1 for (c, t), (lv, _) in flags.FLAGS.items() if c == cons and lv == level and pred(t))


n_stay = count("stay", "problem")
n_pour = count("contain_particles", "problem")
n_clean = sum(1 for (c, t), (lv, _) in flags.FLAGS.items() if c == "exclude_particles" and lv == "problem"
              and ("clean" in t or "empty" in t))
cross = [(lv, head, body.format(n_stay=n_stay, n_pour=n_pour, n_clean=n_clean)) for lv, head, body in flags.CROSS]

tot = {"problem": 0, "check": 0, "all": 0}
sections = []
for cons in ORDER:
    rule = flags.RULES[cons]
    rows = []
    n_p = n_c = 0
    for split in ("train", "heldout"):
        for t in BANK[cons][split]:
            lv, note = flags.FLAGS.get((cons, t), ("ok", ""))
            n_p += lv == "problem"
            n_c += lv == "check"
            badge = {"problem": "Problem", "check": "Check", "ok": "OK"}[lv]
            rows.append(f'<tr class="lv-{lv}" data-flag="{lv}"><td class="t">{tpl(t)}</td>'
                        f'<td class="sp">{"held out" if split == "heldout" else "training"}</td>'
                        f'<td><span class="badge b-{lv}">{badge}</span></td><td class="note">{html.escape(note)}</td></tr>')
    n_all = len(BANK[cons]["train"]) + len(BANK[cons]["heldout"])
    tot["problem"] += n_p
    tot["check"] += n_c
    tot["all"] += n_all
    exs = []
    for e in EX.get(cons, [])[:4]:
        for s in e["train"][:2]:
            exs.append(f'<li><span class="ex">{html.escape(s)}</span> <span class="ex-task">{html.escape(e["task"])}</span></li>')
    sections.append(f'''
<section id="{cons}" class="cons">
  <header class="cons-head">
    <h2>{html.escape(TITLE[cons])} <code class="key">{cons}</code></h2>
    <p class="counts"><span class="badge b-problem">{n_p} problem</span> <span class="badge b-check">{n_c} check</span>
      <span class="of">of {n_all} templates</span></p>
  </header>
  <div class="rule">
    <dl>
      <dt>Tasks</dt><dd>{html.escape(rule["tasks"])}</dd>
      <dt>What the rule measures</dt><dd>{html.escape(rule["measures"])}</dd>
      <dt>Scale</dt><dd>{html.escape(rule["scale"])}</dd>
    </dl>
  </div>
  <h3>Real sentences from recorded scenes</h3>
  <ul class="examples">{"".join(exs) or "<li>none recorded</li>"}</ul>
  <h3>Templates</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>Template</th><th>Split</th><th>My call</th><th>Why</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</section>''')

cross_html = "".join(f'<li class="x-{lv}"><span class="badge b-{lv}">{"Problem" if lv == "problem" else "Check"}</span>'
                     f'<div><strong>{html.escape(h)}</strong><p>{html.escape(b)}</p></div></li>' for lv, h, b in cross)
toc = "".join(f'<a href="#{c}">{html.escape(TITLE[c])}</a>' for c in ORDER)
words = []
for k in ("beads",):
    e = OBJECTS["categories"][k] if "categories" in OBJECTS and k in OBJECTS["categories"] else None
reg = OBJECTS.get("regions", {})
reg_rows = "".join(f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(', '.join(v.get('names', [])))}</td>"
                   f"<td>{html.escape(', '.join(v.get('synonyms', []) or []))}</td></tr>" for k, v in reg.items())
mat = None
for sec in OBJECTS.values():
    if isinstance(sec, dict) and "beads" in sec and isinstance(sec["beads"], dict):
        mat = sec["beads"]
mat_txt = ", ".join((mat or {}).get("names", []) + ((mat or {}).get("synonyms") or [])) if mat else "beads"
held_words = ", ".join(BANK.get("heldout_words", []))

page = f'''<title>Constraint Sentence Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; a sticky index of the seven rules; each rule = definition, examples, template table. */
:root {{
  --bg: #f5f7f8; --panel: #ffffff; --fg: #1d2529; --muted: #5b6970; --line: #dce3e6;
  --accent: #1f6f73; --slot-bg: #e3f1f1; --slot-fg: #165457;
  --p-bg: #fbe6e3; --p-fg: #9a2b1e; --c-bg: #fbf0d9; --c-fg: #85580d; --ok-bg: #edf1f2; --ok-fg: #5b6970;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12181b; --panel: #192125; --fg: #e3e9ec; --muted: #9aa8af; --line: #2b363b;
  --accent: #6cc3c7; --slot-bg: #1f3a3c; --slot-fg: #a6e0e2;
  --p-bg: #42201b; --p-fg: #f2a497; --c-bg: #3d2f12; --c-fg: #ecc574; --ok-bg: #222c30; --ok-fg: #9aa8af; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #12181b; --panel: #192125; --fg: #e3e9ec; --muted: #9aa8af; --line: #2b363b;
  --accent: #6cc3c7; --slot-bg: #1f3a3c; --slot-fg: #a6e0e2;
  --p-bg: #42201b; --p-fg: #f2a497; --c-bg: #3d2f12; --c-fg: #ecc574; --ok-bg: #222c30; --ok-fg: #9aa8af; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }}
.wrap {{ max-width: 1040px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
h1, h2 {{ font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; }}
h1 {{ font-size: 30px; letter-spacing: .2px; }}
h2 {{ font-size: 22px; }}
h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 22px 0 8px; font-weight: 600; }}
.lede {{ max-width: 70ch; color: var(--muted); margin: 10px 0 0; }}
.lede strong {{ color: var(--fg); }}
code, .t, .key {{ font-family: var(--mono); font-size: 13px; }}
.key {{ color: var(--muted); font-size: 13px; font-weight: 400; margin-left: 6px; }}
.slot {{ background: var(--slot-bg); color: var(--slot-fg); border-radius: 4px; padding: 0 4px; }}
nav.toc {{ position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg);
  display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 12px 0; border-bottom: 1px solid var(--line); margin-top: 20px; align-items: center; }}
body::before {{ content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); }}
nav.toc a {{ color: var(--accent); text-decoration: none; font-size: 14px; }}
nav.toc a:hover, nav.toc a:focus-visible {{ text-decoration: underline; }}
nav.toc label {{ margin-left: auto; font-size: 14px; color: var(--muted); display: flex; gap: 6px; align-items: center; }}
.summary {{ margin-top: 22px; display: grid; gap: 10px; padding: 0; list-style: none; }}
.summary li {{ display: grid; grid-template-columns: 82px 1fr; gap: 12px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 12px 14px; }}
.summary li > div {{ min-width: 0; }}
.summary p {{ margin: 4px 0 0; color: var(--muted); max-width: 80ch; }}
.badge {{ display: inline-block; font-size: 12px; font-weight: 600; border-radius: 999px; padding: 1px 9px; white-space: nowrap;
  align-self: start; text-align: center; }}
.b-problem {{ background: var(--p-bg); color: var(--p-fg); }}
.b-check {{ background: var(--c-bg); color: var(--c-fg); }}
.b-ok {{ background: var(--ok-bg); color: var(--ok-fg); }}
.cons {{ margin-top: 40px; scroll-margin-top: 70px; }}
.cons-head {{ display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; justify-content: space-between; }}
.counts {{ margin: 0; display: flex; gap: 6px; align-items: center; }}
.of {{ color: var(--muted); font-size: 13px; }}
.rule {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin-top: 12px; }}
.rule dl {{ display: grid; grid-template-columns: 170px 1fr; gap: 6px 14px; margin: 0; }}
.rule dt {{ color: var(--muted); font-size: 13px; }}
.rule dd {{ margin: 0; min-width: 0; }}
.examples {{ margin: 0; padding-left: 18px; display: grid; gap: 3px; }}
.ex {{ font-style: italic; }}
.ex-task {{ color: var(--muted); font-size: 12px; font-family: var(--mono); margin-left: 4px; }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }}
table {{ border-collapse: collapse; width: 100%; min-width: 640px; }}
th, td {{ text-align: left; vertical-align: top; padding: 7px 12px; border-bottom: 1px solid var(--line); }}
th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }}
tr:last-child td {{ border-bottom: 0; }}
td.sp {{ color: var(--muted); font-size: 13px; white-space: nowrap; }}
td.note {{ color: var(--muted); font-size: 14px; }}
tr.lv-problem td.t {{ box-shadow: inset 3px 0 0 var(--p-fg); }}
tr.lv-check td.t {{ box-shadow: inset 3px 0 0 var(--c-fg); }}
.words {{ margin-top: 40px; }}
.words p {{ max-width: 80ch; }}
footer {{ margin-top: 40px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); padding-top: 14px; }}
@media (max-width: 640px) {{
  .rule dl {{ grid-template-columns: 1fr; }}
  .summary li {{ grid-template-columns: 1fr; }}
  nav.toc label {{ margin-left: 0; }}
}}
</style>
<div class="wrap">
  <h1>Constraint Sentence Review</h1>
  <p class="lede">Every sentence template the filter is trained and tested with, next to what its rule actually measures.
    A template's <span class="slot">obj</span> / <span class="slot">region</span> / <span class="slot">material</span> slot is
    filled from the scene ("the red cup", "the green mat", "beads"). <strong>{tot["all"]} templates in 7 rules:
    {tot["problem"]} I think say the wrong thing, {tot["check"]} are vague or have small slips.</strong> Mark what you want
    changed; the bank is <code>envs/main/item_envs/language/paraphrases.yaml</code>.</p>
  <ul class="summary">{cross_html}</ul>
  <nav class="toc">{toc}<label for="only-flagged"><input type="checkbox" id="only-flagged"> Show only Problem / Check</label></nav>
  {"".join(sections)}
  <section class="words" id="words">
    <h2>Words that fill the slots</h2>
    <p><strong>Beads</strong>: {html.escape(mat_txt)}. <strong>Held-out words</strong> (test only, never in training):
      {html.escape(held_words)}.</p>
    <div class="tablewrap"><table><thead><tr><th>Region</th><th>Names</th><th>Synonyms</th></tr></thead>
      <tbody>{reg_rows}</tbody></table></div>
  </section>
  <footer>Generated 2026-10-20 from paraphrases.yaml and objects.yaml (after the 10-12 rename) by
    docs/lang_filter/template_review/build_page.py; my notes are in flags.py next to it. Example sentences come from
    recorded validation scenes, refreshed to today's object words.</footer>
</div>
<script>
(function () {{
  var box = document.getElementById("only-flagged");
  function apply() {{
    document.querySelectorAll("tr[data-flag]").forEach(function (tr) {{ tr.hidden = box.checked && tr.dataset.flag === "ok"; }});
  }}
  box.addEventListener("change", apply);
  apply();
}})();
</script>
'''
(HERE / "review.html").write_text(page)
print(f"wrote review.html: {tot}; stay problems {n_stay}, pour problems {n_pour}, clean/empty {n_clean}")
