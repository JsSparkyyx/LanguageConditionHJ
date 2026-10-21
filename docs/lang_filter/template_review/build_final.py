"""The constraint-sentence page after the 2026-10-20 rewrite (Yuxuan: five plain sentences per rule, four for
training and one held out for the OOD test; keep-on-the-table dropped; plain slot words). Reads the bank itself,
flags.RULES (what each rule measures) and examples_v2.json (real scenes, refreshed to today's words).

    python docs/lang_filter/template_review/build_final.py    # -> review.html (the published review page)
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
EX = json.loads((HERE / "examples_v2.json").read_text())
ORDER = ["avoid", "stay", "upright", "zone_exclude", "contain_particles", "exclude_particles"]
TITLE = {"avoid": "Don't touch", "stay": "Don't move", "upright": "Don't tip over", "zone_exclude": "Keep out of a mat",
         "contain_particles": "Don't spill", "exclude_particles": "Keep beads out of a bin or plate"}


def tpl(t):
    return re.sub(r"\{(obj|region|material)\}", lambda m: f'<span class="slot">{m.group(1)}</span>', html.escape(t))


cards = []
for c in ORDER:
    r = flags.RULES[c]
    train = "".join(f"<li>{tpl(t)}</li>" for t in BANK[c]["train"])
    held = "".join(f"<li>{tpl(t)}</li>" for t in BANK[c]["heldout"])
    exs = "".join(f'<li>{html.escape(s)} <span class="task">{html.escape(e["task"])}</span></li>'
                  for e in EX.get(c, [])[:3] for s in e["train"][:1] + [f"OOD: {h}" for h in e["heldout"][:1]])
    cards.append(f'''
<section class="card" id="{c}">
  <header><h2>{html.escape(TITLE[c])}</h2><code>{c}</code></header>
  <p class="tasks">{html.escape(r["tasks"])}</p>
  <p class="measures">{html.escape(r["measures"])}</p>
  <div class="cols">
    <div><h3>Training (4)</h3><ol>{train}</ol></div>
    <div><h3>OOD test (1)</h3><ol class="ood">{held}</ol></div>
  </div>
  <h3>In real scenes</h3><ul class="ex">{exs}</ul>
</section>''')

reg = OBJECTS.get("regions", {})
words = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(', '.join(v.get('names', [])))}</td>"
                f"<td>{html.escape(', '.join(v.get('synonyms', []) or []) or '-')}</td></tr>"
                for k, v in reg.items() if k in ("table_top", "zone", "bin", "plate"))
beads = OBJECTS["categories"]["beads"]
page = f'''<title>Constraint Sentence Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
/* Layout: one reading column; a card per named rule (what it measures, 4 training + 1 OOD sentence, real fill-ins). */
:root {{
  --bg: #f5f7f8; --panel: #ffffff; --fg: #1d2529; --muted: #5b6970; --line: #dce3e6; --accent: #1f6f73;
  --slot-bg: #e3f1f1; --slot-fg: #165457; --ood-bg: #fbf0d9; --ood-fg: #85580d;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12181b; --panel: #192125; --fg: #e3e9ec; --muted: #9aa8af; --line: #2b363b; --accent: #6cc3c7;
  --slot-bg: #1f3a3c; --slot-fg: #a6e0e2; --ood-bg: #3d2f12; --ood-fg: #ecc574; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #12181b; --panel: #192125; --fg: #e3e9ec; --muted: #9aa8af; --line: #2b363b; --accent: #6cc3c7;
  --slot-bg: #1f3a3c; --slot-fg: #a6e0e2; --ood-bg: #3d2f12; --ood-fg: #ecc574; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }}
.wrap {{ max-width: 960px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
h1, h2 {{ font-family: var(--display); font-weight: 600; margin: 0; text-wrap: balance; }}
h1 {{ font-size: 30px; }} h2 {{ font-size: 21px; }}
h3 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 14px 0 6px; font-weight: 600; }}
.lede {{ color: var(--muted); max-width: 72ch; margin: 10px 0 0; }}
.changes {{ margin: 18px 0 0; padding: 12px 16px 12px 34px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
.changes li + li {{ margin-top: 4px; }}
.slot {{ background: var(--slot-bg); color: var(--slot-fg); border-radius: 4px; padding: 0 4px; font-family: var(--mono); font-size: 13px; }}
.grid {{ display: grid; gap: 14px; margin-top: 22px; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px 16px; scroll-margin-top: 16px; }}
.card header {{ display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }}
.card header code {{ color: var(--muted); font-family: var(--mono); font-size: 13px; }}
.tasks {{ margin: 4px 0 0; color: var(--accent); font-size: 14px; }}
.measures {{ margin: 6px 0 0; max-width: 80ch; }}
.cols {{ display: grid; grid-template-columns: 2fr 1fr; gap: 6px 22px; }}
.cols > div {{ min-width: 0; }}
ol {{ margin: 0; padding-left: 20px; }}
ol.ood li {{ background: var(--ood-bg); color: var(--ood-fg); border-radius: 6px; padding: 1px 8px; list-style-position: inside; margin-left: -20px; }}
ul.ex {{ margin: 0; padding-left: 18px; font-style: italic; }}
.task {{ font-style: normal; color: var(--muted); font-family: var(--mono); font-size: 12px; margin-left: 4px; }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); margin-top: 10px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }}
tr:last-child td {{ border-bottom: 0; }}
footer {{ margin-top: 36px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); padding-top: 12px; }}
@media (max-width: 640px) {{ .cols {{ grid-template-columns: 1fr; }} }}
</style>
<div class="wrap">
  <h1>Constraint Sentence Review</h1>
  <p class="lede">The sentences the filter is trained and tested with, after the 2026-10-20 rewrite: five plain sentences
    for each named rule, four for training and one held out for the out-of-distribution test. A
    <span class="slot">obj</span> / <span class="slot">region</span> / <span class="slot">material</span> slot is filled from
    the scene ("the red block", "the green mat", "beads").</p>
  <ul class="changes">
    <li>211 wordy templates replaced by 30 plain ones (6 rules x 5).</li>
    <li>"Don't move" never says touch, bump or knock: in T1 the same object also has "don't touch".</li>
    <li>"Don't spill" never says "keep the beads in": in T2 and T6 they are poured out on purpose.</li>
    <li>Keep-on-the-table is no longer named (it is still measured, never used).</li>
    <li>Plain slot words: mat, bin, plate, table; beads are "beads" or "balls" ("little balls" only in the held-out word test).
      Old recordings are re-named to these words when they are loaded.</li>
  </ul>
  <div class="grid">{"".join(cards)}</div>
  <h2 style="margin-top:34px">Slot words</h2>
  <p class="lede">Beads: {html.escape(", ".join(beads["names"] + list(beads.get("synonyms") or [])))} ("little balls" is the
    held-out word). Object names are the 10-12 list (unchanged).</p>
  <div class="tablewrap"><table><thead><tr><th>Region</th><th>Names</th><th>Synonyms</th></tr></thead><tbody>{words}</tbody></table></div>
  <footer>Generated 2026-10-20 from envs/main/item_envs/language/paraphrases.yaml and configs/objects.yaml by
    docs/lang_filter/template_review/build_final.py. Example sentences come from recorded validation scenes, re-named
    with today's words.</footer>
</div>
'''
(HERE / "review.html").write_text(page)
print("wrote review.html")
