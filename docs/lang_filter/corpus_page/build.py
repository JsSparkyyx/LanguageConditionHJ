"""One page for the training tasks, the constraint sentences and the object words (Yuxuan, 2026-10-22).

Reads the bank (language/paraphrases.yaml), the corpus (configs/objects.yaml), the splits (configs/splits.yaml),
the rule descriptions (template_review/flags.py), the recorded examples (template_review/examples_v2.json) and the
object tiles cut from runs/lang/object_gallery (img/tiles.json), so the page matches the code.

    python docs/lang_filter/corpus_page/build.py    # -> index.html next to this file (images in img/)
"""

import html
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ENV = REPO / "envs/main/item_envs"
sys.path.insert(0, str(REPO / "docs/lang_filter/template_review"))
import flags  # noqa: E402

BANK = yaml.safe_load((ENV / "language/paraphrases.yaml").read_text())
OBJ = yaml.safe_load((ENV / "configs/objects.yaml").read_text())
SPLITS = yaml.safe_load((ENV / "configs/splits.yaml").read_text())
EX = json.loads((REPO / "docs/lang_filter/template_review/examples_v2.json").read_text())
TILES = json.loads((HERE / "img/tiles.json").read_text())
HELD = set(BANK["heldout_words"])
E = html.escape

ORDER = ["avoid", "stay", "upright", "zone_exclude", "contain_particles"]   # exclude_particles was T6 only
TITLE = {"avoid": "Don't touch", "stay": "Don't move", "upright": "Don't tip over", "zone_exclude": "Keep out of a mat",
         "contain_particles": "Don't spill", "exclude_particles": "Keep beads out of a bin or plate"}

# ------------------------------------------------------------------ tasks ----
# Plain-language summaries of envs/docs/item_envs/tasks_overview.md and the task configs (configs/tasks/T*.yaml).
TASKS = [
    dict(n=1, key="selective_avoid", name="Don't touch the named object",
         job="One of the five CrossSafe pick-and-place jobs (put the cup on the plate, burger and fries on the tray, "
             "stack two blocks, stack two bowls, bread into the basket). Two or three extra objects stand on the table, "
             "all of different kinds, drawn from the hand-picked pool; one or two stand right on the arm's path.",
         wrong="The arm, or what it is holding, brushes past or bumps one of the extra objects.",
         rules=["avoid", "stay"], per="each extra object",
         objects="The pool chosen on 2026-10-22: wine glass, milk box, apple, bottle (3 training models each, the "
                 "apple 1), coloured box, coloured tall cylinder. A layout with no extra object is skipped. The job's own objects (cup or bowl, "
                 "plate, burger, fries, blocks, bowls, bread, basket, tray) are never named, but a phrase must still "
                 "pick out one object among them.",
         inway="Objects on or near the arm's path, plus off-path objects within 10 cm of a second path.",
         h="Don't touch: the gap between the robot (and what it holds) and the object's real shape, minus 1 cm; "
           "1.0 at a 6 cm gap, 0 at 1 cm. Don't move: moved more than 3 cm (tilt counts as moving).",
         note="The one task where the rule is about contact, so a hand-made distance filter should match ours. "
              "Reported as the control task."),
    dict(n=2, key="granular_pour", name="Don't spill",
         job="Pour beads from a mug into a bowl.",
         wrong="The pour is too fast, too high or off to the side, so beads miss the bowl and land on the table.",
         rules=["contain_particles"], per="the beads (one rule)",
         objects="A mug (6 mug models), a bowl, up to 120 beads of 6 mm. The mug and bowl are never named.",
         inway="Only one rule, so the sentence does not change the answer here.",
         h="The worst bead's depth inside the column above the mug or the bowl, in cm. Once beads are out it counts "
           "them: one bead = -0.5, 20 or more = -1."),
    dict(n=3, key="push_zone", name="Push without harming the named thing",
         job="Push a coloured box across the table into a dark frame. Coloured mats and tall coloured cylinders "
             "lie on or beside the way.",
         wrong="The straight push slides the box over a mat, or tips or shoves a standing object.",
         rules=["zone_exclude", "upright", "stay"], per="each mat (keep the box out), each standing object (upright, stay)",
         objects="Pushed: a coloured box. Standing: coloured tall cylinders. Mats in red, green or blue.",
         inway="Mats and objects on the push path.",
         h="Mat: the box footprint's distance to the mat edge, 1.0 per 5 cm, negative once a corner is over it. "
           "Upright: (10 deg - tilt) / 10 deg. Don't move: (1.5 cm - moved) / 1.5 cm.",
         note="Keep-the-box-on-the-table is still measured but never named (no sentences)."),
    dict(n=5, key="tray_transport", name="Don't let the named block fall",
         job="Slide a tray carrying two standing blocks to a target frame: one tall and thin, one short.",
         wrong="A fast slide with a sharp stop tips the tall block over. The short block stays up.",
         rules=["upright"], per="each block",
         objects="A tray, a tall and a short block in red, green or blue (different colours).",
         inway="The tall block. The named block is the tall one in 75% of episodes, a short one in 25%.",
         h="(10 deg - tilt) / 10 deg: 1 standing straight, 0 at 10 deg, -1 at 20 deg."),
    dict(n=7, key="gap_place", name="Don't disturb the named block",
         job="Slide a wooden block into a narrow gap between two small standing blocks.",
         wrong="The slide is a bit off-centre, so the wooden block scrapes a neighbour and tips it or shoves it away.",
         rules=["upright", "stay"], per="each neighbour",
         objects="The carried wooden block, two neighbours (a slim pillar or a short wide block) and sometimes a "
                 "bystander block, in red, green and blue.",
         inway="The neighbour the slide scrapes (named in 60% of episodes; the other neighbour 25%, the bystander 15%).",
         h="(2 cm - moved) / 2 cm, with 10 deg of tilt counting like 2 cm. Upright: (10 deg - tilt) / 10 deg."),
]
RULE_TASKS = {c: [t for t in TASKS if c in t["rules"]] for c in ORDER}
USED = {t["key"] for t in TASKS}
# flags.RULES describes the seven-task set; the spill rule's text without T4 / T6
MEASURES = {"contain_particles": "No bead may land on the table. The beads are poured from the mug into the bowl on "
                                 "purpose: a bead in either container is fine, a bead on the table is the violation."}


def chip(c, small=False):
    return f'<a class="chip{" sm" if small else ""}" href="#rule-{c}">{E(TITLE[c])}</a>'


def task_card(t):
    rules = "".join(chip(c) for c in t["rules"])
    note = f'<p class="note">{E(t["note"])}</p>' if t.get("note") else ""
    return f'''
<article class="task" id="task-{t["key"]}">
  <figure><img src="img/scene_{t["key"]}.jpg" alt="Task {t["n"]} scene: {E(t["key"])}" loading="lazy" width="640" height="460"></figure>
  <div class="tbody">
    <header><span class="tnum">T{t["n"]}</span><h3>{E(t["name"])}</h3><code>{t["key"]}</code></header>
    <dl>
      <dt>Robot's job</dt><dd>{E(t["job"])}</dd>
      <dt>What goes wrong</dt><dd>{E(t["wrong"])}</dd>
      <dt>Named rules</dt><dd><span class="chips">{rules}</span> <span class="per">per {E(t["per"])}</span></dd>
      <dt>Objects</dt><dd>{E(t["objects"])}</dd>
      <dt>In the way</dt><dd>{E(t["inway"])}</dd>
      <dt>Safety number</dt><dd>{E(t["h"])}</dd>
    </dl>{note}
  </div>
</article>'''


# -------------------------------------------------------------- sentences ----
def tpl(s):
    return re.sub(r"\{(obj|region|material)\}", lambda m: f'<span class="slot">{m.group(1)}</span>', E(s))


def rule_card(c):
    measures = MEASURES.get(c, flags.RULES[c]["measures"])
    train = "".join(f"<li>{tpl(s)}</li>" for s in BANK[c]["train"])
    held = "".join(f"<li>{tpl(s)}</li>" for s in BANK[c]["heldout"])
    tasks = " ".join(f'<a class="tref" href="#task-{t["key"]}">T{t["n"]}</a>' for t in RULE_TASKS[c])
    seen, exs = set(), []
    ex_all = [e for e in EX.get(c, []) if e["task"] in USED]
    for e in ex_all:   # one recorded scene per task first, then more
        if e["task"] in seen and len(exs) < 3:
            continue
        seen.add(e["task"])
        exs.append(e)
    for e in ex_all:
        if len(exs) >= 3:
            break
        if e not in exs:
            exs.append(e)
    rows = "".join(f'<li><span class="ex">{E(e["train"][0])}</span><span class="ex">{E(e["train"][1])}</span>'
                   f'<span class="ex ood">{E(e["heldout"][0])}</span><span class="etask">{E(e["task"])}</span></li>'
                   for e in exs[:3])
    return f'''
<article class="rule" id="rule-{c}">
  <header><h3>{E(TITLE[c])}</h3><code>{c}</code><span class="trefs">{tasks}</span></header>
  <p class="measures">{E(measures)}</p>
  <div class="cols">
    <div><h4>Training (4)</h4><ol>{train}</ol></div>
    <div><h4>OOD test (1)</h4><ol class="oodl">{held}</ol></div>
  </div>
  <h4>Filled in, from recorded scenes <span class="h4note">two training fills and the OOD fill of one rule</span></h4>
  <ul class="exs">{rows}</ul>
</article>'''


# ---------------------------------------------------------------- objects ----
CATS = OBJ["categories"]
ASSETS = OBJ["assets"]
POOL = OBJ["t1_pool"]


def words(cat):
    c = CATS[cat]
    names = list(c.get("names") or [])
    syn = [s for s in (c.get("synonyms") or []) if s not in HELD]
    held = [s for s in (c.get("synonyms") or []) + names if s in HELD]
    return names, syn, held


# (category, where it is named, models) for every object a rule can name
NAMED = [
    ("wine_glass", "T1", "088_wineglass ids 0, 1, 2; held out: id 3", ""),
    ("milk_box", "T1", "038_milk-box ids 0, 1, 2; held out: id 3", ""),
    ("apple", "T1", "035_apple id 0; held out: id 1", ""),
    ("bottle", "T1", "001_bottle ids 0, 4, 13; held out: id 16", ""),
    ("box", "T1, T3", "primitive box, 4-8 cm sides, coloured", "always gets a colour word when needed"),
    ("cylinder", "T1, T3", "primitive cylinder, 10-18 cm tall, coloured", ""),
    ("block", "T5, T7", "primitive blocks, coloured, tall / short", "\"short\" / \"tall\" added when it picks one out"),
]
NEVER = [("cup", "T1 job object (put the cup on the plate)"), ("wooden_block", "T1 job object (stack two blocks)"),
         ("bowl", "T1 job object, T2 bowl"), ("pour_mug", "T2 pouring mug (039_mug)"),
         ("tray", "T1 job object, T5 tray"),
         ("hamburg", "T1 job object"), ("frenchfries", "T1 job object")]


def word_row(cat, where, models, note):
    names, syn, held = words(cat)
    n = ", ".join(E(x) for x in names)
    s = ", ".join(E(x) for x in syn) or '<span class="dim">none</span>'
    h = ", ".join(f'<span class="oodw">{E(x)}</span>' for x in held) or '<span class="dim">none</span>'
    col = ' <span class="tag">+ colour</span>' if CATS[cat].get("colored") else ""
    return (f"<tr><td><b>{n}</b>{col}</td><td>{s}</td><td>{h}</td><td>{E(where)}</td>"
            f"<td class=\"small\">{E(models)}{('<br><span class=dim>' + E(note) + '</span>') if note else ''}</td></tr>")


TILE_TITLE = {"088_wineglass": "wine glass (088_wineglass)", "038_milk-box": "milk box (038_milk-box)",
              "043_book": "book (043_book)", "092_notebook": "book (092_notebook)", "035_apple": "apple (035_apple)",
              "091_kettle": "kettle (091_kettle)", "001_bottle": "bottle (001_bottle)",
              "039_mug": "mug (039_mug), T2 pouring mug"}
HELD_IDS = {a: set((OBJ["assets"][a].get("heldout_ids") or [])) for a in TILE_TITLE}
ID_NOTE = {("088_wineglass", 0): "shrunk to 87%", ("038_milk-box", 1): "shrunk to 88%",
           ("038_milk-box", 2): "shrunk to 90%"}


def gallery(asset):
    tiles = "".join(
        f'<figure class="tile{" held" if i in HELD_IDS[asset] else ""}"><img src="img/{f}" alt="{E(asset)} id {i}" loading="lazy">'
        f'<figcaption>id {i}{(" · " + E(ID_NOTE[(asset, i)])) if (asset, i) in ID_NOTE else ""}'
        f'{"<b> · held out</b>" if i in HELD_IDS[asset] else ""}</figcaption></figure>'
        for i, f in TILES[asset])
    return f'<div class="gal"><h4>{E(TILE_TITLE[asset])}</h4><div class="tiles">{tiles}</div></div>'


def swatch(c, held=False):
    r, g, b = (int(round(255 * v)) for v in OBJ["color_rgb"][c])
    return f'<span class="sw{" ood" if held else ""}"><i style="background:rgb({r},{g},{b})"></i>{E(c)}</span>'


REG = OBJ["regions"]


def reg_row(key, where, colours, note):
    r = REG[key]
    n = ", ".join(E(x) for x in r["names"])
    return (f"<tr><td><b>{n}</b>{' <span class=tag>+ colour</span>' if r.get('colored') else ''}</td>"
            f"<td>{E(where)}</td><td>{colours}</td><td class=\"small\">{E(note)}</td></tr>")


def build():
    beads = CATS["beads"]
    bead_train = [w for w in beads["names"] + beads["synonyms"] if w not in HELD]
    bead_held = [w for w in beads["names"] + beads["synonyms"] if w in HELD]
    train_c = OBJ["colors"]["train"]
    held_c = OBJ["colors"]["heldout"]
    n_train = sum(len(BANK[c]["train"]) for c in ORDER)
    n_held = sum(len(BANK[c]["heldout"]) for c in ORDER)
    pool_tiles = "".join(gallery(a) for a in ["088_wineglass", "038_milk-box", "035_apple", "001_bottle"])
    prims = (f'<div class="gal"><h4>coloured box and tall cylinder (primitives)</h4><p class="small">Box sides 4-8 cm, '
             f'5-15 cm tall; cylinder radius 2-3.5 cm, 10-18 cm tall. Colour drawn from the training colours: '
             f'{"".join(swatch(c) for c in train_c)}</p></div>')
    named_rows = "".join(word_row(*x) for x in NAMED)
    never_rows = "".join(f"<tr><td><b>{', '.join(E(x) for x in CATS[c]['names'])}</b></td>"
                         f"<td>{', '.join(E(x) for x in CATS[c].get('synonyms') or []) or '<span class=dim>none</span>'}</td>"
                         f"<td>{E(w)}</td></tr>" for c, w in NEVER)
    reg_rows = (reg_row("zone", "T3", "red, green, blue", "keep the box out of it")
                + reg_row("table_top", "T3", "-", "keep-the-box-on-the-table is measured but never named; "
                                                   "\"on the table\" in the spill sentences is fixed text"))
    ood_templates = "".join(f"<li>{tpl(BANK[c]['heldout'][0])} <span class=\"dim\">({E(TITLE[c])})</span></li>"
                            for c in ORDER)
    held_words = "".join(f'<span class="oodw">{E(w)}</span>' for w in BANK["heldout_words"])
    return f'''<title>Named-Object Task Corpus</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column of reference sections (tasks, sentences, words, held-out tests); task cards put the
   scene beside its facts; object tiles sit in wrapping grids. Same palette and faces as the sentence review page. */
:root {{
  --bg: #f4f6f7; --panel: #ffffff; --fg: #1c2427; --muted: #5a676d; --line: #dbe2e5; --accent: #1f6f73;
  --accent-soft: #e2f0f0; --slot-bg: #e3f1f1; --slot-fg: #165457; --ood-bg: #fbefd6; --ood-fg: #835509;
  --tile-bg: #ffffff; --tile-fg: #4b585e; --tile-held: #835509;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #11171a; --panel: #182024; --fg: #e2e8eb; --muted: #98a6ad; --line: #2a353a; --accent: #6cc3c7;
  --accent-soft: #1d3436; --slot-bg: #1f3a3c; --slot-fg: #a6e0e2; --ood-bg: #3b2d11; --ood-fg: #ecc574;
  --tile-bg: #eef1f2; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #11171a; --panel: #182024; --fg: #e2e8eb; --muted: #98a6ad; --line: #2a353a; --accent: #6cc3c7;
  --accent-soft: #1d3436; --slot-bg: #1f3a3c; --slot-fg: #a6e0e2; --ood-bg: #3b2d11; --ood-fg: #ecc574;
  --tile-bg: #eef1f2; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding-block: 30px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
h1, h2, h3 {{ font-family: var(--display); font-weight: 600; margin: 0; text-wrap: balance; }}
h1 {{ font-size: 32px; letter-spacing: -.01em; }}
h2 {{ font-size: 24px; margin-top: 46px; padding-top: 14px; border-top: 2px solid var(--fg); scroll-margin-top: 12px; }}
h3 {{ font-size: 19px; }}
h4 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 14px 0 6px; font-weight: 600; }}
.h4note {{ text-transform: none; letter-spacing: 0; font-weight: 400; margin-left: 6px; }}
p {{ margin: 0; }}
.lede {{ color: var(--muted); max-width: 74ch; margin-top: 10px; }}
.facts {{ display: flex; flex-wrap: wrap; gap: 6px 22px; margin-top: 16px; font-size: 14px; }}
.facts b {{ font-family: var(--display); font-size: 20px; font-weight: 600; margin-right: 5px; font-variant-numeric: tabular-nums; }}
nav {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 18px; }}
nav a {{ color: var(--accent); text-decoration: none; border: 1px solid var(--line); background: var(--panel); border-radius: 999px;
  padding: 3px 12px; font-size: 14px; }}
nav a:hover, nav a:focus-visible {{ border-color: var(--accent); }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.secintro {{ max-width: 76ch; margin-top: 10px; }}
.secintro + .secintro {{ margin-top: 8px; }}
code {{ font-family: var(--mono); font-size: 13px; color: var(--muted); }}
.slot {{ background: var(--slot-bg); color: var(--slot-fg); border-radius: 4px; padding: 0 4px; font-family: var(--mono); font-size: 13px; }}
.stack {{ display: grid; gap: 14px; margin-top: 18px; }}
.task {{ display: grid; grid-template-columns: minmax(0, 330px) minmax(0, 1fr); gap: 18px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 10px; padding: 14px; scroll-margin-top: 12px; }}
.task figure {{ margin: 0; }}
.task img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
.tbody {{ min-width: 0; }}
.task header, .rule header {{ display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; }}
.tnum {{ font-family: var(--display); font-weight: 600; font-size: 19px; color: var(--accent); }}
dl {{ display: grid; grid-template-columns: 9.5em minmax(0, 1fr); gap: 5px 14px; margin: 10px 0 0; font-size: 14px; }}
dt {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; padding-top: 2px; }}
dd {{ margin: 0; min-width: 0; }}
.note {{ margin-top: 10px; font-size: 14px; color: var(--muted); border-left: 3px solid var(--line); padding-left: 10px; }}
.chips {{ display: inline-flex; flex-wrap: wrap; gap: 5px; vertical-align: middle; }}
.chip {{ display: inline-block; background: var(--accent-soft); color: var(--accent); border-radius: 999px; padding: 0 9px;
  font-size: 13px; text-decoration: none; font-weight: 500; }}
.per {{ color: var(--muted); }}
.rule {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px 16px; scroll-margin-top: 12px; }}
.trefs {{ display: inline-flex; gap: 5px; margin-left: auto; }}
.tref {{ font-family: var(--mono); font-size: 12px; color: var(--accent); text-decoration: none; border: 1px solid var(--line);
  border-radius: 4px; padding: 0 5px; }}
.measures {{ margin-top: 6px; max-width: 82ch; font-size: 14px; }}
.cols {{ display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 4px 24px; }}
.cols > div {{ min-width: 0; }}
ol {{ margin: 0; padding-left: 22px; }}
ol li + li {{ margin-top: 2px; }}
ol.oodl {{ list-style: none; padding-left: 0; }}
ol.oodl li {{ background: var(--ood-bg); color: var(--ood-fg); border-radius: 6px; padding: 1px 8px; }}
ul.exs {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }}
ul.exs li {{ display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: baseline; font-size: 14px; }}
.ex {{ font-style: italic; border: 1px solid var(--line); border-radius: 6px; padding: 0 7px; }}
.ex.ood {{ background: var(--ood-bg); color: var(--ood-fg); border-color: transparent; }}
.task, .rule {{ box-sizing: border-box; }}
.etask {{ font-style: normal; color: var(--muted); font-family: var(--mono); font-size: 12px; background: none; border: 0;
  padding: 0; display: inline; }}
.forms {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
.forms span {{ font-family: var(--mono); font-size: 13px; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 1px 8px; }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); margin-top: 14px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
.small {{ font-size: 13px; }}
.dim {{ color: var(--muted); }}
.tag {{ font-size: 11px; color: var(--muted); border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; white-space: nowrap; }}
.oodw {{ display: inline-block; background: var(--ood-bg); color: var(--ood-fg); border-radius: 4px; padding: 0 6px; margin: 1px 2px 1px 0; }}
.gals {{ display: grid; gap: 18px; margin-top: 14px; }}
.gal h4 {{ margin-top: 0; text-transform: none; letter-spacing: 0; font-size: 14px; color: var(--fg); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)); gap: 8px; }}
.tile {{ margin: 0; background: var(--tile-bg); border: 1px solid var(--line); border-radius: 8px; padding: 6px; display: flex;
  flex-direction: column; align-items: center; }}
.tile img {{ width: 100%; height: 118px; object-fit: contain; object-position: center bottom; }}
.tile figcaption {{ font-family: var(--mono); font-size: 11.5px; color: var(--tile-fg); margin-top: 2px; }}
.tile.held {{ border: 2px solid var(--ood-fg); }}
.tile.held b {{ color: var(--tile-held); font-weight: 600; }}
.sw {{ display: inline-flex; align-items: center; gap: 5px; margin: 2px 10px 2px 0; font-size: 13px; }}
.sw i {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid var(--line); display: inline-block; }}
.sw.ood {{ color: var(--ood-fg); }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 14px; margin-top: 14px; }}
.box {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; min-width: 0; }}
.box h3 {{ font-size: 17px; }}
.box p, .box ul, .box ol {{ margin-top: 8px; font-size: 14px; }}
.box ul {{ padding-left: 18px; }}
footer {{ margin-top: 40px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); padding-top: 12px; }}
@media (max-width: 760px) {{
  .task {{ grid-template-columns: 1fr; }}
  .cols {{ grid-template-columns: 1fr; }}
  dl {{ grid-template-columns: 1fr; gap: 0 0; }}
  dd {{ margin-bottom: 6px; }}
  .trefs {{ margin-left: 0; }}
}}
</style>
<div class="wrap">
  <h1>Named-Object Task Corpus</h1>
  <p class="lede">What the language-conditioned safety filter is trained on: the five table-top tasks, the sentences that
    name a rule, and the words used for objects, regions and beads, with the parts held out for the out-of-distribution
    (OOD) tests. Everything below is read from the code (2026-10-20 sentence rewrite, five tasks from 2026-10-22).</p>
  <div class="facts"><span><b>{len(TASKS)}</b>tasks</span><span><b>{len(ORDER)}</b>named rules</span>
    <span><b>{n_train + n_held}</b>sentences ({n_train} training, {n_held} OOD)</span>
    <span><b>6</b>object types in the T1 pool</span><span><b>{len(HELD)}</b>held-out words</span></div>
  <nav><a href="#tasks">Tasks</a><a href="#sentences">Sentences</a><a href="#objects">Objects</a>
    <a href="#regions">Regions, beads, colours</a><a href="#heldout">Held-out tests</a></nav>

  <h2 id="tasks">Training tasks</h2>
  <p class="secintro">Each task has a careless nominal controller (too fast, a bit off-aim, tilted) that sometimes harms
    an item. A sentence names one rule; the filter must protect that item and leave the others alone. Every rule gives a
    safety number from -1 to 1: 1 = comfortably safe, 0 = the moment of harm, below 0 = harmed.</p>
  <p class="secintro">Robot: UR5 with the WSG gripper (all current training data). Five of the seven built tasks are
    used from 2026-10-22: T4 (carry a bowl of beads) and T6 (keep beads out of a bin or plate) were dropped, so the task
    numbers skip 4 and 6. The T1-only run trains on task 1 alone.
    "In the way" is decided by the scene layout and gives each named rule its harm or benign label for evaluation.</p>
  <div class="stack">{"".join(task_card(t) for t in TASKS)}</div>

  <h2 id="sentences">Constraint sentences</h2>
  <p class="secintro">Five plain sentences per rule: four for training, one held out for the OOD test. A slot is filled
    from the scene: <span class="slot">obj</span> and <span class="slot">region</span> with a phrase that picks out
    exactly one thing in that scene, <span class="slot">material</span> with a bead word. No numbers ever appear in a
    sentence, and the model gets no rule or task label: the sentence is the only description of the rule.</p>
  <p class="secintro">Phrase forms for an object, used only when no other object in the scene answers to them:</p>
  <div class="forms"><span>the noun</span><span>the colour noun</span><span>the size noun</span><span>the size colour noun</span></div>
  <p class="secintro">At training time every sample draws one rule of its episode and one training sentence. Batches
    balance harmful and benign rules, and half of each batch pairs the same moment with a harmful and a benign rule.
    Set and pair rules ("don't touch any cup") and keep-on-the-table are recorded but never named.</p>
  <div class="stack">{"".join(rule_card(c) for c in ORDER)}</div>

  <h2 id="objects">Object corpus</h2>
  <p class="secintro">The words a sentence may use for each object a rule can name. Names and training synonyms are used
    in training sentences; held-out words appear only in the held-out word test. A colour word is added for coloured
    objects (primitives, mats) when it is needed to pick one out.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Name</th><th>Training synonyms</th><th>Held-out word</th><th>Named in</th><th>Models</th></tr></thead>
    <tbody>{named_rows}</tbody></table></div>
  <h3 style="margin-top:22px">Objects that are never named</h3>
  <p class="secintro">They stand in the scenes and have names, so a phrase like "the cup" is rejected when one of them
    would also answer to it.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Name</th><th>Synonyms</th><th>Where</th></tr></thead><tbody>{never_rows}</tbody></table></div>
  <h3 style="margin-top:22px">The models</h3>
  <p class="secintro">Every model the tasks draw (chosen by hand on 2026-10-22). Each object type has three training
    models and one held-out model (the apple one and one); a held-out model, marked, never appears in a training scene.
    T1 draws an object type uniformly, at most one of each type per scene, and never one of the job's own types. The
    three widest models (wine glass 0, milk box 1 and 2) are shrunk by 10-13% so that they fit the 6 cm on-path limit
    (the sizes printed on the renders are the original ones). The
    book and the kettle were removed on 2026-10-22: they never fit on or near the arm's path, so they were never at
    risk. The objects used before 2026-10-22 (cups, cans, pill bottles, tea and coffee boxes, wooden blocks, drink
    bottles) are no longer drawn.</p>
  <div class="gals">{pool_tiles}{prims}{gallery("039_mug")}</div>

  <h2 id="regions">Regions, beads and colours</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Region</th><th>Task</th><th>Colours</th><th>Rule</th></tr></thead><tbody>{reg_rows}</tbody></table></div>
  <div class="grid2">
    <div class="box"><h3>Beads</h3>
      <p>Training words: {", ".join(f"<b>{E(w)}</b>" for w in bead_train)}. Held-out word:
        {", ".join(f'<span class="oodw">{E(w)}</span>' for w in bead_held)}.</p>
      <p>Spheres of 6 mm (SAPIEN has no fluids). Test sentences never say rice, beans or grains.</p></div>
    <div class="box"><h3>Colours</h3>
      <p>Training: {"".join(swatch(c) for c in train_c)}</p>
      <p>Held out: {"".join(swatch(c, True) for c in held_c)}</p>
      <p class="dim">Training scenes use only the training colours (each task uses a subset).</p></div>
  </div>

  <h2 id="heldout">Held-out tests</h2>
  <div class="grid2">
    <div class="box"><h3>Held-out sentence</h3><p>The OOD sentence of each rule. Training never uses it.</p>
      <ol>{ood_templates}</ol></div>
    <div class="box"><h3>Held-out word</h3><p>Nouns never used in training. The test fills the slot with these
      only.</p><p>{held_words}</p></div>
    <div class="box"><h3>Held-out object</h3><p>Objects that never appear in a training scene.</p>
      <ul><li>T1: one held-out model per object type (marked in the renders above)</li>
        <li>all tasks: anything purple or orange</li></ul>
      <p class="dim">The OOD scenes for T1 come from the held-out pool: only held-out models, and the box and cylinder
        in held-out colours.</p></div>
  </div>

  <footer>Built by docs/lang_filter/corpus_page/build.py from envs/main/item_envs/language/paraphrases.yaml,
    configs/objects.yaml, configs/splits.yaml and configs/tasks/, the rule notes in
    docs/lang_filter/template_review/flags.py and recorded validation scenes (examples_v2.json, re-named with today's
    words). Renders from runs/lang/object_gallery.</footer>
</div>
'''


if __name__ == "__main__":
    (HERE / "index.html").write_text(build())
    imgs = sorted(p.name for p in (HERE / "img").glob("*.jpg"))
    print("wrote index.html;", len(imgs), "images")
