#!/usr/bin/env python
"""Record presentation videos: loo (filtered) + nominal on IDENTICAL scenes,
picked so the NOMINAL collides while the filter stays clean when possible.

Candidates come from presentation_candidates.json (ranked rows harvested
from the loo_* eval_full jsonls — set_global_seeds(seed) +
RolloutController(cfg_ep, seed) reproduces the scene). Per combo we first
record the LOO video and verify the REPLAYED outcome (eval rows are only a
hint — GPU nondeterminism across machines can flip contact); rejected
replays are deleted and the next candidate is tried, up to --max-trials.
The nominal is then recorded on the same seed; scenes where the nominal
shows no collision (no contact AND force < 30 N) fall to the next
candidate, so the kept pair has nominal-collides / loo-clean whenever a
qualifying scene exists.

Panel (320x480, supersampled 2x for anti-aliasing; all math rendered with
matplotlib mathtext, no unit clutter):
  $t$ / $h(z_t)$ (min h) / $Q(z_t, \\pi_{\\phi}^{\\mathrm{safe}}(z_t))$ /
  $F$ (max)
then a legend block (its own lines above the plot — never overlaps the
traces): blue $h(z_t)$, and a right brace grouping black "nominal actor" /
red "safe actor" pointing to $Q(z_t, \\pi_{\\phi}^{\\mathrm{safe}}(z_t))$
— both colors are the SAME quantity, the color encodes the controller.
Then a full-episode plot in the style of the AnySafe figure:
  x = time (t) with an arrow axis, prominent dashed zero line,
  blue curve = failure function $h(z_t)$,
  Q curve = $Q(z_t, \\pi_{\\phi}^{\\mathrm{safe}}(z_t))$ EVERY tick
  (flt.q_actor, so it never stalls when the arm idles), fragmented black
  (nominal actor) / red (safe actor).
Frames are rendered one tick late so plot and readouts always reflect the
frame's own tick. No bbox, no force in the plot, no Q anywhere for the
nominal pass.

Run on a GPU node (login nodes have no CUDA):

    python tests/record_presentation.py \
        --candidates $TMP/cehj/presentation_candidates.json \
        --out $TMP/cehj/videos_presentation

    # smoke: one combo only
    python tests/record_presentation.py --candidates ... --out ... \
        --only "stack_blocks_two/franka-panda"

    # layout check (CPU only, safe on the login node): one synthetic panel
    python tests/record_presentation.py --preview-frame /tmp/panel.png \
        --preview-src <any existing presentation mp4>

Output: <out>/<run>_<emb>_<task>_ep<NNN>.mp4 (filtered) and
<out>/nominal_<emb>_<task>_ep<NNN>.mp4 (nominal, same seed), plus
<out>/selections_used.json (chosen candidates + nominal outcomes).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "tests"

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path("/work/nvme/bifq/project/tmp/cehj/runs")
FORCE_VIOL_N = 30.0          # same collision definition as the eval

BLUE = (37, 99, 235)         # h(z_t)
BLACK = (20, 20, 20)         # Q fragments on nominal-actor ticks
RED = (200, 30, 30)          # Q fragments on safe-actor ticks / warnings
GRAY = (110, 110, 110)

Q_MATH = r"$Q(z_t, \pi_{\phi}^{\mathrm{safe}}(z_t))$"
H_MATH = r"$h(z_t)$"


def _load_font(size):
    from PIL import ImageFont

    candidates = []
    try:
        import matplotlib

        candidates.append(Path(matplotlib.get_data_path())
                          / "fonts/ttf/DejaVuSans.ttf")
    except Exception:
        pass
    candidates += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for c in candidates:
        if c.exists():
            return ImageFont.truetype(str(c), size)
    return ImageFont.load_default()


_MATH_CACHE: dict = {}


def render_math(s: str, size: float, color, max_w=None, cache=False):
    """Render a string with $...$ mathtext segments to a PIL RGBA image.

    size = target em height in pixels of the DESTINATION image (already
    supersampled by the caller); dpi=72 makes 1 pt == 1 px. Shrinks the
    fontsize until the rendered width fits max_w (same units) if given.
    Static strings should pass cache=True; the cache is capped.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    key = (s, size, color, max_w)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    fs = float(size)
    width_in = 16.0
    img = None
    for _ in range(4):
        fig = Figure(figsize=(width_in, 2.0), dpi=72)
        fig.patch.set_alpha(0.0)
        FigureCanvasAgg(fig)
        txt = fig.text(0.01, 0.35, s, fontsize=fs, va="baseline",
                       color=tuple(c / 255.0 for c in color))
        fig.canvas.draw()
        bbox = txt.get_window_extent()
        if max_w is not None and bbox.width > max_w and fs > 6:
            fs *= 0.96 * max_w / bbox.width
            continue
        buf = np.asarray(fig.canvas.buffer_rgba())
        Hb, Wb = buf.shape[:2]
        if bbox.x1 > Wb - 2:                 # string overflows the canvas
            width_in *= 1.7
            continue
        x0 = max(int(np.floor(bbox.x0)) - 2, 0)
        x1 = min(int(np.ceil(bbox.x1)) + 2, Wb)
        y0 = max(Hb - int(np.ceil(bbox.y1)) - 2, 0)
        y1 = min(Hb - int(np.floor(bbox.y0)) + 2, Hb)
        img = Image.fromarray(buf[y0:y1, x0:x1].copy(), "RGBA")
        break
    if img is None:                          # pragma: no cover
        img = Image.new("RGBA", (10, 10))
    if cache:
        if len(_MATH_CACHE) > 400:
            _MATH_CACHE.clear()
        _MATH_CACHE[key] = img
    return img


class CleanPanelVideoWriter:
    """Observer frame + 320px clean white panel -> 960x480 mp4.

    Panel is rendered at 2x and downscaled (anti-aliased text/lines); all
    math is matplotlib mathtext. `pq` is a live reference to
    PresentationRollout.pq: per-tick (t, Q, safe) with
    Q = Q(z_t, pi_safe(z_t)) at EVERY tick. Frames are rendered ONE TICK
    LATE (pending buffer): the pq entry for a tick is computed right after
    its capture, so deferring one frame makes plot/readouts exact for the
    frame's own tick.
    """

    SS = 2                     # supersample factor

    def __init__(self, path: Path | None, fps: float = 25.0,
                 h_scale: float = 20.0, show_q: bool = True):
        self.cm = 100.0 / h_scale
        self.show_q = show_q
        self.writer = None
        if path is not None:
            import imageio.v2 as imageio

            self.writer = imageio.get_writer(
                str(path), fps=fps, codec="libx264", format="FFMPEG",
                pixelformat="yuv420p", macro_block_size=1,
            )
        self.hist_t, self.hist_h = [], []
        self.pq = []
        self._max_force = 0.0
        self._min_h = float("inf")
        self._pending = None   # (frame_rgb, t, h, force, touch, iv_ticks)
        self.fonts = [_load_font(s * self.SS) for s in (15, 13, 11)]

    def _fit_text(self, d, x, y, text, color, max_w):
        """Draw plain text with the largest font that fits max_w."""
        for f in self.fonts:
            if d.textlength(text, font=f) <= max_w:
                d.text((x, y), text, fill=color, font=f)
                return
        d.text((x, y), text, fill=color, font=self.fonts[-1])

    def _dashed_hline(self, d, x0, x1, y, color, width, dash=8, gap=6):
        x = x0
        while x < x1:
            d.line([x, y, min(x + dash, x1), y], fill=color, width=width)
            x += dash + gap

    def _panel(self, t, h, force, touch, iv_ticks):
        from PIL import Image, ImageDraw

        S = self.SS
        cm = self.cm
        self._min_h = min(self._min_h, h)
        self.hist_t.append(t)
        self.hist_h.append(h * cm)
        pq = [e for e in self.pq if e[0] <= t + 1e-9]
        safe_now = bool(pq[-1][2]) if pq else False

        W, H = 320 * S, 480 * S
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)

        def X(v):
            return int(round(v * S))

        def put(x, y, s, size, color, max_w=None, cache=False):
            mi = render_math(s, size * S, color,
                             None if max_w is None else max_w * S,
                             cache=cache)
            img.paste(mi, (X(x), X(y)), mi)

        # ---- readouts ----
        y = 12
        put(12, y, f"$t = {t:.2f}\\,\\mathrm{{s}}$", 15, BLACK)
        y += 26
        min_h_cm = self._min_h * cm
        put(12, y,
            f"$h(z_t) = {h * cm:+.1f}$ (min ${min_h_cm:+.1f}$)",
            15, RED if min_h_cm < 0 else BLACK)
        y += 26
        if self.show_q:
            if pq:
                put(12, y,
                    f"{Q_MATH[:-1]} = {pq[-1][1] * cm:+.1f}$",
                    15, RED if safe_now else BLACK, max_w=296)
            else:
                put(12, y, Q_MATH[:-1] + " = -$", 15, BLACK, max_w=296)
            y += 26
        self._max_force = max(self._max_force, force)
        hit = touch or force >= 20.0
        put(12, y,
            f"$F = {force:.1f}\\,\\mathrm{{N}}$ "
            f"(max ${self._max_force:.1f}$)",
            15, RED if hit else BLACK)
        y += 26

        # ---- legend block (own lines above the plot — never overlaps the
        # traces). Row 1: blue h(z_t) + black nominal actor; row 2: red
        # safe actor. A right brace groups the two actor entries and
        # points to Q(z_t, pi_safe(z_t)): both curve colors are the SAME
        # quantity, the color says which controller drove the tick.
        y += 22
        row_h = 18
        f_leg = self.fonts[2]
        h_mi = render_math(H_MATH, 11 * S, BLUE, cache=True)
        h_part = 22 * S + h_mi.width
        y1, y2 = X(y), X(y + row_h)
        if self.show_q:
            w_nom = int(d.textlength("nominal actor", font=f_leg))
            w_safe = int(d.textlength("safe actor", font=f_leg))
            a_part = 22 * S + max(w_nom, w_safe)
            brace = render_math(r"$\}$", 26 * S, BLACK, cache=True)
            q_mi = render_math(Q_MATH, 11 * S, BLACK, cache=True)
            total = (h_part + 20 * S + a_part + 4 * S + brace.width
                     + 2 * S + q_mi.width)
        else:
            total = h_part
        x0 = (320 * S - total) // 2
        d.line([x0, y1 + 5 * S, x0 + 18 * S, y1 + 5 * S], fill=BLUE,
               width=2 * S)
        img.paste(h_mi, (x0 + 22 * S, y1), h_mi)
        if self.show_q:
            xa = x0 + h_part + 20 * S
            d.line([xa, y1 + 5 * S, xa + 18 * S, y1 + 5 * S], fill=BLACK,
                   width=2 * S)
            d.text((xa + 22 * S, y1), "nominal actor", fill=BLACK,
                   font=f_leg)
            d.line([xa, y2 + 5 * S, xa + 18 * S, y2 + 5 * S], fill=RED,
                   width=2 * S)
            d.text((xa + 22 * S, y2), "safe actor", fill=RED, font=f_leg)
            xb = xa + a_part + 4 * S
            span = X(row_h) + 11 * S
            img.paste(brace, (xb, y1 + (span - brace.height) // 2), brace)
            img.paste(q_mi, (xb + brace.width + 2 * S,
                             y1 + (span - q_mi.height) // 2), q_mi)
            y += row_h

        # ---- full-episode plot ----
        px0, py0 = X(30), X(y + 36)
        pw, ph = X(266), X(480 - (y + 36) - 42)
        if len(self.hist_h) >= 2:
            vals = list(self.hist_h) + [0.0]
            if self.show_q and pq:
                vals += [e[1] * cm for e in pq]
            lo, hi = min(vals), max(vals)
            pad = max(0.1 * (hi - lo), 0.5)
            lo, hi = lo - pad, hi + pad
            t_max = max(self.hist_t[-1], 1.0)

            def sx(tt):
                return px0 + tt / t_max * pw

            def sy(v):
                return py0 + ph - (v - lo) / (hi - lo) * ph

            # dashed zero line (prominent) + label
            zy = sy(0.0)
            self._dashed_hline(d, px0, px0 + pw, int(zy), (90, 90, 90),
                               2 * S)
            zl = render_math("$0$", 11 * S, GRAY, cache=True)
            img.paste(zl, (px0 + pw - zl.width - X(2),
                           int(zy) - zl.height - X(2)), zl)
            # h(z_t) — blue failure function
            pts = [(sx(tt), sy(v)) for tt, v in zip(self.hist_t,
                                                    self.hist_h)]
            d.line(pts, fill=BLUE, width=2 * S)
            # Q(z_t, pi_safe(z_t)) — black on nominal ticks, red on safe
            if self.show_q and len(pq) >= 2:
                qp = [(sx(e[0]), sy(e[1] * cm), e[2]) for e in pq]
                for i in range(1, len(qp)):
                    col = RED if qp[i][2] else BLACK
                    d.line([qp[i - 1][:2], qp[i][:2]], fill=col,
                           width=2 * S)
            # x axis with arrow + label
            ay = py0 + ph
            d.line([px0, ay, px0 + pw + X(8), ay], fill=BLACK, width=2 * S)
            d.polygon([(px0 + pw + X(14), ay),
                       (px0 + pw + X(4), ay - X(5)),
                       (px0 + pw + X(4), ay + X(5))], fill=BLACK)
            tl = render_math("time ($t$)", 12 * S, BLACK, cache=True)
            img.paste(tl, (px0 + (pw - tl.width) // 2, ay + X(8)), tl)
        return img.resize((320, 480), Image.LANCZOS)

    def add(self, frame_rgb, t, step, n_steps, h, V, control, ratio,
            extras=None):
        force = float(extras.get("contact_force", 0.0)) if extras else 0.0
        touch = bool(extras.get("contact_touch", False)) if extras else False
        iv = extras.get("intervened", (0, 0.0))[0] if extras else 0
        # render the PREVIOUS frame now — its pq entry has been computed
        if self._pending is not None:
            pf, pt, ph_, pf_, ptouch, piv = self._pending
            panel = self._panel(pt, ph_, pf_, ptouch, piv)
            self.writer.append_data(np.hstack([pf, np.asarray(panel)]))
        self._pending = (frame_rgb, t, h, force, touch, iv)

    def close(self):
        if self._pending is not None:
            pf, pt, ph_, pf_, ptouch, piv = self._pending
            panel = self._panel(pt, ph_, pf_, ptouch, piv)
            self.writer.append_data(np.hstack([pf, np.asarray(panel)]))
            self._pending = None
        if self.writer is not None:
            self.writer.close()


def make_presentation_rollout_cls():
    """RolloutController subclass logging a per-tick Q series (self.pq).

    pq holds (t, Q, safe) for EVERY tick with Q = Q(z_t, pi_safe(z_t))
    (flt.q_actor — the safe actor is re-queried at every state, so the
    curve updates instantly even when the arm idles). `safe` marks whether
    the tick was actually DRIVEN by the safe actor (curve color).
    """
    from main.train.rollout import RolloutController, SRC_ACTOR

    class PresentationRollout(RolloutController):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pq = []

        def _capture_tick(self):
            obs = super()._capture_tick()
            if (self.flt is not None and self.filter_active
                    and self._obs_cache is not None):
                _, body, enc = self._obs_cache
                q = float(self.flt.q_actor(
                    enc, body["Jlin"][None].cuda(),
                    body["Jang"][None].cuda()))
                t_now = self._n_physics / self.env.PHYSICS_FREQ
                self.pq.append((t_now, q, self._tick_source == SRC_ACTOR))
            return obs

    return PresentationRollout


def record_one(cfg, seed, task, emb, budget, encoder, kin, model_kw,
               video_path, show_q):
    """Replay one eval scene and record the clean-panel video -> trace."""
    from main.train.eval_ckpt import set_global_seeds

    cfg_ep = dataclasses.replace(cfg, task=task, embodiment=emb)
    video = CleanPanelVideoWriter(
        video_path, fps=25.0, h_scale=float(getattr(cfg, "h_scale", 1.0)),
        show_q=show_q)
    set_global_seeds(seed)
    ro = make_presentation_rollout_cls()(
        cfg_ep, seed, mode="filtered", encoder=encoder, kin=kin, video=video,
        max_steps=int(budget * 250.0), **model_kw)
    assert ro.env.obstacle is not None, f"spawn failed for seed {seed}"
    video.pq = ro.pq
    trace = ro.run()
    video.close()
    ro.env.close()
    return trace


def trace_collision(trace) -> bool:
    """Eval 'collision' definition: any contact touch or force >= 30 N."""
    if any(trace["contact_touch"]):
        return True
    f = trace["contact_force"]
    return bool(len(f) and max(f) >= FORCE_VIOL_N)


def _preview_frame(out_png: Path, src_mp4: Path | None) -> None:
    """Render ONE fully-populated panel with synthetic data (CPU only)."""
    import imageio.v2 as imageio

    w = CleanPanelVideoWriter(None, fps=25.0, h_scale=20.0, show_q=True)
    dt = 0.04
    ts = np.arange(310) * dt                     # history up to t = 12.36
    h_cm = (12.0 + 10.0 * np.exp(-((ts - 2.0) / 2.5) ** 2)
            - 18.0 * np.exp(-((ts - 8.0) / 1.8) ** 2))
    h_cm = np.maximum(h_cm, 1.5)
    q_cm = h_cm - 2.5 - 1.0 * np.sin(ts * 2.1)
    safe = (ts >= 6.5) & (ts <= 13.0)            # red tail on the Q curve
    w.hist_t = list(ts)
    w.hist_h = list(h_cm)
    w.pq = [(float(tt), float(qq / w.cm), bool(ss))
            for tt, qq, ss in zip(ts, q_cm, safe)]
    w._min_h = float(np.min(h_cm) / w.cm)
    w._max_force = 29.8
    panel = w._panel(float(ts[-1] + dt), float(h_cm[-1] / w.cm),
                     12.3, False, 9)
    if src_mp4 is not None and Path(src_mp4).exists():
        rd = imageio.get_reader(str(src_mp4))
        left = rd.get_data(150)[:, :640]
        rd.close()
    else:
        left = np.full((480, 640, 3), 235, np.uint8)
    imageio.imwrite(str(out_png), np.hstack([left, np.asarray(panel)]))
    print(f"[preview] wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--only", type=str, default=None,
                   help='single combo "task/embodiment" (smoke test)')
    p.add_argument("--max-trials", type=int, default=6,
                   help="candidates tried per combo before falling back to "
                        "the first loo-clean pair")
    p.add_argument("--preview-frame", type=Path, default=None,
                   help="render ONE synthetic panel PNG (CPU only) and exit")
    p.add_argument("--preview-src", type=Path, default=None,
                   help="mp4 whose mid frame supplies the observer image")
    args = p.parse_args()

    if args.preview_frame is not None:
        _preview_frame(args.preview_frame, args.preview_src)
        return

    combos = json.loads(args.candidates.read_text())
    if args.only:
        combos = {args.only: combos[args.only]}
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from main.train.config import FrozenConfig
    from main.train.eval_ckpt import build_models

    def load_cfg(run):
        cfg = FrozenConfig.load(RUNS_DIR / run / "config.json")
        cfg.obstacle_mode = "on_path"
        cfg.filter_release_margin = 0.0     # matches eval_full/..._rm0
        return cfg

    # nominal encoder is ckpt-free — build once for all combos
    nom_kind, nom_encoder, nom_kin_fn, _ = build_models(load_cfg("loo_franka"),
                                                        None)
    assert nom_kind == "nominal"
    model_cache = {}

    def loo_models(run):
        if run not in model_cache:
            cfg = load_cfg(run)
            kind, encoder, kin_fn, model_kw = build_models(
                cfg, RUNS_DIR / run / "checkpoint.pt")
            assert kind == "hb", kind
            model_cache[run] = (cfg, encoder, kin_fn, model_kw)
        return model_cache[run]

    def run_selection(cands, task, emb, strict_clean):
        """Try candidates in order: verify the LOO replay first (GPU
        nondeterminism can flip the outcome vs the eval row — the row is
        only a hint), then record the nominal on the same seed. Returns
        (chosen_cand, nominal_collides), preferring a pair where the
        nominal collides; falls back to the first loo-clean pair."""
        fallback = None    # (cand, loo_vp, nom_vp): loo clean, nominal clean
        for i, cand in enumerate(cands[: args.max_trials]):
            # candidates may span tasks (extra pool) — use the cand's own
            ctask, cemb = cand["task"], cand["embodiment"]
            cfg, encoder, kin_fn, model_kw = loo_models(cand["run"])
            loo_vp = out_dir / (f"{cand['run']}_{cemb}_{ctask}"
                                f"_ep{cand['episode']:03d}.mp4")
            trace = record_one(cfg, cand["seed"], ctask, cemb,
                               cand["budget"], encoder, kin_fn(cemb),
                               model_kw, loo_vp, show_q=True)
            n = len(trace["h"])
            cm = 100.0 / float(cfg.h_scale)
            min_h = min(trace["h"]) * cm if n else float("nan")
            touch = any(trace["contact_touch"])
            fmax = max(trace["contact_force"]) if trace["contact_force"] else 0.0
            clean = not touch and fmax < FORCE_VIOL_N
            loo_ok = bool(trace["success"]) and (
                clean or (cand["contact"] and not strict_clean))
            print(f"[loo] {ctask}/{cemb} cand{i} {cand['run']} "
                  f"ep{cand['episode']:03d} seed={cand['seed']}: "
                  f"success={trace['success']} contact={touch} "
                  f"fmax={fmax:.1f}N min_h={min_h:.2f}cm dur={n / 25:.1f}s "
                  f"{'' if loo_ok else '— REJECT (replay != eval row)'}")
            if not loo_ok:
                loo_vp.unlink(missing_ok=True)
                continue
            # loo is good on this scene — now the nominal on the same seed
            nom_vp = out_dir / (f"nominal_{cemb}_{ctask}"
                                f"_ep{cand['episode']:03d}.mp4")
            trace_nom = record_one(load_cfg(cand["run"]), cand["seed"],
                                   ctask, cemb, cand["budget"], nom_encoder,
                                   nom_kin_fn(emb), {}, nom_vp, show_q=False)
            coll = trace_collision(trace_nom)
            print(f"[nom] {ctask}/{cemb} cand{i} seed={cand['seed']}: "
                  f"success={trace_nom['success']} collision={coll} "
                  f"dur={len(trace_nom['h']) / 25:.1f}s")
            if coll:
                if fallback is not None:
                    fallback[1].unlink(missing_ok=True)
                    fallback[2].unlink(missing_ok=True)
                return cand, True
            if fallback is None:
                fallback = (cand, loo_vp, nom_vp)
            else:
                loo_vp.unlink(missing_ok=True)
                nom_vp.unlink(missing_ok=True)
        if fallback is not None:
            print(f"[sel] {task}/{emb}: no colliding nominal in "
                  f"{min(len(cands), args.max_trials)} trials — keeping "
                  f"the first loo-clean pair (seed {fallback[0]['seed']})")
            return fallback[0], False
        print(f"[sel] {task}/{emb}: NO clean loo replay among "
              f"{min(len(cands), args.max_trials)} candidates — SKIPPED")
        return None

    used = {}
    for combo, cands in combos.items():
        task, emb = combo.split("/")
        res = run_selection(cands, task, emb, strict_clean=False)
        if res is not None:
            used[combo] = {**res[0], "nominal_collision": res[1]}

    # ---- extra set: one loo_franka pair per embodiment, ANY task —
    # strict clean success, smallest min_h first, nominal collision
    # preferred. Scenes already picked above are not reused.
    extra = {}
    if not args.only:
        taken = {(u["embodiment"], u["task"], u["episode"])
                 for u in used.values()}
        embs = sorted({k.split("/")[1] for k in combos})
        for emb in embs:
            pool = []
            for combo, cands in combos.items():
                task, e2 = combo.split("/")
                if e2 != emb:
                    continue
                for cand in cands:
                    if cand["run"] != "loo_franka":
                        continue
                    if (emb, task, cand["episode"]) in taken:
                        continue
                    pool.append(cand)
            pool.sort(key=lambda c: c["min_h_cm"])   # smaller min_h first
            if not pool:
                print(f"[extra] {emb}: no loo_franka candidates — SKIPPED")
                continue
            res = run_selection(pool, pool[0]["task"], emb,
                                strict_clean=True)
            # NOTE: candidates in the pool may span tasks; run_selection
            # prints the task of each cand itself via the cand dict
            if res is not None:
                extra[emb] = {**res[0], "nominal_collision": res[1]}

    (out_dir / "selections_used.json").write_text(json.dumps(
        {"main": used, "extra_loo_franka": extra}, indent=1))
    n_coll = sum(1 for v in used.values() if v["nominal_collision"])
    n_coll_x = sum(1 for v in extra.values() if v["nominal_collision"])
    print(f"RECORD DONE — main: nominal collides in "
          f"{n_coll}/{len(used)} pairs; extra loo_franka: "
          f"{n_coll_x}/{len(extra)} pairs")


if __name__ == "__main__":
    main()
