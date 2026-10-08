"""RoundSchedule: global (task, embodiment) cursor, T1 host cursor per embodiment, state round trip."""

from collections import Counter

from main.lang.config import CANON_EMBODIMENTS, CANON_T1_HOSTS, CANON_TASKS
from main.lang.rounds import RoundSchedule


def test_schedule_balanced_and_resumable():
    s = RoundSchedule(CANON_TASKS, CANON_EMBODIMENTS, CANON_T1_HOSTS)
    slots = [s.next() for _ in range(14 * 5)]
    assert set(Counter((t, e) for t, e, _ in slots).values()) == {5}
    hosts = Counter((e, h) for t, e, h in slots if t == "selective_avoid")
    assert set(hosts.values()) == {1} and len(hosts) == 10   # every host once per embodiment
    st = s.state()
    a = [s.next() for _ in range(9)]
    s2 = RoundSchedule(CANON_TASKS, CANON_EMBODIMENTS, CANON_T1_HOSTS)
    s2.load(st)
    assert [s2.next() for _ in range(9)] == a


# ---------------------------------------------------------------- stop / resume inside a round (2026-10-07) --
import dataclasses  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from main.lang.config import LangConfig  # noqa: E402
from main.lang.metrics import LangRoundMetrics  # noqa: E402
from main.lang.rounds import run_rounds  # noqa: E402
from main.lang.train import EXIT_RESUME, LangTrainer  # noqa: E402

from .fake_data import make_root  # noqa: E402
from .test_train import StubText  # noqa: E402


class FakeCollect:
    """Stands in for run_episodes: one line per slot (seed = offset + attempt, filter drawn from rng_policy
    as run_episodes does), a metrics row per filter episode, two attempts per slot (one retry). kill_at /
    stop_at = (round, slot): raise after writing the line (a hard kill mid-episode) / request a clean stop."""

    def __init__(self, tr=None, kill_at=None, stop_at=None, skip_at=(1, 0)):
        self.tr, self.kill_at, self.stop_at, self.skip_at = tr, kill_at, stop_at, skip_at

    def __call__(self, cfg, out_root, slots, offset, buf, side, *, rng_policy, attempt0, slot_offset, extra_line,
                 round_metrics, **kw):
        r = int(extra_line["round"])
        task, emb, host = slots[0]
        filt = bool(rng_policy.random() < 0.8)
        if self.skip_at == (r, slot_offset):   # every build of this slot failed (run_episodes' skipped line)
            with open(Path(out_root) / "lang_episodes.jsonl", "a") as f:
                f.write(json.dumps({"skipped": True, "task": task, "embodiment": emb, "slot": slot_offset,
                                    "attempt_next": attempt0 + 5, **extra_line}) + "\n")
            return {"episodes": 0, "attempt": attempt0 + 5, "wall": 0.01}
        eid = offset + attempt0 + 1
        with open(Path(out_root) / "lang_episodes.jsonl", "a") as f:
            f.write(json.dumps({"episode_id": eid, "round": r, "slot": slot_offset, "task": task,
                                "embodiment": emb, "filter": filt}) + "\n")
        if filt:
            rm, ri = round_metrics
            rm.add({"round": ri, "task": task, "host": host, "embodiment": emb, "consequence": "avoid",
                    "designation": "harm" if slot_offset % 2 == 0 else "benign", "episode_id": eid,
                    **{k: 0.5 if k in ("intervention_rate", "v_le_h_frac", "harm_named_frac") else False
                       for k in LangRoundMetrics.KEYS}})
        if self.kill_at == (r, slot_offset):
            raise RuntimeError("killed mid-episode")
        if self.stop_at == (r, slot_offset):
            self.tr._stop = True
        return {"episodes": 1, "attempt": attempt0 + 2, "wall": 0.01}


@pytest.fixture
def rounds_root(tmp_path):
    return make_root(tmp_path / "data", episodes=4, capacity=400)


def _rounds(run, root, collect, resume, stop_step=None):
    cfg = dataclasses.replace(LangConfig(), batch_size=8, max_steps_per_episode=10)
    tr = LangTrainer(cfg, run, [root], (), (), use_wandb=False, prefetch=False, device="cpu",
                     text_encoder=StubText(), seed=0)
    collect.tr = tr
    if stop_step is not None:   # a clean stop requested in the middle of a round's training
        step0 = tr.grad_step

        def grad_step(*a, **k):
            out = step0(*a, **k)
            if tr.step == stop_step:
                tr._stop = True
            return out

        tr.grad_step = grad_step
    args = SimpleNamespace(run=str(run), resume=resume, collect_rounds=3, episodes_per_round=2,
                           grad_steps_per_round=4, probe_every_rounds=0, video_every=0, no_wandb=True, seed=0,
                           time_budget=None)
    rc = run_rounds(cfg, args, [root], [], [], None, None, trainer=tr, collect_fn=collect)
    return rc, tr


def _episodes(run):
    lines = [json.loads(x) for x in (Path(run) / "ring" / "lang_episodes.jsonl").read_text().splitlines()]
    return [(d["round"], d["slot"], d.get("episode_id"), d["task"], d.get("filter")) for d in lines]


def test_rounds_stop_and_resume_inside_a_round(tmp_path, rounds_root):
    torch.set_num_threads(4)
    rc, ref = _rounds(tmp_path / "a", rounds_root, FakeCollect(), resume=False)
    assert rc == 0 and ref.step == 4 * 4             # rounds 0..3 train 4 steps each
    want = _episodes(tmp_path / "a")
    assert [(r, s) for r, s, *_ in want] == [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    assert want[0][2] is None                          # slot (1, 0) skipped: no episode
    assert len({e for _, _, e, *_ in want[1:]}) == 5   # one seed per episode

    run = tmp_path / "b"
    with pytest.raises(RuntimeError):                  # 1. killed inside round 1's second episode
        _rounds(run, rounds_root, FakeCollect(kill_at=(1, 1)), resume=False)
    rc, tr = _rounds(run, rounds_root, FakeCollect(stop_at=(2, 0)), resume=True)   # 2. stop between episodes
    assert rc == EXIT_RESUME and tr.round == 2
    st = torch.load(run / "state.pt", weights_only=False)
    assert (st["phase"], st["slots_done"]) == ("collect", 1)
    rc, tr = _rounds(run, rounds_root, FakeCollect(), resume=True, stop_step=10)    # 3. stop mid-training
    assert rc == EXIT_RESUME and tr.step == 10
    st = torch.load(run / "state.pt", weights_only=False)
    assert (st["round"], st["phase"], st["steps_target"]) == (2, "train", 12)
    rc, tr = _rounds(run, rounds_root, FakeCollect(), resume=True)                  # 4. to the end
    assert rc == 0 and tr.step == ref.step
    assert _episodes(run) == want                      # same episodes, seeds and filter draws, no duplicates
    rows = [json.loads(x) for x in (run / "eval" / "rounds.jsonl").read_text().splitlines()]
    ref_rows = [json.loads(x) for x in (tmp_path / "a" / "eval" / "rounds.jsonl").read_text().splitlines()]
    assert [r["episode_id"] for r in rows] == [r["episode_id"] for r in ref_rows]
    assert (run / "eval" / "trend.png").exists()
    for x, y in zip(ref.critics.state_dict().values(), tr.critics.state_dict().values()):
        assert torch.allclose(x, y, atol=1e-6)         # the interrupted run trained exactly like the reference


def test_make_ring_reopens_out_of_lockstep_and_checks_capacity(tmp_path):
    """O2: a pair whose headers differ (a kill between the two flushes) reopens, for the resume to roll
    back; O3: resuming with more rounds than the ring was sized for refuses."""
    from main.lang.rounds import make_ring

    cfg = dataclasses.replace(LangConfig(), max_steps_per_episode=10)
    buf, side = make_ring(tmp_path, cfg, 3, 2)
    hp = tmp_path / "ring" / "item_buffer" / "header.json"
    h = json.loads(hp.read_text())
    h["n"] = h["head"] = 5                                   # sidecar flushed, StepBuffer not
    hp.write_text(json.dumps(h))
    buf2, side2 = make_ring(tmp_path, cfg, 3, 2)
    assert (len(buf2), len(side2)) == (0, 5)
    with pytest.raises(SystemExit, match="capacity"):
        make_ring(tmp_path, cfg, 30, 2)


def test_make_ring_pairs_a_missing_sidecar(tmp_path):
    """B2: the first launch died after the StepBuffer header and before the sidecar existed."""
    import shutil

    from main.lang.rounds import make_ring

    cfg = dataclasses.replace(LangConfig(), max_steps_per_episode=10)
    make_ring(tmp_path, cfg, 3, 2)
    shutil.rmtree(tmp_path / "ring" / "item_buffer")
    buf, side = make_ring(tmp_path, cfg, 3, 2)
    assert side.capacity == buf.capacity and len(side) == len(buf) == 0


def test_resume_refuses_another_h_definition(tmp_path, rounds_root):
    """Review: state.pt records the h definition; a resume under another one refuses."""
    run = tmp_path / "r"
    with pytest.raises(RuntimeError):   # (1, 0) is FakeCollect's skipped slot: kill at (1, 1)
        _rounds(run, rounds_root, FakeCollect(kill_at=(1, 1)), resume=False)
    st = torch.load(run / "state.pt", weights_only=False)
    assert st.get("h_code") and st.get("h_code_v2") and "git_head" in st and st["data_roots"]
    good = dict(st)
    st["h_code_v2"] = "0" * 12
    torch.save(st, run / "state.pt")
    with pytest.raises(SystemExit, match="h definition changed"):
        _rounds(run, rounds_root, FakeCollect(), resume=True)
    # a state saved before h_code_v2 existed is compared on the first hash
    old = {k: v for k, v in good.items() if k not in ("h_code_v2", "data_roots")}
    old["h_code"] = "0" * 12
    torch.save(old, run / "state.pt")
    with pytest.raises(SystemExit, match="h definition changed"):
        _rounds(run, rounds_root, FakeCollect(), resume=True)


def test_resume_refuses_other_data_or_code(tmp_path, rounds_root):
    """Review round 3, C4: state.pt records the resolved data roots and the git commit; a resume with other
    data refuses, and one at another commit refuses unless --accept-code-change."""
    from main.lang.train import check_resume_identity

    run = tmp_path / "r"
    with pytest.raises(RuntimeError):
        _rounds(run, rounds_root, FakeCollect(kill_at=(1, 1)), resume=False)
    st = torch.load(run / "state.pt", weights_only=False)
    assert st["data_roots"] == [str(Path(rounds_root).resolve())]
    ident = {k: st.get(k) for k in ("h_code", "h_code_v2", "git_head", "data_roots")}
    check_resume_identity(st, ident, run)                                  # the same: no refusal
    with pytest.raises(SystemExit, match="data roots changed"):
        check_resume_identity(st, {**ident, "data_roots": ["/elsewhere"]}, run)
    with pytest.raises(SystemExit, match="another git commit|--accept-code-change"):
        check_resume_identity({**st, "git_head": "f" * 40}, {**ident, "git_head": "e" * 40}, run)
    check_resume_identity({**st, "git_head": "f" * 40}, {**ident, "git_head": "e" * 40}, run, accept_code_change=True)
