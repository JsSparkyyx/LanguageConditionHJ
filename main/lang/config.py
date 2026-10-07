"""LangConfig (VANILLA_PLAN Section 11) and the seed / episode-id scheme (Section 4.2).

``FrozenConfig`` defaults stay unchanged; ``LangConfig`` subclasses it. Every
data root, run and checkpoint saves its LangConfig and logs its hash.

The per-episode env overrides ``item_host``, ``item_risk``,
``item_active_rule`` and ``item_out_root`` are declared fields:
``dataclasses.replace`` (used by ``collect.sample_scene`` and every episode
loop here) silently drops undeclared attributes, and ``build_item_training_env``
reads them from the cfg. ``item_out_root`` must never be None (the runner
would build ``Path(None)`` and every episode build would fail).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from main.train.config import FrozenConfig

from . import REPO

# canonical index tuples: emb_idx / task_idx / host_idx are indices HERE, never
# in the (possibly filtered) task_choices / embodiment_choices, so a
# single-task or single-embodiment run evaluates on the same seeds as `full`
CANON_TASKS = ("selective_avoid", "granular_pour", "push_zone", "granular_carry",
               "tray_transport", "clean_pour", "gap_place")
CANON_EMBODIMENTS = ("ur5-wsg", "franka-panda")
CANON_T1_HOSTS = ("place_container_plate", "place_burger_fries", "stack_blocks_two",
                  "stack_bowls_two", "place_bread_basket")  # registry order (host_tasks)

TASK_SHORT = {t: f"T{i + 1}" for i, t in enumerate(CANON_TASKS)}

# Section 4.2 bounds
WARMUP_G_MAX = 50
VAL_G_MAX = 30
STREAM_SPAN = 1_000_000
EVAL_BASE = 1_500_000_000
INT32_MAX = 2**31 - 1


def warmup_offset(g: int) -> int:
    """episode_offset of warmup shard g: (1 + g) * 1e6 (ids 1M ... 51M)."""
    assert 0 <= g < WARMUP_G_MAX, f"warmup shard g={g} outside [0, {WARMUP_G_MAX})"
    return (1 + g) * STREAM_SPAN


def val_offset(g: int) -> int:
    """episode_offset of validation shard g: (60 + g) * 1e6 (ids 60M ... 90M)."""
    assert 0 <= g < VAL_G_MAX, f"validation shard g={g} outside [0, {VAL_G_MAX})"
    return (60 + g) * STREAM_SPAN


def round_offset(r: int) -> int:
    """episode_offset of round r: (100 + r) * 1e6 (>= 100M; disjoint from eval below 1400 rounds)."""
    assert 0 <= r < 1400, f"round {r} would collide with the evaluation seeds"
    return (100 + r) * STREAM_SPAN


def shard_g(emb: str, shard: int, n_shards: int) -> int:
    """g = emb_idx * n_shards + shard (emb_idx in CANON_EMBODIMENTS)."""
    assert 0 <= shard < n_shards
    return CANON_EMBODIMENTS.index(emb) * n_shards + shard


def eval_seed(task: str, emb: str, scene_idx: int, attempt: int = 0, host: str | None = None) -> int:
    """Evaluation scene seed (not an episode id): 1.5e9 + task_idx*1e7 +
    host_idx*1e6 + emb_idx*1e5 + attempt*1000 + scene_idx."""
    assert 0 <= scene_idx < 1000 and 0 <= attempt < 10
    host_idx = CANON_T1_HOSTS.index(host) if host is not None else 0
    s = (EVAL_BASE + CANON_TASKS.index(task) * 10_000_000 + host_idx * 1_000_000
         + CANON_EMBODIMENTS.index(emb) * 100_000 + attempt * 1000 + scene_idx)
    assert s <= INT32_MAX
    return s


def check_stream(offset: int, attempt: int) -> int:
    """episode id = offset + attempt, asserted inside its stream and int32."""
    assert 0 <= attempt < STREAM_SPAN, f"attempt {attempt} overflows its stream"
    eid = int(offset) + int(attempt)
    assert eid <= INT32_MAX
    return eid


def bank_hash() -> str:
    """Hash of the paraphrase bank + splits.yaml (written into every checkpoint)."""
    h = hashlib.md5()
    for p in (REPO / "envs/main/item_envs/language/paraphrases.yaml",
              REPO / "envs/main/item_envs/configs/splits.yaml"):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


@dataclass
class LangConfig(FrozenConfig):
    # --- tasks and embodiments (FrozenConfig fields, new defaults) ---
    task: str = "selective_avoid"
    embodiment: str = "ur5-wsg"
    task_choices: tuple = CANON_TASKS
    embodiment_choices: tuple = CANON_EMBODIMENTS
    t1_hosts: tuple = CANON_T1_HOSTS
    # control ticks per episode (x 10 physics steps); Phase 0 measured the
    # item tasks (docs/lang_filter/recon.md)
    max_steps_per_episode: int = 750

    # --- per-episode env overrides (declared: dataclasses.replace keeps them) ---
    item_host: str | None = None
    item_risk: dict | None = None
    item_active_rule: str | None = None
    item_out_root: str = "runs/item_envs"

    # --- model ---
    k_prev: int = 2
    n_queries: int = 4
    max_text_tokens: int = 24
    text_backend: str = "holobrain_bert"          # | bert_raw | sentence_generic
    backend_paths: dict = field(default_factory=lambda: {
        "sentence_generic": "/work/nvme/bifq/project/tmp/hf-cache/all-mpnet-base-v2",
    })
    trunk_depth: int = 3
    critic_depth: int = 2
    lang_scene: bool = True
    lang_body: bool = True
    lang_film: bool = True
    lang_queries: bool = True
    lang_rtoken: bool = True
    film_zero_init: bool = True
    softmin_T_h: float = 0.05                      # h units, NOT scaled by h_scale
    activation_checkpointing: bool = False

    # --- sampling / relabeling ---
    shuffle_text: bool = False
    relabel: str = "paired"                        # | random | fixed
    harmful_frac: float = 0.5
    pair_frac: float = 0.5
    task_balance: bool = True
    recall_horizon_ticks: int = 7

    # --- rounds / collection ---
    filter_release_margin: float = 0.0
    named_harm_frac: float = 0.5
    episodes_per_round: int = 4
    grad_steps_per_round: int = 2048
    collect_rounds: int = 100
    max_inits_per_process: int = 200
    warmup_episodes_per_task: int = 30
    val_episodes_per_task: int = 5

    bank_hash: str = ""
    lang_version: int = 1

    def __post_init__(self):
        # Trainer.gamma() interpolates gamma -> gamma_final: they must match
        assert self.gamma_final == self.gamma, "LangConfig: gamma_final must equal gamma"
        assert self.relabel in ("paired", "random", "fixed"), self.relabel
        assert self.text_backend in ("holobrain_bert", "bert_raw", "sentence_generic"), self.text_backend
        assert self.item_out_root, "item_out_root must never be None/empty"
        if not self.bank_hash:
            self.bank_hash = bank_hash()

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LangConfig":
        names = {f.name: f for f in dataclasses.fields(cls)}
        kw = {}
        for k, v in d.items():
            if k not in names:
                continue
            default = names[k].default
            if isinstance(v, list) and (isinstance(default, tuple) or k in ("feature_level", "eval_seeds")):
                v = tuple(v)
            kw[k] = v
        return cls(**kw)

    @classmethod
    def load(cls, path: str | Path) -> "LangConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def hash(self) -> str:
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def path_switches(self) -> dict:
        return {k: getattr(self, k) for k in ("lang_scene", "lang_body", "lang_film", "lang_queries", "lang_rtoken")}
