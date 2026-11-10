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
import functools
import hashlib
import math
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
# the tasks new runs train, collect and evaluate on: T1, T3, T5, T6, T7 (Yuxuan 2026-10-22: T4 granular_carry dropped,
# "we don't need that much tasks"; 2026-10-23: T2 granular_pour replaced by the redesigned T6 clean_pour, three bowls,
# since T2's one rule is not text-conditioned). The dropped tasks' episodes stay in the shards and CANON_TASKS keeps
# their indices; runs saved with other task lists keep them
TRAIN_TASKS = tuple(t for t in CANON_TASKS if t not in ("granular_carry", "granular_pour"))
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


VALG_G_MAX = 9


def valg_offset(g: int) -> int:
    """episode_offset of grounding-validation shard g: (91 + g) * 1e6 (ids 91M ... 100M; the T1 / T7 ur5-wsg
    validation of review round 3, Part II E1), between the validation and the rounds streams."""
    assert 0 <= g < VALG_G_MAX, f"grounding-validation shard g={g} outside [0, {VALG_G_MAX})"
    return (91 + g) * STREAM_SPAN


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


@functools.lru_cache(maxsize=None)
def bank_hash(vocab: str = "current") -> str:
    """Hash of the paraphrase bank, splits.yaml and the object names (written into every checkpoint), of the given
    vocabulary (legacy_pool.vocab_files: "current" = today's files, "pre_1012" = the vanilla runs' files)."""
    from .legacy_pool import vocab_files

    objects, bank = vocab_files(vocab)
    h = hashlib.md5()
    for p in (bank, REPO / "envs/main/item_envs/configs/splits.yaml",
              objects):   # the object names too (2026-10-12)
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


@dataclass
class LangConfig(FrozenConfig):
    # --- HJ discount (FrozenConfig fields, new default; Yuxuan 2026-10-08, decision 1). A saved config keeps
    # its own value, so runs started at 0.9 resume at 0.9 ---
    gamma: float = 0.95
    gamma_final: float = 0.95

    # --- tasks and embodiments (FrozenConfig fields, new defaults) ---
    task: str = "selective_avoid"
    embodiment: str = "ur5-wsg"
    task_choices: tuple = TRAIN_TASKS
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
    # TF32 tensor-core matmuls (inputs rounded to 10 mantissa bits, fp32 accumulation) inside the training
    # step only; collection, probes and evaluation keep fp32 matmuls. A config saved before this field
    # (2026-10-08) loads as False, so older runs resume as they trained
    tf32_train: bool = True
    # actor head: an MLP of this width (network.LangTokenActor; Yuxuan, 2026-10-08), 0 = CrossSafe's single
    # Linear(256, 2). A config saved before the field loads as 0: those checkpoints hold the linear head
    actor_hidden: int = 256
    # review round 3, Part II E2 (short grounding runs only; all off by default, the official runs unchanged):
    text_out_norm: bool = False          # V1: a final RMSNorm on the text adapter's outputs (r0, q, w')
    mask_special_tokens: bool = False    # V2: [CLS] / [SEP] out of the keys of body / scene-reads-words
    oracle_ref_tag: bool = False         # V3: simulator-state tag of the named object's scene tokens (offline
    oracle_tag_sigma: float = 0.04       #     measuring tool only; the rollout refuses it), width in metres
    # V4 (Yuxuan 2026-10-08, decision 5b): readout query q1 predicts the named item's centre at frames t and 0
    # (simulator labels, training only; the head is unused at deployment). Huber loss in metres, weight
    # aux_loss_weight relative to the critic loss; it trains the encoder and the head
    aux_target_loss: bool = False
    aux_loss_weight: float = 0.1
    # review round 4, item 3: prediction and target in units of aux_unit (0.05 m, the h unit of distances) and the
    # Huber delta in those units (in metres a 10 cm miss cost ~0.005: a few % of the critic loss)
    aux_unit: float = 0.05
    aux_huber_delta: float = 1.0
    # V4b (round 4, item 3, optional): q1's scene attention (last encoder block, head-averaged, renormalized
    # within frames t and 0) pulled to the named item's centre: sum_j a_f,j |p_j - c_f|^2 / aux_unit^2
    aux_attention_loss: bool = False
    aux_attention_weight: float = 0.1
    # V5 (FUSION_PLAN, 2026-10-09): HoloBrain's frozen text-image fusion (feature_enhancer) runs on the neck maps
    # with the sample's sentence before the spatial enhancer, so scene tokens AND their positions depend on the
    # sentence. Needs version-2 shards (neck_buffer/). Off by default until V5 passes its gate (E5 decision rule:
    # adopt, then flip); a config saved before the field loads as False
    scene_fusion: bool = False
    fusion_words: bool = True       # with scene_fusion: the text adapter reads the fused words (frame t; t+1 target)
    fusion_dtype: str = "fp32"      # the frozen stage's matmuls everywhere (train, eval, rollout): fp32 | tf32 | bf16
    fusion_chunk: int = 32          # frames per fusion pass (memory only, no effect on values)
    # decision 6 (Yuxuan, 2026-10-09): the critic's readout. "mlp" = an MLP over [r, queries, masked body mean]
    # (FUSION_PLAN 3.6); "softmin" = CrossSafe's soft minimum over the body / query values (softmin_T_h). A config
    # saved before the field loads as "softmin", so those checkpoints rebuild the head they trained
    critic_readout: str = "mlp"
    critic_mlp_hidden: int = 512
    # ACTION_SENSITIVITY_PLAN (2026-11-10), section 3. critic_injection: how the action enters the critic. "linear" =
    # CrossSafe's one Linear on the raw action effects, added to the body tokens. "revised" = RMSNorm on the body
    # tokens first, the effects divided by their bounds, MLP(. -> 256 -> 256) + RMSNorm, and the mean action embedding
    # of the valid body tokens joins the MLP readout's input. critic_chunk: the ticks one decision covers: 1 = one
    # tick (the action is dtheta [B, A]); 12 = a 0.48 s chunk (the action is the per-tick commands [B, 12, A],
    # summarized by their cumulative sums after 4, 8 and 12 ticks). A config saved before the fields reads
    # "linear" / 1, so its checkpoint rebuilds the head it trained
    critic_injection: str = "linear"
    critic_chunk: int = 1
    # E6 scene-count curve: train on the first N episodes per (task, embodiment) of a fixed nested order (0 = all).
    # Training sampler only (the validation / probe sampler always sees every episode)
    train_episodes_per_task: int = 0
    # V6 (QWEN_PLAN, Yuxuan 2026-10-14; contract docs/lang_filter/qwen_impl.md): the frozen front end.
    # "holobrain_gd" = HoloBrain-GD scene tokens (3 cameras x 400 per frame) with BERT words from the TextStore.
    # "holobrain_qwen" = HoloBrain's Qwen2.5-VL variant: 3 x 99 scene tokens per frame that never see the sentence
    # (vision tower + layer 0 over the image prefix, causal) and words from layer 0 run on the sentence against
    # the frame's cached prefix keys / values (kv_buffer/), so the words are image-conditioned. A config saved
    # before the field loads as GD
    encoder: str = "holobrain_gd"
    # Qwen encoder only: the prefix the words attend to. "frame" = frame t's (frame t+1's for the target);
    # "blank" = one fixed prefix built from black images (the V6-blank control of QWEN_PLAN E8)
    qwen_words_prefix: str = "frame"
    # Qwen encoder only: the vision tower's and layer 0's attention kernel (numerics only): sdpa | flash_attention_2 |
    # eager (eager returns attention weights, E7)
    qwen_attn: str = "sdpa"
    # T1 candidate pool. "current" = objects.yaml as it is (the hand-picked corpus from 2026-10-22, training models
    # and colours only). "heldout" = the OOD scenes: only the held-out models (objects.yaml heldout_ids) and the
    # held-out colours, every rule in the heldout_item split. "pre_1012" = the pool before the 2026-10-12 renaming
    # (068_boxdrink ids 0-3 as drink_box, 086_woodenblock ids 0-4, from vocab_pre_1012/objects.yaml): replays of the
    # pre-10-12 shards draw the layouts the vanilla runs saw. Names still come from the current vocabulary
    spawn_pool: str = "current"
    # The vocabulary (objects.yaml + paraphrases.yaml) of the whole process: "current" = today's files; "pre_1012" =
    # the files before 2026-10-12, which both vanilla rounds runs use (legacy_pool.install_vocab: same physics, old
    # T1 pool, old names, sampleable rules and held-out words). V6 pre-run review, 2026-10-15
    vocab: str = "current"
    # --- VLM failure function (docs/lang_filter/VLM_FAILURE_PLAN.md, Yuxuan 2026-10-26) ---
    # Where a training sample's h comes from: "sim" = the simulator's h_rules (item sidecar); "vlm:<name>" = the
    # scores of the VLM <name> (vlm_prompts.VLMS) in <root>/vlm_labels/<name>/: -1 / 0 / 1 at every labelled tick,
    # each covering the ticks nearest to it (hsource.py). Episodes without labels are not trained on. Validation and
    # the probes always use the simulator's h
    h_source: str = "sim"
    # The rules a run names (filter episodes) and samples (training and probes): "all" = every sampleable rule;
    # "vlm_prompts" = per task, the consequences main/lang/vlm_prompts.yaml has a prompt for (T1 avoid, T5 upright,
    # T6 exclude_particles; a task it does not list keeps all its rules)
    rule_consequences: str = "all"
    # Collection: save the agent-view (observer camera) frame at tick 0 and every N ticks under <root>/agent_view/ for
    # the VLM labeler (agent_view.py); 0 = off. Rendering only, the episode is the same
    agent_view_every: int = 0

    # --- sampling / relabeling ---
    shuffle_text: bool = False
    relabel: str = "paired"                        # | random | fixed
    harmful_frac: float = 0.5
    pair_frac: float = 0.5
    task_balance: bool = True
    # None: derived from gamma, round(ln 0.5 / ln gamma) = 7 at 0.9, 14 at 0.95 (review round 4, item 6); configs
    # saved with an explicit value keep it (7 before 2026-10-08)
    recall_horizon_ticks: int | None = None
    # bead-task rows (T2 / T4 / T6) of shards written before the bead-count h (2026-10-07, no
    # item_buffer/n_violating.npy) carry the old bead h: skip them (other tasks' rows are unchanged)
    drop_stale_bead_rows: bool = True
    # T6 episodes of the 2026-10-04 design (bins / plates; the item key clean_pour is shared with the 2026-10-23
    # three-bowl design) are skipped: the two designs never mix in training or validation
    drop_old_t6_rows: bool = True

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
        assert self.text_backend in ("holobrain_bert", "bert_raw", "sentence_generic", "holobrain_qwen"), \
            self.text_backend
        assert self.item_out_root, "item_out_root must never be None/empty"
        assert self.fusion_dtype in ("fp32", "tf32", "bf16"), self.fusion_dtype
        assert self.critic_readout in ("mlp", "softmin"), self.critic_readout
        assert self.critic_injection in ("linear", "revised"), self.critic_injection
        assert int(self.critic_chunk) == 1 or (int(self.critic_chunk) >= 3 and int(self.critic_chunk) % 3 == 0), \
            "critic_chunk is 1 or a multiple of 3 (three summary points)"
        assert (self.critic_injection == "linear" and int(self.critic_chunk) == 1) or self.critic_readout == "mlp", \
            "the revised injection and the chunk critic use the MLP readout"
        assert self.train_episodes_per_task >= 0, self.train_episodes_per_task
        # the frozen fusion reads HoloBrain's text_feat_map words (256-d); other backends cannot feed it
        assert not self.scene_fusion or self.text_backend == "holobrain_bert", \
            "scene_fusion needs text_backend = holobrain_bert"
        # V6: the Qwen words exist only with the Qwen front end (they read its cached prefix), and the Qwen front
        # end feeds no other words
        assert self.encoder in ("holobrain_gd", "holobrain_qwen"), self.encoder
        assert (self.text_backend == "holobrain_qwen") == (self.encoder == "holobrain_qwen"), \
            "encoder holobrain_qwen and text_backend holobrain_qwen go together"
        assert not (self.scene_fusion and self.encoder == "holobrain_qwen"), "scene_fusion is a GD-only stage"
        # V2 drops each sentence's first and last token as [CLS] / [SEP]; the Qwen words are the instruction only
        assert not (self.mask_special_tokens and self.text_backend == "holobrain_qwen"), \
            "mask_special_tokens assumes BERT's [CLS] / [SEP]"
        assert self.qwen_words_prefix in ("frame", "blank"), self.qwen_words_prefix
        assert self.qwen_attn in ("sdpa", "flash_attention_2", "eager"), self.qwen_attn
        assert self.spawn_pool in ("current", "pre_1012", "heldout"), self.spawn_pool
        assert self.vocab in ("current", "pre_1012"), self.vocab
        from .vlm_prompts import vlm_name

        vlm_name(self.h_source)   # raises on an unknown source
        assert self.rule_consequences in ("all", "vlm_prompts"), self.rule_consequences
        assert int(self.agent_view_every) >= 0, self.agent_view_every
        # a VLM-labelled run trains only rules the VLM answers for
        assert self.h_source == "sim" or self.rule_consequences == "vlm_prompts", \
            "h_source vlm:<name> needs rule_consequences = vlm_prompts"
        # the bank hash of this config's vocabulary; a hash of another known vocabulary is a stale copy (a config
        # built with dataclasses.replace from a default one), any other stored hash (a saved config) is kept
        known = {bank_hash(v) for v in ("current", "pre_1012")}
        if not self.bank_hash or (self.bank_hash in known and self.bank_hash != bank_hash(self.vocab)):
            self.bank_hash = bank_hash(self.vocab)

    @property
    def recall_ticks(self) -> int:
        """The fails_soon_recall horizon in ticks: recall_horizon_ticks, or derived from gamma when unset."""
        if self.recall_horizon_ticks is not None:
            return int(self.recall_horizon_ticks)
        return int(round(math.log(0.5) / math.log(float(self.gamma))))

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
        if "tf32_train" not in d:   # saved before the field existed: those runs trained in fp32
            kw["tf32_train"] = False
        if "actor_hidden" not in d:  # ... and with CrossSafe's linear actor head
            kw["actor_hidden"] = 0
        if "scene_fusion" not in d:  # saved before V5 (2026-10-09): unfused scene tokens
            kw["scene_fusion"] = False
        if "critic_readout" not in d:  # ... and the softmin readout (decision 6)
            kw["critic_readout"] = "softmin"
        return cls(**kw)

    @property
    def fused_words(self) -> bool:
        """The text adapter reads the fusion's words (fusion_words matters only with scene_fusion on)."""
        return bool(self.scene_fusion and self.fusion_words)

    @property
    def qwen(self) -> bool:
        """V6: the HoloBrain-Qwen front end (scene tokens 3 x 99 per frame, words from layer 0 on the prefix)."""
        return self.encoder == "holobrain_qwen"

    @property
    def scene_tokens_per_frame(self) -> int:
        """Scene tokens of one frame (all cameras): 3 x 400 GD, 3 x 99 Qwen. Shards carry the width themselves;
        this is for code that builds a model without one (eval_closed)."""
        return 297 if self.qwen else 1200

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
