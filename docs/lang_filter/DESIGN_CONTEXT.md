# Item-centric, language-conditioned HJ safety filter: design context

This file records the design decisions and the reasons behind them, from the planning discussions
(late September to early October 2026). Read it before changing the method, the data, or the
environments.

- Current implementation plan: `docs/lang_filter/VANILLA_PLAN.md`, with the V5 addendum
  `docs/lang_filter/FUSION_PLAN.md` (2026-10-09; the code's contract is `docs/lang_filter/fusion_impl.md`).
- Environments (implemented): `envs/docs/item_envs/` (`tasks_overview.md`, `relabeling.md`,
  `task_cards/`, `final_report.md`; plans `ITEM_SAFETY_ENVS_PLAN.md` and `PLAN.md`).

---

## 1. Thesis and motivation

- CrossSafe (the `main/` code of this repo) is a learned HJ safety filter on frozen HoloBrain-0
  encoders. Its failure function is the distance from the robot (plus held payload) to a static
  obstacle.
- **Problem with that setting:** when the safety-relevant item's state is a fixed function of the robot
  state (a static obstacle, a rigidly held object), h and its time derivative follow from known robot
  dynamics. A state-based CBF with perceived geometry (for example VLSA, arXiv 2512.11891) then beats
  learning.
- **Thesis:** safety is usually about an **item**, not the embodiment. Learning is justified when the
  item is **decoupled** from the robot: its state evolves on its own through contact, friction,
  coasting, or flow. h is easy to write on simulator state, but its dynamics are unknown and
  object-specific. A CBF can only use a conservative surrogate ("don't touch it"), which often makes the
  task impossible.
- **Role of language:** the sentence identifies which item and what must not happen to it. Numbers
  never go through a text encoder. There is no rule-type or task-type input anywhere: the model must
  read the sentence, including rephrasings.

## 2. Tasks (RoboTwin, dual-arm, `ur5-wsg` default, `franka-panda` also)

Seven tasks, built, calibrated and validated (`envs/docs/item_envs/tasks_overview.md`; demo page
"Named-Object Safety Tasks").

**Five of them are used** (Yuxuan, 2026-10-22: "we don't need that much tasks"): T1, T2, T3, T5, T7. T4 and T6
stay built and their episodes stay in the shards, but new runs train, collect and evaluate without them
(`main/lang/config.py` `TRAIN_TASKS`, the default `task_choices`; `CANON_TASKS` keeps all seven, because the
stored task ids and the evaluation seeds index it). Runs saved with all seven keep them. With T6 gone,
`exclude_particles` (beads in a bin or plate) is never named, and the beads appear only in T2.

| task | robot's job | what can go wrong | sampleable rules per episode |
|---|---|---|---|
| T1 `selective_avoid` | the 5 CrossSafe pick-and-place hosts with 2 to 3 extra objects of different types (3 to 5 before 2026-10-22) from the hand-picked pool: wine glass, milk box, apple, bottle, coloured box, coloured cylinder; a layout with no extra object has no rules and is skipped (new seed) | arm or payload touches or moves an object | `avoid` and `stay` per extra object |
| T2 `granular_pour` | pour beads from a mug into a bowl | beads miss the bowl | one `contain_particles` |
| T3 `push_zone` | push a box into a frame past mats and standing objects | box over a mat or off the table, object knocked or moved | `zone_include` (table), `zone_exclude` per mat, `upright` and `stay` per object |
| T4 `granular_carry` | carry a full bowl of beads | sloshing out at a tilt or a sharp stop | one `contain_particles` |
| T5 `tray_transport` | slide a tray with two standing blocks (tall and short; a third, middle one before 2026-10-22) | a fast stop tips the tall block | `upright` per block |
| T6 `clean_pour` | pour beads into a big bowl with small bins and plates around | beads land in a bin or plate | `exclude_particles` per bin or plate, plus the spill rule |
| T7 `gap_place` | slide a block into a narrow gap | a neighbour is scraped, moved, or knocked | `upright` and `stay` per neighbour |

- The nominal controller is careless on purpose (the environment's risk knob). In multi-item tasks the
  scene layout decides which items are **in the way** of the careless motion, deterministically per
  seed: T1 on-path objects, T3 on-path mats and objects, T5 the tall block, T6 the container the stream
  is aimed at, T7 the neighbour the slide scrapes. T2 and T4 have only the beads. Items the nominal hits
  only in some runs count as in the way too, so designations are only harm or benign (Yuxuan, 2026-10-08):
  T1 near-path objects, T1 off-path objects within 10 cm of a second corridor, and T6's diagonal container
  are harm-designated. The nominal breaks the diagonal container's rule in only about 8% of episodes, so T6
  harm runs are reported split by "aimed-at container" vs "diagonal" (`scene_roles.in_way_reasons`).
- Each task also draws a **primary** rule from a calibrated mix of in-the-way and benign scenarios (for
  example T6: the hit container 70%, another one 30%). The primary rule is a label only. Harm and benign
  designations for named rules come from "in the way".
- T1 "don't touch X" is the coupled case, where a CBF should match us. Report it as a control.
- SAPIEN has no fluids, so liquids are beads. Rolling resistance is emulated with angular damping.
- cuRobo planning is nondeterministic run to run. The same seed gives the same layout, not exactly the
  same motion.

## 3. Rules and labels (implemented in `envs/main/item_envs/`)

- A rule = (consequence, item ids, region ids, parameters).
  - Record consequence names: `avoid`, `upright`, `stay`, `zone_exclude`, `zone_include`,
    `contain_particles`, `exclude_particles`.
  - Set, pair and ambiguous rules are recorded but never named.
  - A rule is **sampleable** if it is single-referent, in the training split, and has a sentence whose
    referent phrase is unique in its scene.
- **One h scale for every rule**: `h = clip(margin / h_unit, −1, 1)`, where 1 = comfortably safe,
  0 = harm, −1 = badly harmed. The unit `h_unit` is:
  - 5 cm for distances and zones (avoid keeps a 1 cm buffer on real shapes);
  - 1 cm for "don't spill";
  - the tilt limit (10°) for `upright`;
  - the movement limit (1.5 cm in T3, 2 cm in T7, tilt counted as movement) for `stay`.

  No per-rule gains are needed.
- **Bead rules count beads once violated** (user, 2026-10-07): with n beads breaking the rule,
  `h = −(0.5 + 0.5·(min(n, 20) − 1)/19)`, so one bead reads −0.5 and 20 or more read −1, whatever
  their depth. The safe side stays `clip(margin / h_unit, 0, 1)` with a hard min over beads. Reason:
  under the plain clip, one stray bead 1 cm out and a whole spill both read −1. Example: 3 beads on
  the table give h = −(0.5 + 0.5·2/19) = −0.553. Training uses this h as stored, with no relabel to −1
  (Yuxuan, 2026-10-08).
- The sidecar buffer (lockstep with `StepBuffer`) stores, per tick, every rule's `h_rules`, its
  `fails_soon` bit (h < 0 within the next 25 ticks), raw margins, `t_in_ep`, and token masks.
  - Nominal collection records all rules at once (StepBuffer h = min over rules). Filter episodes name
    one rule. Training draws its own rule per sample either way.
  - Per-episode records (`item_buffer/episodes/{id}.json`) hold objects, regions and rules with the
    words a sentence needs.
- **Relabeling** is the training signal for language. Each sample draws a sampleable rule of its episode
  and a training paraphrase, and the target uses that rule's h.
  - A rule is **harmful** at tick t if its `fails_soon` bit is set, else **benign**.
  - Batches balance harmful and benign rules.
  - Half of each batch is **paired**: the same transition twice, with a harmful and a benign rule.
- Sentences are generated at sample time from the record and the paraphrase bank
  (`language/paraphrases.yaml`, split by template). Referent phrases must pick out exactly one item in
  the scene.
  - **Five plain everyday sentences per rule, four for training and one held out for the OOD test**
    (Yuxuan, 2026-10-20: the earlier ~30 per rule were wordy, AI-sounding phrasings; review page
    `docs/lang_filter/template_review/`). Each sentence says what its rule measures: "stay" never says
    touch / bump / knock (in T1 the same object also has "avoid", which is exactly "don't touch"); the
    spill rule never says "keep the beads in" (T2 / T6 pour them out on purpose); every bin / plate
    sentence names the beads.
  - **zone_include (keep the box on the table) is never named** (not needed, 2026-10-20): it has no
    sentences, so it is never sampleable; its h is still recorded.
  - Slot words are plain names (2026-10-20): mats "mat", bins "bin", plates "plate", the table "table",
    beads "beads" / "balls" ("little balls" is the held-out word). `language.refresh_record` re-names the
    regions of old records too.
- Held-out splits:
  - `heldout_template`: templates never used in training;
  - `heldout_word`: referent nouns never used in training;
  - `heldout_item` (Yuxuan, 2026-10-22): objects never seen in training. Each T1 object type has 3 training
    models and 1 held-out model (apple 1 + 1; `objects.yaml` `ids` / `heldout_ids`), and there are 3 training
    colours (red, green, blue) and 2 held-out ones (purple, orange). Held-out models and colours never appear in
    training scenes; the OOD scenes come from the "heldout" spawn pool (`cfg.spawn_pool`, T1: only held-out models,
    primitives in held-out colours, every rule in this split). Until 2026-10-21 this split meant categories and
    colours that training rules never named, although they stood in training scenes.
- The object corpus (Yuxuan, 2026-10-22, chosen by hand): T1 candidates are the wine glass (088_wineglass), milk box
  (038_milk-box), apple (035_apple), bottle (001_bottle), and the coloured box and tall cylinder (book and kettle
  removed the same day: they never fit on or near the arm's path, so they were never at risk; the three models
  wider than the 6 cm on-path limit, wine glass 0 and milk box 1 / 2, are shrunk to 87-90%, `objects.yaml`
  `scale_by_id`); T3 pushes only the coloured box past coloured cylinders and mats; T5 / T7
  use the coloured blocks; T2 the beads. Words: `objects.yaml`, held-out words in `paraphrases.yaml`.

## 4. Rejected approaches and why

| approach | why rejected |
|---|---|
| attention bias from GroundingDINO text-to-patch similarity | measured: 3D token positions are accurate, but text-to-patch similarity is poor |
| external detector or segmenter at runtime (GroundingDINO, YOLO, SAM) | makes the method an engineering pipeline. If the target's 3D geometry is given and the dynamics are coupled, a CBF suffices. Allowed only for offline labeling of real data and as an ablation |
| rule-type or task-type inputs (type embedding, role schema per type, type-dependent readout) | the model would not really read the sentence and could not handle rephrasings. Everything must come from the text |
| numbers through BERT | numbers tokenize badly. Thresholds are fixed per task, not stated in the sentence |
| frame differencing per token index (`c_t − c_{t−k}`) | pairs tokens by pixel cell, not by physical point. Wrong under lateral motion and on moving wrist cameras. Past frames are separate tokens with their own world positions instead |
| zero-initialized tanh gates on the new attentions | only useful for warm-starting from CrossSafe. We train from scratch on frozen HoloBrain |
| fixed learned object queries (DETR style, for the advanced version) | queries specialize to table regions. Slots must be seeded from the data |

## 5. Method

### 5.1 Vanilla model (current, being implemented)

CrossSafe plus text, trained only with the HJ-SAC losses.

- **Scene**: frozen HoloBrain tokens of frames t, t−k (k = 2 ticks) and 0 (episode start). Each frame
  goes through CrossSafe's scene adapter (shared) plus a learned time embedding. 3600 tokens, each with
  its own world position from its frame's depth.
- **Frozen fusion stage** (V5, `FUSION_PLAN.md`, 2026-10-09; switch `scene_fusion`, off by default until E5
  passes, section 8): HoloBrain's own `feature_enhancer` (6 layers of text-image fusion, 277 tensors from the
  same `pretrain/` checkpoint as the neck, the spatial enhancer and `text_feat_map`) runs frozen on the 4 neck
  maps with the sample's constraint sentence, then the spatial enhancer runs on the fused maps, in
  HoloBrain's order.
  - Both the scene tokens and their 3D positions then depend on the sentence (the spatial enhancer predicts
    depth from the fused maps). Example: in one frame, "don't touch the red mug" and "don't touch the blue
    box" give two different sets of tokens and positions; E4 measures whether the change sits on the named
    item.
  - The stage sees the constraint sentence only, never the task description, and nothing in it is trained.
  - `fusion_words` (default on, matters only with `scene_fusion`): the text adapter reads the fused words of
    frame t (t+1 for the target) instead of the cached BERT words. Off, the fusion changes only the scene
    side (the one new ablation).
  - Precision: one `fusion_dtype` for every place the stage runs (training, evaluation, rollout), fp32 by
    default; tf32 or bf16 only after the bench and the parity numbers.
  - Data: needs version-2 shards (`neck_buffer/`, the neck maps per tick); a version-1 shard with
    `scene_fusion` on is an error. Sentences are fused unpadded, grouped by length, because HoloBrain's
    forward does not mask padded words.
- **Robot**: CrossSafe `RobotInjection`, unchanged (22 tokens: 20 links + 2 payload, positions,
  Jacobians).
- **Text**: frozen HoloBrain BERT + `text_feat_map` (24 word tokens, 256-d, cached), then a 2-layer
  self-attention adapter over [r, q₁..q₄, words + position embedding]. Outputs: `r₀` (sentence
  summary), 4 readout queries, adapted words `w'`.
- **Scene reads words once** (cross-attention, plain residual), then scene tokens are read-only.
- **Trunk, 3 blocks**, each with 5 pre-norm residual sub-layers:
  - (a) body reads scene with CrossSafe's geometric bias;
  - (b) body reads words;
  - (c) queries read scene (no geometric bias);
  - (d) self-attention over [body, r, queries];
  - (e) FiLM from `r₀` on body and queries, then FFN.
- **Actor**: CrossSafe `TokenActor` (per joint token: a tanh-squashed Gaussian, shared over joints) with an
  MLP head, RMSNorm → Linear(256, 256) → SiLU → Linear(256, 2), about 66k weights (Yuxuan, 2026-10-08;
  `actor_hidden`, `network.LangTokenActor`). CrossSafe's head was one Linear(256, 2), 514 weights, and the
  actor loss trains only the actor on detached encoder features, so the head is the policy's only
  nonlinearity. Configs saved before the field keep the linear head.
- **Twin critic**: CrossSafe action injection and post-action geometry, 2 more blocks with the same 5
  sub-layers, then a readout to one Q per twin (`critic_readout`):
  - "mlp", the default (decision 6, Yuxuan 2026-10-09): one shared RMSNorm on r, q₁..q₄ and the masked mean
    of the 22 body tokens (padding excluded), concatenated (6 × 256 = 1536), then Linear(1536, 512) → SiLU →
    Linear(512, 1) = Q. Reason: under the softmin, nearly all of a sample's gradient goes to the one token
    that holds the minimum, so the other 25 heads learn only where they hold it; the MLP gives every sample a
    dense gradient and does not assume which token owns the rule.
  - "softmin", the ablation: value heads on the 22 body tokens and the 4 queries, `Q = softmin_T` over those
    26 values, T = 0.05 in h units. For contact rules a link carries the minimum. For spill, tilt,
    displacement and zones a query does. Configs saved before the field keep the softmin.
- **Loss**: CrossSafe HJ-SAC. Target `y = (1−γ) h + γ min(h, mean of target twins at t+1)`, γ = 0.95
  (0.9 until 2026-10-08; from 2026-10-09 no γ 0.9 run is started, continued or used as a baseline; runs
  keep the γ of their saved config).
  The critic loss trains the encoder, all new modules and the critics. The actor loss trains the actor
  only.
- **Frozen**: HoloBrain vision, robot state encoder, BERT, `text_feat_map`, and with `scene_fusion` the
  fusion stage (built from the checkpoint, kept outside the trained modules and their target copy).
  Everything else from scratch.
- **Deployment**: each tick, encode cameras (history kept for t−k and 0), encode the sentence once per
  episode, engage the actor for 3 ticks when `min(Q1, Q2)(s, a_nom) < 0`. With `scene_fusion`, each new
  frame is fused once with the episode's sentence (twice in monitor mode, the measuring mode that scores a
  second, counterfactual sentence every tick and never engages; that sentence keeps its own history) and the
  history keeps fused frames, so the sentence must be set before the first frame.
  With `fusion_words` the text adapter runs every tick, because the fused words change with the frame
  (1.6 M parameters on 29 tokens, small next to the fusion pass).
- **Text paths**: text reaches the model in five places, each with a switch for the Q2 ablations:
  - scene-reads-words;
  - body-reads-words;
  - FiLM;
  - readout queries;
  - the rule token `r` in self-attention (switched off, `r` is a learned sentence-independent token).

  With all five off, the sentence cannot affect Q. With `scene_fusion` on it still can, through the fused
  scene tokens and positions (a sixth path, not behind these switches).

### 5.2 Advanced version (planned after the vanilla results)

Object slots plus an entity pointer, supervised with simulator labels, still with **no rule or task
types**:
- **Slots**: objectness-seeded slot attention over the scene tokens. Slot geometry (mean, covariance) and
  motion come from re-assigning frames t−k and 0.
- **Entity pointer**: entity queries in the text adapter compose referring phrases. Each slot gets a tag
  (sigmoid of its best match to an entity) and the matched phrase embedding.
- **Where tags act**: they bias link-to-scene attention, tagged slots join self-attention, and the
  readout is a softmin over links and tagged slots.
- **Extra losses**: slot grouping (Hungarian matching to the sidecar token masks) and pointer BCE (which
  slots the sentence names). The pointer fixes the referent only; the consequence stays learned from
  text by the HJ loss.
- It is motivated only if the vanilla results show referent grounding as the bottleneck (Q2).

## 6. Research questions

| id | question | evidence |
|---|---|---|
| Q1 | Does HJ loss plus counterfactual relabeling teach the filter to read the sentence? | paired vs random relabeling vs one fixed rule per episode vs shuffled sentence |
| Q2 | Where must language enter? Hypothesis: the referent needs early token-level fusion, the consequence needs global modulation | text-path ablations (five switches), scored separately on referent decisions (same consequence, different item) and consequence decisions (same item, different consequence, as in T1, T3, T7) |
| Q3 | Which text space: vision-aligned HoloBrain BERT or a generic sentence encoder? | text backend swap |
| Q4 | Which generalization axis fails first? | held-out templates, held-out words, held-out items |

Metrics:
- **offline**: pairwise rank accuracy of Q between harmful and benign rules at the same state, AUROC,
  paraphrase consistency, attention enrichment on the named item (ground-truth masks used only to
  measure).
- **closed loop**:
  - named rule = a rule whose item is in the way (harm runs) or one whose item is not (benign runs).
    This comes from the scene layout, because cuRobo makes outcomes differ between runs of one seed;
  - protection rate relative to repeated nominal runs, intervention and false-alarm rates when the
    named item is not at risk, success cost;
  - every result reported per language axis.

## 7. Real world (later)

Train in simulation and deploy zero-shot first. If needed, fine-tune with real frames labeled offline
(click plus SAM 2 tracking). A segmenter is a labeling tool, never part of the deployed filter.
Hardware: SO-101 or the UR5 setup, to be decided.

## 8. Defaults and open decisions

- k = 2 ticks for the past frame. T = 0.05. Batch 64, lr 3e-4. TF32 matmuls in the training step only
  (user, 2026-10-08; `tf32_train`), fp32 everywhere else.
- γ = 0.95 only (FUSION_PLAN decision 1, Yuxuan 2026-10-09): no run at 0.9 is started, continued or
  reported as a baseline; the job scripts pass 0.95 (no `GAMMA` variable). The γ 0.9 models, the grounding
  report's official `lang_full_paired_off_1007` included, are historical references only. Runs keep the γ of
  their saved config, so old checkpoints still load.
- Validation: 20 ur5-wsg episodes per task, all 7 tasks (T4 / T6 skipped from 2026-10-22), masks on, their own episode ids (91M-100M) and text
  cache (T1 / T7 in `data/lang/val_grounding`, T2-T6 in `data/lang/val20`).
- Object reading (grounding), in this order, each step only if the previous one does not fix it (T1 and T7,
  ur5-wsg, 6,000 steps, 2 seeds, scored on "already harmed" and "not yet" pairs with intervals): (a) text
  fixes V1 (a final RMSNorm on the text adapter's outputs) and V2 (V1 plus [CLS] / [SEP] hidden from the word
  attention); (b) V4, an auxiliary target loss: readout query q₁ predicts the named item's 3D centre at frames
  t and 0 through a small MLP head (Huber loss in metres, λ = 0.1 of the critic loss; labels from the sidecar
  `item_pose` / `region_T`, training only; the head is unused at deployment; `aux_target_loss`), on the better
  of V1 / V2 if either helped, else on V0; (c) the oracle tag V3 stays the ceiling reference. Decision: if a or
  b reaches ≥ 0.85 on "already harmed" referent pairs (T1 and T7 pooled), keep the vanilla model with that
  change; only if neither does but V3 does, propose slots plus a pointer (5.2). Outcome
  (`grounding_report_2026-10-09.md`): a and b fail (V0, V1, V2 0.59, V4 0.60), V3 is ignored (0.58), and the
  name → object map is memorized per training scene.
- V5, the next step (`FUSION_PLAN.md` decisions 2-6, Yuxuan 2026-10-09; the code's contract is
  `fusion_impl.md`):
  - slots plus a pointer stay on the shelf: the report's evidence (V3 ignored, V4 memorized) does not
    support them. The next step adds a pretrained grounding stage instead (5.1, frozen fusion stage);
  - the fusion stage sees the constraint sentence only, the same string the text adapter sees;
  - the grounding gate stays: ≥ 0.85 rank accuracy on "already harmed" object pairs, T1 + T7 pooled.
    "Not yet" pairs are reported, not gating. V5 runs are scored on the re-collected version-2 validation
    set (the report's validation shards hold tokens, not neck maps), with the old V0 checkpoints scored on
    it too to link the two sets;
  - the scene-count curve (E6: about 15, 30, 60 and 120 scenes = 8 and 15 episodes per task, the report's
    V0 and the d2 run; V0 settings with the softmin readout, version-1 data) runs in parallel with V5. It separates "too few scenes" from "the architecture cannot ground";
  - the critic's readout is an MLP by default (5.1); E5 isolates its own effect with one extra row
    (V0'-softmin).
- V5 order (2026-10-09): code → version-2 validation collection (T1 / T7 ur5-wsg, 40 episodes, the recorded
  seeds replayed) → E4, no training: does the frozen fusion single out the named item → only if E4 passes,
  the version-2 warmup (the 60 recorded layouts replayed) → E5 (6,000 steps, 2 seeds). E6 runs now, on
  version-1 data.
  - E4 gate, on T1: enrichment ≥ 2 and contrast ≥ 0.7. A token's change is the distance between its fused
    value under the rule's sentence and under the content-free sentence ("" → [CLS][SEP]). Enrichment = the
    mean change over the named item's tokens ÷ the mean over the other tokens. Contrast = for two rules of
    the same consequence in one scene, the share of the named item's tokens that change more under its own
    sentence than under the other one. Example: enrichment 3 means the mug's tokens move 3× as much as the
    rest of the scene under "don't touch the red mug".
  - E5 rows: V0' (fusion off on the version-2 shards, the reference), V5 (fusion and fused words on), V5w
    (fused words off), V5-d2 (V5 on both robots' T1 / T7, 120 scenes), V0'-softmin (V0' with the softmin
    readout).
  - E5 decision rule: V5 ≥ 0.85, adopt (re-collect the full warmup in version 2, retrain the official model,
    turn `scene_fusion` on by default); clearly above V0' but below 0.85 and V5-d2 > V5, collect more T1 / T7
    scenes and re-test; at the V0 level with E4 passed, the next lever is the data, not the architecture.
    Until then `scene_fusion` stays off by default, because every shard today is version 1.
- T6 success: left as is (its nominal succeeds in 3-7% of episodes); success cost is not reported on T6.
  (T6 is no longer used from 2026-10-22.)
- Projection solver for deployment: deferred.
- Novelty check needed against:
  - object-centric RL;
  - language-conditioned latent safety filters (arXiv 2608.00315, closest prior work);
  - latent safety filters (Nakamura et al., RSS 2025).
- Target venue: RSS.

## 9. Lineage of ideas

CrossSafe for the safety backbone. FiLM for conditioning. Hindsight Experience Replay for relabeling.
Flamingo and Perceiver-style learned query tokens for the text adapter and readout queries. For the
advanced version: Slot Attention, SAVi, DETR set matching, two-stage Deformable DETR and DINO-DETR query
selection, MDETR, GLIP and GroundingDINO alignment, Pointer Networks.
