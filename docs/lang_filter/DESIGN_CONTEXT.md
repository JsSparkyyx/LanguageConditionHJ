# Item-centric, language-conditioned HJ safety filter: design context

This file records the design decisions and the reasons behind them, from the planning discussions
(late September to early October 2026). Read it before changing the method, the data, or the
environments.

- Current implementation plan: `docs/lang_filter/VANILLA_PLAN.md`.
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

| task | robot's job | what can go wrong | sampleable rules per episode |
|---|---|---|---|
| T1 `selective_avoid` | the 5 CrossSafe pick-and-place hosts with 3 to 5 extra objects of different types | arm or payload touches or moves an object | `avoid` and `stay` per extra object |
| T2 `granular_pour` | pour beads from a mug into a bowl | beads miss the bowl | one `contain_particles` |
| T3 `push_zone` | push a box into a frame past mats and standing objects | box over a mat or off the table, object knocked or moved | `zone_include` (table), `zone_exclude` per mat, `upright` and `stay` per object |
| T4 `granular_carry` | carry a full bowl of beads | sloshing out at a tilt or a sharp stop | one `contain_particles` |
| T5 `tray_transport` | slide a tray with standing blocks | a fast stop tips the tall block | `upright` per block |
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
  (`language/paraphrases.yaml`: about 30 templates per consequence, split by template). Referent
  phrases must pick out exactly one item in the scene.
- Held-out splits:
  - `heldout_template`: templates never used in training;
  - `heldout_word`: referent nouns never used in training;
  - `heldout_item`: colors and categories that training rules never name, though they appear in
    training scenes.

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
- **Actor**: CrossSafe `TokenActor`, unchanged.
- **Twin critic**: CrossSafe action injection and post-action geometry, 2 more blocks with the same 5
  sub-layers, value heads on the 22 body tokens and the 4 queries, `Q = softmin_T` over those 26 values,
  T = 0.05 in h units. For contact rules a link carries the minimum. For spill, tilt, displacement and
  zones a query does.
- **Loss**: CrossSafe HJ-SAC. Target `y = (1−γ) h + γ min(h, mean of target twins at t+1)`, γ = 0.95
  (0.9 until 2026-10-08; runs keep the γ of their saved config).
  The critic loss trains the encoder, all new modules and the critics. The actor loss trains the actor
  only.
- **Frozen**: HoloBrain vision, robot state encoder, BERT, `text_feat_map`. Everything else from
  scratch.
- **Deployment**: each tick, encode cameras (history kept for t−k and 0), encode the sentence once per
  episode, engage the actor for 3 ticks when `min(Q1, Q2)(s, a_nom) < 0`.
- **Text paths**: text reaches the model in five places, each with a switch for the Q2 ablations:
  - scene-reads-words;
  - body-reads-words;
  - FiLM;
  - readout queries;
  - the rule token `r` in self-attention (switched off, `r` is a learned sentence-independent token).

  With all five off, the sentence cannot affect Q.

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
- γ = 0.95 by default (Yuxuan, 2026-10-08; `GAMMA` defaults to 0.95 in the job scripts). Runs keep the γ of
  their saved config.
- Validation: 20 ur5-wsg episodes per task, all 7 tasks, masks on, their own episode ids (91M-100M) and text
  cache (T1 / T7 in `data/lang/val_grounding`, T2-T6 in `data/lang/val20`).
- Object reading (grounding), in this order, each step only if the previous one does not fix it (T1 and T7,
  ur5-wsg, 6,000 steps, 2 seeds, scored on "already harmed" and "not yet" pairs with intervals): (a) text
  fixes V1 (a final RMSNorm on the text adapter's outputs) and V2 (V1 plus [CLS] / [SEP] hidden from the word
  attention); (b) V4, an auxiliary target loss: readout query q₁ predicts the named item's 3D centre at frames
  t and 0 through a small MLP head (Huber loss in metres, λ = 0.1 of the critic loss; labels from the sidecar
  `item_pose` / `region_T`, training only; the head is unused at deployment; `aux_target_loss`), on the better
  of V1 / V2 if either helped, else on V0; (c) the oracle tag V3 stays the ceiling reference. Decision: if a or
  b reaches ≥ 0.85 on "already harmed" referent pairs (T1 and T7 pooled), keep the vanilla model with that
  change; only if neither does but V3 does, propose slots plus a pointer (5.2).
- T6 success: left as is (its nominal succeeds in 3-7% of episodes); success cost is not reported on T6.
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
