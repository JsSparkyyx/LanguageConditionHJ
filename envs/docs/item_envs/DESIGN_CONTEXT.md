# Item-centric HJ safety filter: design context

This file records the design decisions, and the reasons behind them, from the planning discussion
(late Sep to early Oct 2026). Read it before changing the method or the environments. The
implementation plan for the environments is `docs/item_envs/ITEM_SAFETY_ENVS_PLAN.md`.

---

## 1. Thesis and motivation

- CrossSafe (this repo) is a learned HJ safety filter on frozen HoloBrain-0 encoders. Its failure
  function is the distance from the robot (plus held payload) to a static obstacle.
- **Problem we identified:** when the safety-relevant item's state is a fixed function of the robot
  state (a static obstacle, a rigidly held object), the failure value h and its time derivative are
  computable from known robot dynamics. A state-based CBF with perceived geometry (for example VLSA,
  arXiv 2512.11891) then beats learning. Learning is not justified there.
- **Thesis:** safety is usually about an **item**, not the embodiment. Learning is justified when the
  item is **decoupled** from the robot: its state evolves on its own through contact, friction,
  coasting, or flow, so h is easy to write on simulator state but its dynamics are unknown and
  object-specific. A CBF can then only use a conservative surrogate ("don't touch it"), which often
  makes the task impossible.
- **Role of language:** it only identifies the referent (which item, which region). Numbers never go
  through a text encoder. Rule types are a small closed set. A filter conditioned on a referent can
  be prompted by text, a click, or a ground-truth mask. The language claim is open-vocabulary
  referent selection, not language understanding.

## 2. Tasks (simulation, RoboTwin, dual-arm)

Robots: dual-arm setups only (HoloBrain is bimanual). Default `ur5-wsg` pair, also `franka-panda` pair.

| ID | Task | Failure | Why the referent matters |
|---|---|---|---|
| T1 | `selective_avoid`: the 5 CrossSafe host tasks with 3 to 5 extra objects on and near the path | robot or held payload touches a **specified** object. Non-specified objects may be pushed aside | only some objects are protected |
| T2 | `granular_pour`: from `place_container_plate`, a bowl of beads poured into a target container | beads leave the containers, or land on a specified forbidden mat | several regions, only one forbidden |
| T3 | `push_zone`: both arms push objects toward goals near the table edge and colored zones | pushed object leaves the table or enters the specified zone, or a specified bystander tips or moves | several zones and bystanders |

Notes:
- T1 as "robot must not touch X" is the coupled case, where a CBF should match or beat us. Report it
  honestly as a control. A decoupled variant protects X's state (`stay`, `upright`) while the robot
  may push other objects, so chained contact becomes the failure.
- Liquids are approximated by beads (SAPIEN has no fluid support exposed). Rolling resistance must
  be emulated with angular damping. Dynamic containers need decomposed (multi-convex) collision.
- RoboTwin's `Actor` sets every object to 10 g. Masses must be set explicitly.

## 3. Rule representation

A rule = (type, subject, optional reference, numeric parameters).

| type | subject | reference | margin (positive = safe) |
|---|---|---|---|
| avoid | object | none | robot + payload distance to the object |
| upright | object | none | θ_max − tilt |
| stay | object | none | d_tol − displacement since rule start |
| inside | object | region | min footprint distance inside the region |
| outside | object | region | min footprint distance outside the region |
| particles_inside | bead set | containers | soft-min over beads of the inside distance |
| particles_outside | bead set | region | soft-min over beads of the outside distance |

Rules are relabeled at training time: for each stored transition a random valid rule is drawn, and its
margin is read from the per-rule margins the environment logged.

## 4. Rejected approaches and why

| approach | why rejected |
|---|---|
| attention bias from text-to-scene-token similarity (HoloBrain GD fusion) | measured: 3D token positions are accurate, but text-to-patch similarity is poor. Fine-tuning the language side through the HJ loss gives a weak, sparse signal |
| external detector or segmenter at runtime that hands the filter the target (GroundingDINO, YOLO, SAM) | makes the method an engineering pipeline. If the target's 3D geometry is given and dynamics are coupled, a CBF suffices. Allowed only as an offline labeling tool for real data, and as an ablation |
| rule type or numbers through BERT | numbers tokenize badly and BERT has no magnitude sense. Rule types are a closed set, so a lookup table plus FiLM suffices |
| fixed learned object queries (DETR style) | queries specialize to table regions, merging nearby objects. Slots must be seeded from the data |
| treating item and region with different machinery | replaced by one role pointer with roles subject and reference |

## 5. Architecture

CrossSafe stays the backbone: per-link tokens for actor and critic, geometric link-to-scene
cross-attention, Jacobian action injection, post-action geometry, softmin readout, HJ-SAC training.
New parts are object slots and a role pointer.

**Step 1, frozen HoloBrain.** 3 RGB-D cameras give 1200 scene tokens c_j (256-d, one per 16×16 or
32×32 pixel cell) with accurate 3D positions p_j. Joint angles give robot state tokens via forward
kinematics.

**Step 2, CrossSafe injection (unchanged).** Link tokens = robot state token + 16 analytic features
(velocity, chain depth, arm id, gripper opening) through an MLP, plus 2 payload tokens at the
grippers. Jacobians J_i per link. Scene adapter: linear + RMSNorm on c_j.

**Step 3, slot attention (new).** Groups the 1200 tokens into S = 10 object slots. Slots are not
parameters. They are computed per frame, like soft k-means centers:
1. an objectness head scores each token as being on an object or not,
2. the top-scoring tokens, kept at least 3 cm apart in 3D, seed the slots,
3. three iterations of: tokens split themselves across slots (softmax over slots), then each slot
   becomes the weighted average of its tokens' features, through a GRU,
4. slot position = weighted average of its tokens' p_j. Slot velocity = change since the previous
   tick (slots are propagated across ticks).

**Step 4, role pointer (new).** Frozen BERT encodes the rule text. Each role of the rule type (subject,
and reference if present) has a learned query that attends over the words, then scores every slot.
A sigmoid gives tags: a 10 × 2 table of how much each slot is the subject and the reference. A missing
role stays zero. Sigmoid (not softmax) allows set referents.

**Step 5, trunk, 3 blocks (CrossSafe TrunkBlock, extended).**
1. link-to-scene geometric cross-attention (CrossSafe distance and direction bias) plus a new bias
   λ_h · log m_j on half the heads, where m_j = Σ_s tag_subject[s] · A[s, j] is how much scene token j
   belongs to the subject,
2. self-attention over links **and** tagged slots, with 3D-distance bias (CrossSafe had links only),
3. per-token FFN, with FiLM from a learned rule-type embedding.

**Step 6, actor (CrossSafe, unchanged).** Per-joint Gaussian head on actuated link tokens, tanh,
scaled by the per-tick joint limit, scattered by joint index.

**Step 7, twin critic (CrossSafe, readout extended).** Action effect per link dp_i = J_i a and dω_i,
added to link tokens. Link positions shifted to x_i + dp_i. Two more trunk blocks. Value head on every
link token and every slot token. Readout:
- avoid: softmin over link values (CrossSafe),
- all other rule types: value of the subject slot (tag-weighted).

Q > 0 means safe. The filter engages when Q(s, a_nominal) < 0 (CrossSafe convention, κ = 20 per metre).

## 6. Losses

Simulator labels used (training only): per-object token masks M[o, j] (fraction of token j's pixel
cell showing object o, from the segmentation render), and each rule's object ids and margins.

1. **Matching:** soft IoU between each slot's tokens (assignment A) and each object's tokens (M), then
   the Hungarian algorithm pairs objects with slots.
2. **loss_slot** = cross-entropy pushing each object token (M ≥ 0.5) into its matched slot, plus
   binary cross-entropy on objectness (target: token shows any object). Background tokens skipped.
3. **loss_ptr** = binary cross-entropy between predicted tags and target tags (1 at the matched slot of
   the rule's subject, and of its reference if present), only over roles the rule type has.
4. **loss_critic** = (Q1 − y)² + (Q2 − y)², with y = (1 − γ) h_t + γ min(h_t, mean of the target critics
   at t+1 with the actor's next action), h_t = κ · margin_t, γ = 0.9 (CrossSafe HJ-SAC).
5. **loss_actor** = −min(Q1, Q2)(s, a_π) + α log π + 0.001 · (pre-tanh output)².

Training details:
- the critic and actor use ground-truth tags (teacher forcing), with a growing fraction (0 to 50%) of
  predicted tags, gradients blocked into the pointer,
- the pointer is trained only by loss_ptr (ablation: let HJ gradients reach it),
- HoloBrain and BERT stay frozen. Target critics use Polyak averaging. α is annealed,
- per-task training and co-training of all three tasks (balanced sampling per task and rule type).

## 7. Deployment

Each tick, steps 1 to 7 on live cameras plus the rule text, no simulator labels and no external
models. If Q(s, a_nominal) < 0, the actor's action runs for 3 ticks (CrossSafe filter).

## 8. Real world

- First, train fully in simulation and deploy zero-shot.
- If grouping or pointing fails: fine-tune slots and pointer on real frames labeled offline (click
  each object once, SAM 2 tracks it). The segmenter is a labeling tool, never part of the deployed
  filter.
- Critic fine-tuning only if needed, on AprilTagged rigid objects (T1, T3). Beads stay sim-only.
- Runtime segmenter variant (SAM 2 tiny ~21 ms per frame on A100, per camera; Grounding DINO 1.5 Edge
  ~13 ms with TensorRT) is an ablation and fallback, run asynchronously at 2 to 5 Hz.

## 9. Open questions and first checks

1. **Text-to-object retrieval check** before HJ training: pool frozen features over ground-truth object
   tokens, measure text-to-object retrieval among the objects in a scene, on seen and held-out
   categories. Above about 90% means the pointer is easy. Much lower means held-out generalization is
   the main risk.
2. Slot grouping limits: objects closer than about 2 to 3 cm may merge at stride-16 token resolution.
3. Visible-surface bias: slot centroids are pulled toward the cameras. Optional learned center and
   extent correction supervised in simulation.
4. Novelty check needed against object-centric RL (entity-centric RL, slot-based dynamics models),
   language-conditioned latent safety filters (arXiv 2608.00315, closest prior work), and latent safety filters
   (Nakamura et al., RSS 2025).
5. Target venue: RSS. Hardware: SO-101 or the UR5 setup, to be decided.

## 10. Lineage of ideas

Slot Attention (Locatello 2020), SAVi (Kipf 2022) for temporal slots, DETR (Carion 2020) for Hungarian
set matching, two-stage Deformable DETR and DINO-DETR for data-dependent query selection, Mask2Former
for masked attention, MDETR / GLIP / GroundingDINO for text-to-object alignment, Pointer Networks,
Hindsight Experience Replay for relabeling, scheduled sampling for teacher forcing, FiLM for
conditioning, CrossSafe for the safety backbone.
