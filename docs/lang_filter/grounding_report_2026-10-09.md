# Grounding experiments: results and conclusions (2026-10-09)

For review. This report collects every result of the grounding experiments (review round 3, Part II;
Yuxuan's decisions of 2026-10-08) and states what we conclude from them. The working log with the raw
tables is `docs/lang_filter/grounding_1008.md`; the per-run table is `runs/lang/grounding_1008/summary.md`.

Status: everything below is final except four items still queued (section 7): V4b scoring, V0 trained on
both robots' data, two training-scene checkpoints, and the official model on T3 / T5.

---

## 1. Summary

- **The question.** The official model reads the kind of rule (0.75) but barely the object the sentence
  names (0.57, against 0.56 for a reader that sees only the words). The slots-plus-pointer model (design
  §5.2) is worth building only if finding the named object is the bottleneck. Decision 5 fixed the order of
  cheap fixes to try first.
- **The cheap fixes do not work.** On the main score (object pairs at already-harmed moments, tasks 1 and 7,
  bar 0.85), V0 0.59, V1 0.59, V2 0.59, V4 0.60, all within each other's intervals and far from 0.85.
- **The oracle is not a ceiling.** V3, which is given the named object's location, scores 0.58 and ignores
  the location: zeroing it changes 1.7% and 0.9% of its answers.
- **The model memorizes its training scenes.** The same score on the training episodes is 0.96-0.97 (V0,
  V1, V3) and 0.93 (official), against 0.56-0.63 on new episodes. On training episodes it also predicts which
  object is about to be hit far better than the current h does, which only memory of the episode can explain.
  The new-scene score is flat from 1,000 training steps on.
- **Even direct location supervision is memorized.** V4's head, trained to predict the named object's 3D
  centre, is off by 19 cm (T1) and 6 cm (T7) on new scenes, no better than pointing at the middle of the
  scene's objects (18 and 5 cm); on training batches it is off by 6 cm.
- **Outcome of decision 5.** Step a fails, step b fails, V3 is no ceiling. Neither branch of the rule fires,
  and nothing here supports building slots plus a pointer.
- **Leading explanation:** too few distinct training scenes (60 T1 / T7 episodes), not the text path or the
  architecture. This is a hypothesis, not a result: the official model already trained on 120 T1 / T7 scenes
  (two robots, plus five other tasks) and still scores 0.57. A scene-count curve is the test we propose
  (section 8).

## 2. The question and the plan

The official model is `runs/lang/lang_full_paired_off_1007`, step 20,000. It was trained on all 7 tasks,
both robots (ur5-wsg and franka-panda), γ 0.9. On tasks 1 and 7 it orders rule pairs as follows (section
4.1): pairs that differ in the kind of rule 0.75, pairs that differ in the object 0.57, while the words-only
reader gets 0.56 on the object pairs.

Two explanations lead to different next steps:
- The text side fails to deliver the object's name, or the model cannot find the named object in the scene.
  Then text fixes or object supervision should help, and if only an oracle location helps, slots plus a
  pointer are justified.
- The model can tie name to object, but learns it per training scene and does not generalize. Then the
  architecture is not the problem.

Decision 5 (Yuxuan, 2026-10-08): try (a) V1 / V2, then (b) V4, with V3 as the ceiling reference. If a or b
reaches ≥ 0.85 on already-harmed object pairs (T1 and T7 pooled), keep the vanilla model with that change.
Only if neither does but V3 does, propose slots plus a pointer.

## 3. Setup

### 3.1 Data

- **Training data for the short runs (E2):** T1 and T7 ur5-wsg episodes of `data/lang/warmup`: 30 + 30
  episodes, 212 + 150 sampleable rules.
- **Validation (E1 plus earlier):** 40 new ur5-wsg episodes, 20 per task (`collect validation --stream
  grounding`, episode ids from 91M, masks on, own text cache `data/lang/val_grounding/text_cache.pt`). 38
  are usable: two task-1 episodes threw on their first tick because the host task found no grasp pose. With
  the earlier ur5 validation episodes this gives 23 T1 and 25 T7 episodes. None of them was trained on.
- Validation episodes are nominal runs: the careless planner moves and the filter is off. Every tick records
  the h of every rule in the scene and its `fails_soon` bit (h < 0 within the next 25 ticks, 1 s).

### 3.2 The pair score

- **States:** every 5th tick of every validation episode. States whose 25-tick window runs past a
  time-capped episode end are dropped.
- **Pair:** two sampleable rules at the same state, one **harmful** (`fails_soon` set) and one **benign**
  (not set). Each rule gets a training-template sentence.
- **Scored quantity:** Q(s, a | sentence), where a is the action logged at that tick (the nominal action).
- **Correct** if Q(harmful) < Q(benign); a tie counts 0.5. **Rank accuracy** = mean over pairs; 0.5 is chance.
- **Kinds of pair:**
  - object (referent) pairs: same consequence, different object. Example (T1): a red mug on the arm's
    path, a blue bottle off it; "don't touch the red mug" vs "don't touch the blue bottle".
  - rule-kind (consequence) pairs: same object, different consequence. Example (T7): the slide shoves a
    neighbour 3 cm without tipping it; "don't move the block" (harmful) vs "don't knock over the block"
    (benign).
  - Pairs that differ in both are in neither group.
- **Moments:**
  - already harmed: the harmful rule's h < 0 at this tick. Ranking the pair by the current h gives 1.00 by
    construction, so this tests whether the model reads the named object's present state.
  - not yet: the harmful rule's h first drops below 0 in 1-24 ticks.
- **Main score (decision 5):** object pairs, already harmed, T1 and T7 pooled: 1,735 pairs from 41 episodes.
  Every run is scored on exactly these pairs.
- **Words-only reader ("prior"):** a logistic regression on the set of words in a sentence, fit on the
  training data to predict how often that sentence's rule is about to be broken; it never sees the scene. A
  pair is ranked by the higher predicted probability. It measures how much ordering the words alone carry.
- **Seeds and intervals:** for a version with two seeds, each pair's two outcomes are averaged, then the
  mean is taken. [ ] is a 95% percentile interval from 1,000 bootstrap resamples of whole episodes (pairs of
  one episode are correlated). The interval covers episode variation only, not seed variation.

### 3.3 Runs

All E2 runs: 6,000 steps, batch 64, lr 3e-4, γ 0.95, linear actor head (`actor_hidden 0`), TF32 in the
training step, T1 + T7 ur5 training data, seeds 0 and 1. `grounding_summary.py` refuses any run whose
settings or scored set differ from V0's.

| run | change from V0 | code |
|---|---|---|
| V0 | none | e9f1d98 |
| V1 | RMSNorm on the text adapter's outputs (`text_out_norm`) | e9f1d98 |
| V2 | V1, plus [CLS] and [SEP] hidden from the word attention (`mask_special_tokens`) | e9f1d98 |
| V3 | oracle tag: a learned vector e_tag (zero-initialized) added to each scene token j before scene-reads-words, with weight w_j = exp(−d_j²/2σ²), d_j = distance from token j's 3D position to the named object's centre in that token's frame, σ = 4 cm (`oracle_ref_tag`) | e9f1d98 (s1: a5b6055, export fix only) |
| V4 | auxiliary target loss: readout query q₁ → MLP → the named object's centre at frames t and 0, Huber in 5 cm units, weight λ calibrated (section 4.9) (`aux_target_loss`) | 0e97772 |
| V4b | V4, plus an attention loss: q₁'s head-averaged scene attention in the last encoder block, renormalized within frames t and 0, times the squared distance to the named centre (`aux_attention_loss`) | 0e97772 |

## 4. Results

### 4.1 E0: the official model on tasks 1 and 7

Training-template sentences, 41 validation episodes. Prior = words-only reader on the same pairs;
model − prior is paired.

| pairs | moments | model (pairs) | prior | model − prior |
|---|---|---|---|---|
| object | all | 0.57 [0.48, 0.67] (2616) | 0.56 [0.48, 0.64] | 0.01 [−0.10, 0.15] |
| object | already harmed | 0.57 [0.45, 0.71] (1735) | 0.55 [0.47, 0.64] | 0.02 [−0.11, 0.19] |
| object | not yet | 0.57 [0.51, 0.65] (881) | 0.58 [0.49, 0.67] | −0.01 [−0.11, 0.11] |
| rule kind | all | 0.75 [0.66, 0.84] (779) | 0.56 [0.50, 0.61] | 0.19 [0.08, 0.31] |
| rule kind | already harmed | 0.79 [0.68, 0.89] (589) | 0.59 [0.51, 0.65] | 0.20 [0.07, 0.32] |
| rule kind | not yet | 0.62 [0.49, 0.77] (190) | 0.47 [0.42, 0.53] | 0.15 [0.01, 0.31] |

Other sentence sets, same model (object / rule kind): held-out templates 0.57 / 0.63, held-out nouns 0.55 /
0.68, shuffled sentences (control, 0.5 by construction) 0.49 / 0.52.

**Reading:** the model reads the kind of rule clearly above the words alone; it reads the object no better
than the words alone.

### 4.2 E0b: does Q track the named object or any endangered object?

Per (state, rule) of avoid / stay / upright rules: ρ_named = Spearman(Q, the named rule's current h);
ρ_any = Spearman(Q, the lowest current h among the scene's rules of the same consequence, named or not).
"any − named" is the paired difference.

| task | moments | rows (episodes) | ρ_named | ρ_any | any − named |
|---|---|---|---|---|---|
| T1 | all | 7202 (23) | 0.21 [0.14, 0.28] | 0.33 [0.23, 0.41] | +0.12 [0.02, 0.20] |
| T1 | already harmed | 1889 (23) | 0.18 [0.04, 0.35] | 0.03 [−0.11, 0.19] | −0.15 [−0.34, 0.02] |
| T1 | not yet | 969 (23) | 0.13 [0.04, 0.22] | 0.29 [0.18, 0.42] | +0.17 [0.03, 0.29] |
| T7 | all | 8206 (25) | 0.26 [0.16, 0.35] | 0.33 [0.21, 0.47] | +0.07 [0.02, 0.14] |
| T7 | already harmed | 827 (18) | 0.18 [−0.11, 0.51] | 0.20 [0.00, 0.41] | +0.02 [−0.27, 0.39] |
| T7 | not yet | 282 (18) | 0.11 [−0.02, 0.29] | 0.01 [−0.14, 0.18] | −0.10 [−0.23, 0.05] |

**Reading:** on T1, Q follows the most endangered object more closely than the named one (+0.12 overall,
+0.17 before contact). On T7 the overall difference is +0.07; per moment it is inconclusive. All correlations
are weak (≤ 0.33): the model sees danger weakly and ties it to the named object even less. It behaves mostly
like a general "something is about to be hit" detector.

### 4.3 E0c: word attention

`scripts/lang/word_attention.py`, 300 states per task.
- The frozen HoloBrain text encoder uses its sub-sentence mask. The [CLS] feature is the same vector for
  every sentence (400 sentences: cosine to the mean 1.0000). [SEP] is not constant (relative spread 0.28): it
  attends to the whole sentence and works as a summary.
- [CLS] and [SEP] have larger keys than words (RMS 11 vs 6). [CLS] takes 24% of encoder block 1's word
  attention (2.2× its share), 17% in the first critic's last block (1.5×) and 48% in the second critic's
  (4.3×).
- Attention per head is sharp: entropy 0.03-0.11 nats per (head, query), against 2.2 for uniform. Each head
  reads essentially one word.
- The named object's words get more than their share where the body first reads the words: encoder block 1
  2.1× (T1 2.5×), the critics' first blocks 1.4-1.8×, scene-reads-words 1.5×.

**Reading:** the model does attend to the object's words. The visible defect is attention spent on a
content-free [CLS], which V1 and V2 target.

### 4.4 E0a / E3: two sentences along one run

Six validation scenes (the first 3 per task by seed with a harm-designated and a benign-designated rule of
the same consequence). Each scene runs once, nominal motion, filter in monitor-only mode, recording Q(s, a_nom)
for sentence A (harm-designated rule) and sentence B (benign) every tick. Official model (E0a) and V1-V3
seed 0 (E3). Videos: `runs/lang/grounding_1008/videos/`.

Cell: share of ticks with Q_A < Q_B while A's h < 0 (number of such ticks); then mean Q_A − Q_B over the
whole run (negative = right direction).

| scene | A / B | official | V1 | V2 | V3 |
|---|---|---|---|---|---|
| T1 s91000002 | "keep a safe distance from the drink box" / "don't bump into the blue tall cylinder" | 0.00 (2); −0.19 | 1.00 (2); −0.45 | 1.00 (2); −0.27 | 1.00 (1); −0.39 |
| T1 s91000004 | "keep a safe distance from the coffee box" / "don't touch the cup" | 1.00 (2); −0.48 | 1.00 (2); −0.48 | 1.00 (2); −0.39 | 1.00 (2); −0.67 |
| T1 s91000007 | "don't run into the tin" / "go around the green box" | 1.00 (4); −0.08 | 0.00 (4); +0.14 | 0.33 (3); −0.07 | 1.00 (2); −0.15 |
| T7 s91000001 | "the tall green block should stay where it is" / "make sure the blue block doesn't move" | 1.00 (47); −0.18 | 0.75 (48); +0.03 | 0.36 (45); +0.05 | 0.78 (45); −0.17 |
| T7 s91000003 | "don't slide the short green brick" / "don't nudge the blue block" | 0.48 (50); −0.01 | 0.34 (50); +0.03 | 0.30 (50); +0.02 | 0.00 (50); −0.01 |
| T7 s91000005 | "don't shove the red block" / "the blue block should stay where it is" | 0.62 (47); −0.01 | 0.00 (48); +0.58 | 0.00 (48); +0.54 | 0.00 (48); +0.59 |

**Reading:** on T1 the run mean points the right way in 11 of 12 cells; on T7 in 5 of 12, mostly near zero,
and in s91000005 V1-V3 rate the shoved block's sentence clearly safer for the whole harm window. Six scenes
are an illustration, not evidence. A run-long T1 gap may also come from the sentence alone (some objects are
hit more often), which this test cannot separate.

### 4.5 E2: the short runs

**Object pairs** (seed mean [95%]; the main column also gives seed 0 / seed 1):

| run | T1+T7 already harmed (main) | T1 already | T7 already | T1+T7 not yet | T1 not yet | T7 not yet |
|---|---|---|---|---|---|---|
| V0 | 0.59 [0.48, 0.70] (0.56 / 0.61) | 0.59 [0.47, 0.73] | 0.58 [0.40, 0.75] | 0.57 [0.51, 0.65] | 0.58 [0.52, 0.68] | 0.52 [0.31, 0.71] |
| V1 | 0.59 [0.44, 0.75] (0.57 / 0.61) | 0.61 [0.44, 0.80] | 0.55 [0.36, 0.71] | 0.62 [0.54, 0.70] | 0.64 [0.57, 0.73] | 0.53 [0.29, 0.73] |
| V2 | 0.59 [0.50, 0.71] (0.56 / 0.63) | 0.62 [0.51, 0.77] | 0.51 [0.36, 0.66] | 0.62 [0.55, 0.69] | 0.64 [0.57, 0.72] | 0.54 [0.34, 0.73] |
| V3 | 0.58 [0.47, 0.71] (0.58 / 0.59) | 0.59 [0.47, 0.75] | 0.56 [0.36, 0.73] | 0.57 [0.50, 0.65] | 0.58 [0.51, 0.67] | 0.51 [0.30, 0.71] |
| V4 | 0.60 [0.50, 0.72] (0.63 / 0.58) | 0.61 [0.50, 0.76] | 0.59 [0.43, 0.73] | 0.63 [0.57, 0.71] | 0.66 [0.59, 0.74] | 0.52 [0.33, 0.71] |
| prior | 0.55 [0.47, 0.64] | 0.55 [0.44, 0.65] | 0.57 [0.40, 0.71] | 0.58 [0.49, 0.66] | 0.57 [0.45, 0.67] | 0.62 [0.45, 0.77] |
| pairs | 1735 | 1279 | 456 | 881 | 722 | 159 |

**Rule-kind pairs:**

| run | T1+T7 already harmed | T1 already | T7 already | T1+T7 not yet | T1 not yet | T7 not yet |
|---|---|---|---|---|---|---|
| V0 | 0.63 [0.51, 0.76] | 0.60 [0.45, 0.78] | 0.71 [0.61, 0.80] | 0.68 [0.60, 0.77] | 0.71 [0.61, 0.83] | 0.60 [0.50, 0.67] |
| V1 | 0.69 [0.57, 0.80] | 0.69 [0.54, 0.85] | 0.67 [0.57, 0.75] | 0.62 [0.51, 0.72] | 0.62 [0.49, 0.75] | 0.60 [0.51, 0.68] |
| V2 | 0.67 [0.56, 0.79] | 0.66 [0.51, 0.83] | 0.69 [0.61, 0.76] | 0.60 [0.52, 0.69] | 0.61 [0.50, 0.73] | 0.57 [0.44, 0.71] |
| V3 | 0.57 [0.44, 0.70] | 0.56 [0.39, 0.74] | 0.63 [0.55, 0.70] | 0.68 [0.60, 0.77] | 0.70 [0.59, 0.82] | 0.63 [0.56, 0.70] |
| V4 | 0.57 [0.46, 0.70] | 0.56 [0.41, 0.72] | 0.61 [0.52, 0.69] | 0.65 [0.56, 0.74] | 0.67 [0.55, 0.79] | 0.59 [0.49, 0.69] |
| prior | 0.59 [0.52, 0.65] | 0.59 [0.50, 0.67] | 0.56 [0.50, 0.62] | 0.47 [0.42, 0.53] | 0.42 [0.37, 0.46] | 0.63 [0.51, 0.74] |
| pairs | 589 | 453 | 136 | 190 | 144 | 46 |

**Reading:** no version is distinguishable from V0 on the main score, and all are 0.25 below the bar. The
short runs also read the kind of rule worse than the official model (0.57-0.69 vs 0.79 already harmed),
which trained on 7 tasks that share these consequence words.

### 4.6 Does V3 read its tag?

The same V3 checkpoint scored with its learned tag embedding zeroed (`eval_offline --zero-tag`):

| run | main pairs | outcomes changed | with tag | tag zeroed |
|---|---|---|---|---|
| V3 s0 | 1735 | 1.7% | 0.580 | 0.569 |
| V3 s1 | 1735 | 0.9% | 0.588 | 0.581 |

**Reading:** the model ignores the oracle location ("under 5% of outcomes changed" counts as not read; the
threshold was set in the round-4 audit, `TAG_FLIP_MIN`). V3 is therefore no ceiling: it does not show what a
model that uses the object's location could reach.

### 4.7 Training scenes vs new scenes

Main score computed on the T1 / T7 ur5 training episodes (`eval_offline ... --out-name grounding_train`,
2,498 pairs, 47 episodes with pairs) and on the validation episodes. "In-batch" = the training batches' own
pair rank accuracy over the last 1,000 steps (all pair kinds and moments).

| model | training episodes | new episodes | in-batch |
|---|---|---|---|
| official (7 tasks, both robots, 20k steps; 120 T1 / T7 scenes) | 0.93 [0.90, 0.96] | 0.57 [0.45, 0.71] | – |
| V0 s0 | 0.971 | 0.557 | 1.000 |
| V0 s1 | 0.974 | 0.613 | 1.000 |
| V1 s0 | 0.974 | 0.575 | 1.000 |
| V3 s0 | 0.962 | 0.580 | 0.938 |
| V1 s1 / V2 s0 / V2 s1 / V3 s1 / V4 s0 / V4 s1 | not scored | 0.612 / 0.560 / 0.628 / 0.588 / 0.627 / 0.581 | 0.94-1.00 |

Not-yet pairs (all pair kinds), model vs ranking by the current h:

| model | scenes | T1: model / h now | T7: model / h now |
|---|---|---|---|
| official | training | 0.74 / 0.73 | 0.88 / 0.61 |
| V0 s0 | training | 0.84 / 0.73 | 0.96 / 0.61 |
| official | new | 0.58 / 0.67 | 0.61 / 0.62 |

**Reading:** on training scenes the model picks the named object almost every time, and before contact it
ranks far above what the present state allows (T7: 0.96 vs 0.61). Knowing which object a scene's careless
motion will hit, before it happens, can only come from memory of that episode. On new scenes it is at the
words-only level and below the current h.

### 4.8 Learning curve

V0 s0, main score on new scenes by checkpoint: 0.567 (1,000 steps), 0.563 (2,000), 0.561 (3,000), 0.560
(4,000), 0.542 (5,000), 0.557 (6,000).

**Reading:** flat from the first checkpoint. Not undertrained; longer training will not help.

### 4.9 V4: the auxiliary target loss

- **Calibration:** λ set so that λ·L_aux / L_critic ≈ 0.2 (median over steps 400-600) by two 600-step passes
  (`scripts/lang/aux_calibrate.py`): ratio 5.3 at λ = 1 gave λ = 0.0376, the second pass gave λ = 0.0082.
  In the real runs the median ratio over steps 400-600 was 0.26 (s0) and 0.48 (s1), the last logged 0.21
  and 0.08. Both count as tried (threshold 0.05, fixed in round 4).
- **Score:** 0.60 [0.50, 0.72] (seeds 0.627 / 0.581) vs V0 0.59 (0.557 / 0.613).
- **Where the head points**, median centre error at frame t:

| | T1 | T7 |
|---|---|---|
| V4 head, training batches (both tasks together) | 6 cm | |
| V4 head, new scenes | 19 cm | 6 cm |
| middle of the scene's named objects, to the named one | 18 cm | 5 cm |
| nearest other named object, to the named one | 13 cm | 10 cm |

**Reading:** step b fails. On new scenes the head is no closer to the named object than the middle of the
scene (on T1 it is farther than the nearest wrong object). A direct "where is the named object" signal is
learned for the training scenes only.

## 5. Conclusions

Each with the evidence it rests on and how firmly we hold it.

1. **Decision 5 outcome: neither branch fires.** Step a (V1 0.59, V2 0.59) and step b (V4 0.60) are far
   below 0.85; V3 is no ceiling (4.6). Firm.
2. **The text path is not the bottleneck.** The words of the named object are attended above their share
   (4.3), removing the one visible defect changes nothing (V1, V2), and the same model picks the named
   object at 0.96-0.97 on training scenes (4.7), so the sentence does reach the decision. Firm for these
   two tasks.
3. **The model memorizes its training scenes.** Evidence: the training/new gap (0.97 vs 0.56-0.61), the
   above-h foresight on training scenes only (4.7), the flat learning curve (4.8), and V3 ignoring the oracle
   because the training loss is already near zero without it (in-batch pair accuracy 0.94-1.00). Firm.
4. **Slots plus a pointer are not supported by this evidence.** The rule requires V3 to work, and it does
   not. More to the point, V4 already supplied the kind of simulator-labelled "which object" supervision a
   pointer loss would supply, and it did not carry to new scenes. This is an argument, not a test of
   slots plus a pointer.
5. **The kind of rule reads better because its words are shared across scenes** (interpretation). The
   official model, trained on 7 tasks that share upright / stay words, reads it at 0.79 (prior 0.59); the
   short T1 + T7 runs read it at 0.57-0.69. An object name is tied to a few layouts, which are easy to
   memorize.
6. **Leading hypothesis: too few distinct training scenes.** 60 scenes, each seen about 6,400 times in a
   6,000-step run (384k samples), make memorizing which object each scene hits the easiest fit. Caveat: the
   official model had 120 T1 / T7 scenes (two robots, plus five other tasks) and still scores 0.57 on new
   ur5 scenes, so doubling did not help there. Either the needed scale is much larger than 2× or the cause is
   something else. Not yet tested (section 8).

## 6. Limits of these results

- Only ur5-wsg, only T1 and T7. 41 validation episodes; intervals are about ±0.11, so differences under
  about 0.1 are invisible. Percentile intervals with this few clusters tend to under-cover.
- Two seeds per version; the intervals do not include seed variation (seed values differ by up to 0.07).
- Offline: Q at the logged nominal action on nominal runs. No closed-loop result here.
- Labels come from one recorded run per scene; cuRobo makes motion differ run to run.
- "Already harmed" is the simulator's h < 0, which may not be visible to the cameras: avoid has a 1 cm
  buffer (h < 0 within 1 cm, not only at contact), and stay counts 2 cm of drift in T7 (visible only
  against frame 0).
- E0a / E3 are six scenes, an illustration only.
- The E2 runs use the linear actor head; production now uses the MLP head. The critic reads the sentence in
  both, but its targets bootstrap through the actor.

## 7. Still running (queue `runs/lang_grounding_1008/e2.cmds`, checkout `_exp3` at 0e97772)

- **V4b, 2 seeds:** trained to 6,000 steps, not yet scored. λ = 0.0039; the calibration's attention ratio came
  out at 1.1-1.2, stronger than the 0.2 aimed at.
- **V0 trained on both robots' T1 / T7 (120 scenes, `g1008_d2_s0/1`), scored on the same ur5 pairs:** not
  started.
- **V0 s0 training-scene score at 1,000 and 3,000 steps:** shows how fast memorization sets in. Not started.
- **Official model on the new T3 / T5 validation episodes (val20):** not started.

Decision 5 is settled for steps a and b except V4b. V4b would change conclusion 1 only if it reaches 0.85.

## 8. Proposed next step and questions for the reviewer

**Proposal: a scene-count curve before any architecture change.** Train V0 (same settings) on 15, 30 and 60
ur5 T1 + T7 scenes (subsampled from the existing data) and on 120 (d2), and score all on the same new-scene
pairs.
- If the new-scene score rises with the number of scenes, collect many more T1 / T7 scenes (hundreds) and
  re-test.
- If it stays flat at 0.55-0.60 at every size, the cause is not the scene count. We then revisit: the
  input (whether the named object's state is visible in the HoloBrain tokens at all), the label (h < 0 vs
  visible harm), or the architecture.

Questions:
1. Is the memorization evidence (4.7, 4.8) enough to set aside slots plus a pointer for now?
2. Should the bar stay at 0.85 on already-harmed pairs, given that ranking by the current h gives 1.00 there?
   Or should the gate move to not-yet pairs, which test foresight?
3. Is the scene-count curve the right next test, and what scene count would you accept as "enough" if the
   curve keeps rising?

## 9. Provenance

- Code: V0-V3 at e9f1d98 (V3 s1 at a5b6055, checkpoint export fix only); V4 / V4b at 0e97772 (frozen
  checkout `LanguageConditionHJ_exp3`).
- Runs: `runs/lang/grounding_1008/g1008_v{0..4,4b}_s{0,1}`; scores in `<run>/eval_offline/grounding/`
  (new scenes), `grounding_train/` (training scenes), `grounding_notag/` (V3, tag zeroed), `grounding_step*/`
  (learning curve).
- Official model: `runs/lang/lang_full_paired_off_1007/eval_offline/grounding_e0/` (tables.md,
  seeing_vs_selecting.md, word_attention.md) and `grounding_e0_train/`.
- Traces: `runs/lang/grounding_1008/e0a/`, `e3/`, videos in `videos/`.
- Summary: `python scripts/lang/grounding_summary.py --runs 'runs/lang/grounding_1008/g1008_v*_s*'
  --out-name grounding --md runs/lang/grounding_1008/summary.md`.
- Validation data: `data/lang/val_grounding` (E1) plus the earlier ur5 T1 / T7 validation episodes.
