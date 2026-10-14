# Linear probe: does the model know where the named object is? (2026-10-14)

Checkpoint: `runs/lang/rounds_ur5_g095_mlp/checkpoint.pt` at step 95,825 (round 46; copy in
`runs/lang/probe_1013/ckpt_g095.pt`), code 0e97772: γ 0.95, softmin critic read-out, MLP actor, no fusion,
ur5-wsg, all 7 tasks. Task probed: T1 (`selective_avoid`), ur5-wsg.

## The question

On the scenes it was trained on, the filter tells objects apart by the sentence; on new scenes it does not
(grounding report 2026-10-09: 0.92-1.00 on training scenes, 0.49-0.57 on new ones). Is the information missing
from what the critic and the actor read, or is it there and the sentence fails to pick it?

## Short answer

1. **Picking by sentence fails on new scenes.** Take one moment with one object near the arm (< 6 cm) and one
   far from it (> 15 cm), and name each in turn. Every frame and the robot state are the same; only the sentence
   differs. A linear read-out of what the critic and actor read puts the near object closer in 60-65% of these
   pairs on new scenes. A read-out of the sentence's words alone does as well (64%), so this comes from
   preferring some names ("the can" is often near), not from looking at the named object. On unseen kinds and
   colours it falls to 40-54%. The model's own Q scores 47% on new scenes.
2. **The information is there.** A linear read-out of the named object's own scene tokens, chosen with the true
   object mask (which the model never gets), ranks 99-100% of the same pairs right in every set, unseen kinds
   and colours included. It also recovers the distance itself (R² 0.49), where the model's summaries reach
   0.01-0.04.
3. **The summaries know that the arm is near something, not near which object.** They predict the distance to
   the nearest object (R² 0.14-0.20) but not to the named one (0.01-0.04).

So the generalization problem sits in the step from the sentence to the named object's tokens. The image
features and a linear read-out are enough once the right tokens are picked.

## What was measured

### Data

All T1 on ur5-wsg, the careless nominal controller (no filter), segmentation masks on. Collected with the
checkpoint's own code (branch `probe-1013`), so names, spawning and words match its training.

| set | objects | episodes run | with samples | clean samples | near/far pairs |
|---|---|---|---|---|---|
| new scenes, training objects | the training pool (bottle, cup, can, pill bottle, tea box, coffee box, drink box, wooden block, box, cylinder) | 60 + 30 | 88 | 14,214 | 2,530 (57 episodes) |
| training scenes again | the 30 warmup ur5 T1 layouts, same seeds (same layout; cuRobo motion differs a little) | 30 | 29 | 4,282 | 695 (18 episodes) |
| unseen kinds | pen holder, stamp, toothpaste, jar, Rubik's cube, apple | 40 (2 layouts could not be placed) | 35 | 5,614 | 822 (25 episodes) |
| unseen colours | box and cylinder in purple, orange, cyan; shape-and-colour pairs distinct in a scene | 30 | 30 | 6,072 | 1,104 (26 episodes) |

### A sample

- **What the model sees**: tick t of an episode, as in training: the left wrist, right wrist and head
  cameras at frames t, t - 2 and 0 (episode start), plus the robot state.
- **Sentence**: one training template naming one candidate object k, in the checkpoint's own words. Example:
  "steer clear of the tea box". One sentence per object per episode.
- **Label**: d = distance from the robot's collision spheres to the object's collision shape, the distance the
  avoid rule uses (h = clip((d - 1 cm) / 5 cm, -1, 1)). Example: 2.4 cm.
- **Clean**, checked on the head camera at t and at t - 2:
  - at most 5% of the object's silhouette is out of the picture or behind the arm or another object;
  - at least 100 pixels of it are visible;
  - not toppled (tilt ≤ 15°), not held, not touching.
- **Pair**: two clean objects of the same moment, one closer than 6 cm and one farther than 15 cm.

Pictures:
- `runs/lang/probe_1013/examples/probe_examples.png`: real pairs on the front video camera;
- `runs/lang/probe_1013/camera_examples/`: the model's nine input pictures of a few pairs.

### What was read out

What the critic and the actor read (the encoder output, before the action):
- the 4 read-out queries;
- the rule token r and the sentence summary r0;
- the body tokens (their mean, and the 14 joint and payload tokens the actor reads);
- the critic's last block (queries and body after the action is added).

Controls:
- the sentence's adapted words alone (no image);
- the model's own Q (no fitting);
- an empty sentence in place of every sentence.

Oracle (diagnostics only):
- the named object's own scene tokens of frame t, averaged with the true object mask;
- with no fitting at all, the distance from those tokens' 3D positions to the nearest robot link.

Read-outs:
- ridge regression on log(d + 1 cm), and logistic regression on near versus far;
- fitted on the 88 new-scene episodes and scored out of fold (5 folds by episode);
- the other sets are scored by the read-out refitted on all 88.

Intervals are 95% bootstrap over episodes.

## Results

### 1. Same moment, two objects named (only the sentence differs; 0.5 = the sentence makes no difference)

| read-out of | new scenes | training scenes | unseen kinds | unseen colours |
|---|---|---|---|---|
| the model's own Q (no fit) | 0.47 [0.35, 0.60] | 0.68 [0.37, 0.87] | 0.49 [0.28, 0.69] | 0.72 [0.61, 0.82] |
| critic read-out input (queries + r + r0 + body mean) | 0.65 [0.54, 0.76] | 0.66 [0.44, 0.85] | 0.42 [0.25, 0.60] | 0.42 [0.26, 0.60] |
| the 4 read-out queries | 0.61 [0.50, 0.73] | 0.69 [0.49, 0.86] | 0.51 [0.35, 0.67] | 0.40 [0.26, 0.56] |
| joint tokens (the actor's input) | 0.63 [0.51, 0.73] | 0.62 [0.44, 0.79] | 0.51 [0.35, 0.67] | 0.52 [0.34, 0.73] |
| the critic's last block | 0.60 [0.48, 0.71] | 0.65 [0.44, 0.84] | 0.40 [0.27, 0.53] | 0.49 [0.32, 0.66] |
| the sentence's words only (control) | 0.64 [0.52, 0.76] | 0.61 [0.42, 0.80] | 0.53 [0.33, 0.76] | 0.45 [0.28, 0.64] |
| ORACLE: the named object's own tokens | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] |
| ORACLE geometry, no fit | 0.94 [0.89, 0.98] | 0.97 [0.93, 0.99] | 0.93 [0.86, 0.99] | 0.99 [0.98, 0.99] |

The logistic near/far read-outs give the same picture (0.58-0.65 on new scenes, 0.96-0.99 for the oracle).

### 2. How much of the distance each read-out recovers (R², out of fold, new scenes)

| read-out of | distance to the named object | distance to the nearest object, any |
|---|---|---|
| critic read-out input | 0.02 | 0.19 |
| the 4 read-out queries | 0.01 | 0.18 |
| the critic's last block | 0.01 | 0.20 |
| joint tokens | 0.03 | 0.15 |
| body tokens, mean | 0.04 | 0.14 |
| the sentence's words only | 0.01 | 0.00 |
| ORACLE: the named object's own tokens | 0.49 (0.46 / 0.41 / 0.49 on training scenes / unseen kinds / unseen colours) | 0.00 |

### 3. The model's Q prefers some names

The model's own Q on new-scene pairs, by the kind of the near object (share ranked right):

| near object | can | wooden block | bottle | cylinder | drink box | box | tea box | pill bottle | coffee box |
|---|---|---|---|---|---|---|---|---|---|
| Q ranks right | 0.96 | 0.87 | 0.41 | 0.40 | 0.39 | 0.34 | 0.16 | 0.07 | 0.00 |

A sentence about a can reads dangerous whatever the scene; one about a coffee box reads safe whatever the
scene. On unseen colours, pairs of two boxes or two cylinders, where only the colour word differs, give Q 0.62
[0.43, 0.78] (box versus cylinder pairs: 0.78). The summaries' read-out gives 0.44 and the oracle 1.00.

### 4. Same sentence, two moments: explained by the scene alone

This test compares one object named by the same words at two moments of an episode: near the arm at one,
far at the other while another object is near. The read-outs rank it well on new scenes: 0.84-0.87 for the
critic read-out input, the queries and the critic's last block, over 26 objects. With an empty sentence in
place of the real one they score the same, 0.86-0.88. The two moments differ in other ways (the arm's height,
the phase of the motion), and that is what is read. This test shows nothing about the named object.

## What it means

- **Perception is fine.** The frozen HoloBrain tokens, after the scene adapter, carry how close the arm is to
  each object, also for kinds and colours never seen in training. One linear map of an object's own tokens
  gives its distance.
- **Selection is what fails.** The sentence does not steer the queries, the rule token or the body tokens
  toward the named object's tokens on new scenes. What the model learned instead are name-level preferences
  that happen to fit its 60 training scenes ("the can is dangerous"). On those scenes the gain over the
  words-only control is small (0.65-0.69 against 0.61), with wide intervals.
- **This fits the earlier results.** The training-scene accuracy without transfer (memorization; grounding
  report 2026-10-09) and the frozen fusion stage not helping (E4/E5) both point to selection, not perception.
  The oracle tag V3 was ignored (0.58): adding a tag to the named tokens was not enough. The probe says that
  once the right tokens are read, a linear map suffices, so routing the read-out through the named object's
  tokens is the missing piece, not more scene features.

## Caveats

- Linear read-outs only. A nonlinear one might find more in the summaries; the point is that a linear one is
  enough on the right tokens.
- The oracle tokens are 16-32 pixel patches at the object's place. When the arm is over the object they show the
  arm too, so they encode whether the arm is at that object, which is what selection needs.
- Pairs come in runs within an episode, so the intervals are over episodes (57 episodes hold new-scene pairs)
  and stay wide.
- The sentences use the checkpoint's old words ("the tin", "the small bottle"), on purpose.
- Nominal (careless) motion, T1, ur5-wsg only.

## Files

- Code: branch `probe-1013`, worktree `LanguageConditionHJ_probe`:
  - `main/lang/probe_vis.py`;
  - `scripts/lang/probe_{collect,features,fit,report,examples,camera_examples,image_check}.*`;
  - tests in `tests/lang/test_probe_{vis,fit}.py`.
- Data: `data/lang_probe/{ptrain,idtest,train_scenes,ood_kinds,ood_colours}`.
- Features: `runs/lang/probe_1013/features`.
- Tables: `runs/lang/probe_1013/report/report.md`, generated with the empty-sentence control from
  `report_nosent/`; the nearest-object R² is in `report/nearest_any_r2.json`.
