# Known issues in training a language-conditioned HJ safety filter

A living list of what went wrong, why, and what is being done. Last updated 2026-11-12.

Each issue has the same four parts. What happens, why, the evidence, and the status. "Measured" means we have
numbers for it. "Hypothesis" means it is a likely cause that has not been isolated yet.

## Setup in brief

A safety filter watches a nominal controller and takes over when the controller is about to break a rule given
as a sentence, for example "keep the tall block upright".

```
y = (1 − γ) h(s) + γ min( h(s), Q̄(s′, a′) )        critic target (HJ reachability, avoid form)
engage the safe actor when  Q(s, a_nom) < 0
```

- s is the state (camera tokens from a frozen vision-language encoder, robot state, the sentence).
- a is the action, a per-joint displacement for one control tick of dt = 40 ms (25 Hz).
- h(s) is the safety margin of the named rule. 1 is safe, 0 is the moment of harm, −1 is badly harmed.
- s′ is the next state, a′ the actor's action there, Q̄ the target network, γ = 0.95 the discount.
- a_nom is the nominal controller's action. Here it is a motion planner that ignores the protected items.
- The actor is trained as in SAC to maximize Q plus an entropy bonus.

Tasks are tabletop manipulation in simulation where the protected item moves on its own (a block on a sliding
tray, beads being poured, objects that can be knocked). The running example below is the tray. The robot drags
a tray with a tall block on it, and a hard brake at the end tips the block.

## Symptom

Training metrics looked good and the closed loop did not work.

- On training batches the critic ranked a harmful sentence below a harmless one 98% of the time.
- With the filter on, the named rule was still broken in 47% of the episodes where the item was in the way
  (about 70% for the planner alone).
- The robot stalled. 77% of those episodes ran out of time.
- The filter engaged in 54% of the episodes where the named item was never at risk.

## Issue 1. One tick is too small an action for Q to see

**What happens.** Q(s, a) is almost the same for every action a.

**Why.** Q(s, a) is the value of doing a for one tick and then following the policy.

```
Q(s, a) = (1 − γ) h(s) + γ min( h(s), V(s′) ),    s′ = s + f(s, a) dt
```

- h(s) is the margin now. It does not depend on a.
- V(s′) is the value of the next state.
- f is the dynamics.

The action appears only in s′. Two actions a₁ and a₂ lead to next states that differ by about
‖f(s, a₁) − f(s, a₂)‖ dt, so

```
Q(s, a₁) − Q(s, a₂)  ≈  γ ∇V(s′) · ( f(s, a₁) − f(s, a₂) ) dt   ∝ dt
```

The gap between actions shrinks with the tick length, and Q(s, a) tends to V(s) as dt goes to 0 (Tallec et al.,
2019). The part of the target explained by the state does not shrink. A critic trained by regression fits the
large part and never the small one.

**Evidence (measured).** One tick moves the robot's links by a median 2.2 mm. Changing one tick's action changes
the true value by a median 0.03, while the value varies across states with a standard deviation of about 0.6.
The action is roughly (0.03 / 0.6)² ≈ 0.25% of the target's variance. Whole continuations of 1 s or more differ
by 0.49.

**Status.** Not the limit near the boundary (2026-11-12). The 0.03 above is a median over all ticks of the drag.
At states in the last second before the brake, the true value differs by 0.36 on average between two one-tick
choices (a hard stop for a single tick tips the block), 0.51 between two 3-tick choices and 0.72 between two
0.5 s chunks. Critics that take a 3-tick action or a chunk stayed flat as well (Q moved by 0.07 to 0.08 of the
true change). The argument still explains why Q is flat far from the boundary.

## Issue 2. The data has one action per state

**What happens.** Nothing in the replay buffer shows that a different action from the same state leads to a
different outcome.

**Why.** The nominal controller is deterministic, so in the data a = a_nom(s). The critic loss is an average over
the data.

```
L = E_(s, a) ~ data [ ( Q(s, a) − y )² ]
```

The data lies on the curve a = a_nom(s). Any two critics that agree on that curve have the same loss, including
one that ignores a. Episodes run at different speeds do not help, because they are different states, and the
critic can read the speed from the frames.

**Evidence (measured).** Random actions were 0.9% of the planner's ticks. Every other stored action was the
planner's own.

**Status.** Necessary, not sufficient (2026-11-12). Branch data was collected. The simulator is restored to one
saved state near the boundary and continued with several different motions, each run to its real outcome (600
states, 7 to 9 continuations each). Trained on it with a quarter of every batch drawn from the rows that show the
contrast, the critic still moved Q by less than 0.13 of the true change, on its own training states too. More
states did not help. The open question is the training target: whether the bootstrapped target carries the
difference between two choices at all.

## Issue 3. Ignoring the action is a stable solution

**What happens.** The critic drops the action early in training and never recovers it.

**Why.** A critic that outputs V(s) for every action satisfies the target up to the one-tick effect of issue 1.
Its loss is about 0.03² ≈ 10⁻³, below the noise of the targets (bootstrapping, the lagging target network, the
sampled a′). The target itself is built from the critic's value at s′. If that value ignores the action, the
targets carry no information about it either, so the blind solution reproduces itself.

**Evidence (measured).** In every retraining variant (the actor's sample, its mean, the stored next action or a
freeze as a′) the spread of Q over random actions fell from 0.011 to 7.5e-5 within 250 steps, and was still
36 times too small after 20,000.

**Status.** Measured. It follows from issues 1 and 2 and is not treated separately.

## Issue 4. The action is drowned where it enters the network

**What happens.** The action's contribution to the critic's tokens is about one part in 10,000.

**Why.** The action enters as raw physical numbers through one linear layer and is added to the trunk's output
tokens.

```
token = token + W e(a)        ‖token‖ ≈ 10⁴,   ‖W e(a)‖ ≈ 1
```

- e(a) holds the joint step in radians and the link translation in metres, about 0.002 per tick.
- The trunk has no normalization at its output, so its tokens grow from about 100 to about 10⁴.

The geometric attention also uses the link positions after the action, but a 2 mm shift barely changes it.

**Evidence (measured).** Token size 7,850 to 14,000 with an action part of about 1. Scaling the action inputs to
unit size alone raised the spread of Q only 2.1 times.

**Status.** Confirmed as a cause (2026-11-11). With the original injection the critic could not fit action
differences even when trained directly on the true outcomes, and adding branch data or a 0.5 s chunk left Q flat
(Q moved by 0.0001 of the true change or less). A revised injection (a normalization before the action is added,
action features divided by their bounds and passed through a small MLP, the action embedding fed directly to the
readout) raised that to 0.04 with the chunk and the branch data. That is about 400 times better and still far too
flat, so it is necessary and not sufficient.

## Issue 5. A softmin readout sends the gradient to heads that do not see the action

**What happens.** Q is read from tokens that carry no action.

**Why.** The critic had one value head per token and took a soft minimum.

```
Q = −T log Σ_i exp( −V_i / T ),      ∂Q/∂V_i = softmax( −V / T )_i
```

- V_i is the value from token i (robot link tokens, which carry the action, and query tokens, which do not).
- T is the temperature, 0.05.

Almost all of the gradient goes to the token that holds the minimum. Once a query token holds it, the link heads
receive no gradient and drift.

**Evidence (measured).** The minimum sat on a query token in 100% of the states. The link heads sat 150 to 930
units above it.

**Status.** Readout replaced by an MLP over all tokens. That alone was not enough (the spread of Q rose 16 times
and was still 64 times too small).

## Issue 6. With a flat critic the actor collapses to a freeze

**What happens.** The safe actor outputs nearly zero motion in every state.

**Why.** The actor minimizes

```
L_actor = E [ α log π(a | s) − Q(s, a) ]
```

- π is a Gaussian squashed by tanh, α the entropy weight.

With ∂Q/∂a ≈ 0 only the entropy term acts. Its optimum for a tanh-squashed Gaussian is a mean of 0 and a
standard deviation of about 0.87, whatever the state.

**Evidence (measured).** Mean +0.018, standard deviation 0.864. The deployed command was 1.8% of the joint
limit.

**Status.** Measured. It is a consequence of issues 1 to 4 and needs no fix of its own.

## Issue 7. A freeze is not a safe action when the item moves on its own

**What happens.** When the filter engages, stopping the robot causes the harm or locks the episode.

**Why.** For a static obstacle, stopping is always safe. For an item with its own dynamics it is not. Stopping a
moving tray is the hard brake that tips the block. A stopped robot also leaves the state unchanged, so
Q(s, a_nom) stays below 0 and the filter never hands control back.

**Evidence (measured, by restoring one simulator state and trying each continuation).**

| time before the planner's brake | planner | freeze | half speed | smooth stop over 1 s |
|---|---|---|---|---|
| 0.6 s or more | block falls 100% | 0% | 8% | 37% |
| 0.4 to 0.6 s | 100% | 86% | 57% | 57% |
| 0.2 to 0.4 s | 95% | 100% | 91% | 55% |

A freeze at ticks where the planner alone was safe tipped the block in 24 to 75% of runs. A freeze for three
ticks followed by the plan at full speed tipped it in 85 to 100%.

**Status.** Measured. The safe action is a continuation such as slowing down, which again points to decisions
longer than one tick. The filter also has to warn earlier, at least 0.3 to 0.6 s before the harm.

## Issue 8. Long runs drift to "everything is unsafe"

**What happens.** Mean Q falls below h and keeps falling until the filter drives almost every tick.

**Why (reading of the logs, not isolated).** The target is pessimistic three times over (the minimum with h, the
minimum over twin critics, a soft minimum that is below the true minimum) and it bootstraps through an actor that
does not move. Once Q is below 0 everywhere the filter is always on, and the new data are ticks of a stalled
robot.

**Evidence (measured).** In long runs the filter drove 90 to 100% of the ticks.

**Status.** Open. Expected to change once the critic reads the action.

## Issue 9. Naming an object does not generalize to new scenes

**What happens.** In training scenes the critic tells apart "don't touch the red box" and "don't touch the
bottle". In new scenes it does not.

**Why (reading of the results).** With a few dozen training scenes the mapping from a name to an object is
memorized per scene instead of read from the image.

**Evidence (measured).** On new scenes, ranking two sentences that name different objects of the same scene was
right in 0.59 to 0.60 of the pairs (chance 0.5). Text normalization, an auxiliary loss that predicts the named
object's position, and even a ground-truth marker on the named object's tokens did not change it.

**Status.** Open. Separate from the action issues. Candidates are more scenes, and a pretrained text-image fusion
stage in front of the critic.

## Issue 10. The offline metrics did not test what failed

**What happens.** 98% rank accuracy in training sat next to a filter that stalls in closed loop.

**Why.** The metric compared two sentences at the same state and action. It never compared two actions, and it
was computed on training scenes.

**Status.** Add to every evaluation the ranking of several real continuations from one saved state (issue 7's
table as a metric), and score sentence pairs on held-out scenes.

## Summary

| issue | kind | status |
|---|---|---|
| 1. one tick is too small an action | not the limit near the boundary | a lower rate and chunks did not fix it |
| 2. one action per state in the data | necessary, not sufficient | branch data collected, critic still flat, target under test |
| 3. ignoring the action is stable | mechanism | follows from 1 and 2 |
| 4. action drowned at injection | cause (confirmed) | revised injection is necessary, not sufficient alone |
| 5. softmin readout | measured | replaced by an MLP readout, not sufficient alone |
| 6. actor collapses to a freeze | consequence | none needed |
| 7. a freeze is not safe | measured | longer decisions, earlier warning |
| 8. drift to "everything is unsafe" | open | revisit after 1 to 4 |
| 9. object naming does not generalize | open | more scenes, pretrained fusion |
| 10. metrics missed the failure | measured | continuation ranking, held-out scenes |

## Sources and change log

- Numbers come from `STATUS_2026-10-30.md`, `DIAG_ACTOR_2026-10-30.md` and `grounding_report_2026-10-09.md` in
  this folder. The planned test of issues 1, 2 and 4 is `ACTION_SENSITIVITY_PLAN.md`.
- 2026-11-10. First version.
- 2026-11-11. Issue 4 confirmed by the first round of the action-sensitivity test.
- 2026-11-12. Issues 1 and 2 updated after the second round (ten critics, all flat).
