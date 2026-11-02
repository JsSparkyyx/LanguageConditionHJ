# Why the filter's actor only "stops": the four checks (2026-10-30)

The question (Yuxuan, 10-30, after watching the round-training videos): in T5 (slide a tray with a tall and a short
block) the filter's actor only stops when it thinks the state is unsafe, but stopping at once does not save the tall
block; slowing down does. Three candidate causes were proposed: (i) the critic scores Q(s, stop) above Q(s, slow down);
(ii) the actor does not learn the action that maximizes Q; (iii) the one-peaked (unimodal) Gaussian actor. An outside
analysis added a fourth: the critic ignores the action, because its training target follows a random actor.

Four checks were run on the vanilla round run `runs/lang/rounds_ur5_g095_1023` (ur5-wsg, γ = 0.95, softmin critic
readout, MLP actor head; checkpoints of rounds 10, 20, 30 and the last saved one, round 38/39):

1. what the actor outputs on T5 states;
2. the true outcome of four continuations from the same mid-drag states (the planner, freeze, half speed, a smooth stop
   over 1 s), found by branching the simulation;
3. how the critic ranks those continuations' first actions, compared with the truth;
4. the critic trained again from scratch with other target continuations (the stochastic actor as trained, the actor's
   mean, the action stored at the next tick, zero = freeze), plus the stochastic target with the action inputs
   normalized.

Code: `scripts/lang/diag_1030/`. Results: `runs/lang/diag_1030/{actor_stats,t5_branch,retrain,score}/`.
All four checks are complete. Check 4 ran its 20,000 steps fixed in advance (verdict 10-31 08:10, section 5).

**In one paragraph.** The critic gives every action the same value. Its Q moves by about 1e-5 over the first ticks of
four continuations, and by about 3e-4 over 16 random actions (h units). The true outcomes of those four continuations
differ by about 0.5. Its readout always takes the minimum from a head that sees the
action only indirectly. The actor therefore gets no signal from Q, sits at the entropy bonus's optimum, and its mean
command is a freeze. Retraining from scratch with four different target continuations (the actor's sample, its mean,
the next stored command, freeze) gives the same blind critic, and scaling the action inputs does not help either. So
the random target is not the cause. By the rule fixed in advance the cause is the data and the one-tick time step:
almost every stored action is the planner's, and one 40 ms tick changes the outcome only a little. Check 4 cannot
separate this from the readout, which all variants share.

Symbols used below:
- h: the failure value of the named rule, from −1 (badly harmed) through 0 (just harmed) to +1 (comfortably safe). For
  T5's "keep the tall block upright", h = (45° − tilt) / 45°.
- Q(s, a): the critic's value of doing command a now (one control tick, 40 ms) and then following the policy; the
  filter takes over when Q(s, a_nom) < 0, where a_nom is the planner's command.
- V_0.95: the discounted worst h that Q is trained to predict, computed from a real h trace by the backward recursion
  y(t) = 0.05·h(t) + 0.95·min(h(t), y(t+1)).
- dθmax: the largest command per joint and tick (about 0.13 rad). μ and σ: the actor's mean and noise per joint before
  the tanh; the command sent by the deployed filter is a_det = tanh(μ)·dθmax.
- "spread": the range (max − min) of Q over a set of actions at one state, in h units.

## 1. The actor sits where the entropy bonus alone puts it, and its mean action is a freeze

Measured on 320 T5 drag states where the tall block is about to tip (round 39), and on 1,024 states per checkpoint
across the five tasks:

| quantity | value |
|---|---|
| μ, every joint, every state | +0.018 (spread across states 0.0004) |
| σ | 0.864 (cap 1.0; 0 % of joints at the cap) |
| entropy, analytic / logged | 8.200 / 8.199 nats |
| the deployed command a_det | 1.8 % of dθmax per joint (0.0079 rad per tick vs 0.056 for the planner) |
| what the actor actually sent in the run's 26 T5 filter episodes (9,863 ticks) | median 0.6 % of dθmax, largest 4.3 % |
| how far the arm moved on those ticks | median 0.05 mm per tick (planner ticks: 0.25 mm) |

- σ is not at the cap of 1. 8.200 nats is the highest entropy a tanh-squashed Gaussian can have on 12 joints (8.204 at
  σ = 0.874; at σ = 1 it would be 8.04; checked numerically). So the actor sits exactly at the maximum-entropy point,
  which is where the entropy bonus puts it when Q gives no gradient. Rounds 10, 20 and 30 and the Qwen run look the same.
- The output does not depend on the state. The mean command is a near-freeze.
- In 14 T5 filter episodes the filter first took over while the tray was moving (median 0.31 m/s). The tray stopped
  within one tick, at a median deceleration of 7.7 m/s² (range 2.3-9.1); the tall block tips above about 1.9 m/s². Its h
  went below 0 within 25 ticks in 10 of the 14 (13 of 14 tipped later in the episode).

## 2. Ground truth: on T5 only slowing down saves late in the drag, and freezing is harmful by itself

12 T5 episodes on the A40 (ray-traced, as in training). The planner tips the tall block in 10 of them (time scales
0.75-1.3); at 0.75 (one of two) and 0.5 it does not. 141 branch ticks (117 during the drag). At every branch tick the
whole simulation was saved and restored before each continuation:
- **planner**: the rest of the plan as normal;
- **freeze**: command 0 every tick;
- **half speed**: the same planned path at half rate;
- **smooth stop**: the speed falls linearly to 0 over 25 ticks (1 s);
- **freeze 3 ticks, then resume** the plan (what one filter block followed by a release does).

Each ran until 2 s after the last motion of the arm, tray and blocks (cap 300 ticks). The restore is exact (every
recorded quantity equal after a restore; two runs from one restore agree to 2e-5 m and never in outcome). Each cell was
repeated with 3 tiny kicks to the tall block; the outcomes agreed in 463 of 468 cells.

Ticks where the planner tips the block, by the time before the planner's brake. Each cell: the share of runs in which
the tall block fell / mean V_0.95.

| before the brake | n | planner | freeze | half speed | smooth stop | freeze 3, resume | critic Q(s, a_nom), last checkpoint |
|---|---|---|---|---|---|---|---|
| ≥ 0.6 s | 13 | 100 % / 0.15 | 0 % / 0.98 | 8 % / 0.88 | 37 % / 0.58 | 85 % / 0.47 | +0.82 |
| 0.4-0.6 s | 14 | 100 % / −0.10 | 86 % / −0.11 | 57 % / 0.18 | 57 % / 0.16 | 100 % / −0.17 | +0.13 |
| 0.2-0.4 s | 11 | 95 % / −0.22 | 100 % / −0.55 | 91 % / −0.28 | 55 % / 0.09 | 91 % / −0.38 | −0.70 |
| 0-0.2 s | 15 | 100 % / −0.33 | 100 % / −0.47 | 85 % / −0.22 | 93 % / −0.25 | 87 % / −0.35 | −0.50 |
| after the brake | 11 | all tip | all tip | all tip | all tip | all tip | −0.66 |

- Early in the drag (0.6 s or more before the brake, the tray still slow) any slower continuation saves the block,
  freezing included.
- From about 0.5 s on, freezing fails (86 %, then 100 %) while slowing still saves some.
- The last tick at which anything saves the block (9 tipping episodes): a median 0.31 s before the brake (range
  0.10-0.72). The saving action there was a slow-down in all 9 (smooth stop 7, half speed 3; some ticks both) and freezing
  in none.
- Freezing causes harm on its own: at planner-safe drag ticks it tips the block in 24-75 % of runs (75 % mid-drag,
  n = 16), even at 0.15 m/s. "Freeze 3 ticks, then resume" tips it in 85-100 % (the restart at full speed).
- (The smooth stop is worse than half speed early in the drag only because, when the plan ends within ~12 ticks, it
  still plays the plan's own brake, scaled down.)
- In some episodes the block tipped before the planner's brake (e.g. s5003, s5004: h < 0 at 5.04 s, brake at 5.44 /
  5.30 s), so the hazard is not only the final brake.

## 3. The critic ranks nothing: all first actions get the same Q

At the same 117 drag ticks, the four first actions (a_nom; 0; the first tick of half speed and of the smooth stop):

- Q differs between them by 1.5e-5 h units on average (largest 1.4e-4); the true V_0.95 differs by 0.49 on average.
  Ranking agreement with the truth: 0.45-0.54 (n = 311 pairs; chance 0.5), for all four checkpoints.
- Q(s, stop) vs Q(s, half a_nom) on 320 about-to-tip states: median difference 5.7e-7, and stop scores higher in 56 %
  of states (a coin flip). The difference between running the same Q in TF32 (the training precision) and in fp32 is
  up to 3.5e-4, larger than the whole effect of the action.
- The sentence still moves Q a lot: on the same states, −0.86 with the tall block named, +0.96 with the short one.

**Three truths for the same first action** (point 1 of the review). At 15 drag ticks, each first action was also followed
by the last checkpoint's actor: deterministic (60 rollouts) and stochastic (4 per cell, 240 rollouts).

| truth for Q(s, a) | what it measures | last checkpoint's Q vs it (V_0.95, n = 60) | rounds 10 / 20 / 30 |
|---|---|---|---|
| T_full: the four whole continuations | which action is right | mean abs. error 0.47, signed −0.02 | 0.45 / 0.55 / 0.43 |
| T_det: one tick, then the deterministic actor | what Q should say for the deployed actor | mean abs. error 0.32, signed +0.09 (closest) | 0.45 / 0.44 / 0.44 |
| T_stoch: one tick, then the stochastic actor | what Q was nominally trained toward | mean abs. error 0.87, signed +0.74 (Q far above it) | signed +0.83 / +0.57 / +0.95 |

- The stochastic actor (σ 0.86, about uniform noise over the box) tipped the block in 240 of 240 rollouts (mean V_0.95
  about −0.45); the deterministic near-freeze drifts and tipped it late in 49 of 60.
- So Q does **not** estimate "a, then random flailing". Along the planner-driven episodes Q(s, a_nom) tracks the
  planner's own V_0.95 (mean absolute error 0.12-0.37). The likely reason (inferred): once Q ignores the action, the
  action used at s′ in the target makes no difference, so Q learns the value of whatever the data did next, mostly the
  planner.
- **How much one tick can matter at all**: one different first tick followed by the same deterministic actor changes
  V_0.95 by a median of 0.03 (mean 0.11, one outlier 0.9), and followed by the stochastic actor by 0.06, about its
  noise. The four whole continuations differ by 0.75. So even a perfect critic of one-tick actions sees only a small gap.
  The critic's 1e-5 is still about 1,000 times below that small gap.

**When the filter would engage.** Along the nominal episodes Q(s, a_nom) of the last checkpoint first goes below 0 between
0.20 and 0.68 s after the drag starts in every episode, including the two the planner does not tip (0.32 s and 0.68 s
in). In the 9 tipping episodes with such a tick: at that moment freezing saves 0 of 9, a slowing action 3 of 9, nothing
saves 4 of 9, and in 2 the planner itself was still safe. On T5 drag states that never tip, Q(s, a_nom) < 0 in 57 % of
320 states (false alarms). Reading (inferred): Q flags "a drag with the tall block named is under way" more than the speed
that actually tips it.

## 4. Where the action's effect is lost inside the critic (measured; that it is the cause is inferred)

- The action enters the critic only through the 22 body (link) tokens: Linear(8, 256) on [own δθ, J_lin·δθ, J_ang·δθ,
  valid] is added to each link token.
- The softmin readout takes Q from the minimum over 26 heads (22 link tokens, 4 readout queries). At initialization the
  minimum is a link token in 100 % of states; in every trained checkpoint (and every softmin retrain) it is a readout
  query in 100 % of states. The link heads sit far above Q (median 392, 695, 1,193 and 2,027 h units at rounds 10, 20,
  30 and 38), so they get zero weight and zero gradient. The link heads do respond to the action (0.23-0.96 per head over
  16 random actions); the query heads, which set Q, move by 4e-5 to 5e-4.
- The body tokens grow during training: norm 101 at initialization, then 1,010 / 2,432 / 3,614 / 9,598 at rounds
  10 / 20 / 30 / 38, the growth in trunk block 0. The action's contribution stays about 0.9, so a full-box action is
  9e-5 of a body token and the planner's own command 7e-6.
- A freshly initialized critic is about 50 times more action-sensitive (spread 7.2e-3 at the same states), so the
  blindness is learned.

## 5. Check 4: retraining with other target continuations (final, 20,000 steps)

Same data (warmup + the vanilla run's ring), same config, from scratch; only the action a′ used at s′ in the target
changes. Measured every 250 steps on 768 fixed states (512 from validation scenes, 256 from training, 256 of them T5 drag
states). Criteria fixed in advance (`runs/lang/diag_1030/retrain/criteria.json`): "clear spread" = Q spread over 16
random actions ≥ 0.02 h units; "departs" = clear and ≥ 2× the stochastic control at the same step; on T5 drag states the
sign must agree with check 2's ranking (≥ 0.70 of the pairs whose truth differs by > 0.05, a pair counting only if its
Q gap is ≥ 1e-3).

Spread of Q over 16 random actions on the 768 fixed states (the metric fixed in advance), then the T5 scores of the
final checkpoint on check 2's 117 drag states:

| variant (a′ in the target) | step 0 | 250 | 5,000 | 20,000 | ÷ stoch at 20,000 | T5: Q spread over the 4 branch first actions (truth 0.49) | T5 sign test, pairs decided (of 311) | T5 signs agreeing, any gap (chance 0.5) | Q vs the truth of its own continuation, MAE |
|---|---|---|---|---|---|---|---|---|---|
| stoch: a sample of the actor (as trained) | 0.011 | 7.5e-5 | 2.0e-5 | 2.7e-4 | 1.0 | 1.2e-5 | 0 | 0.51 | 0.85 (T_stoch) |
| det: the actor's mean | 0.011 | 7.4e-5 | 4.5e-5 | 2.7e-4 | 1.0 | 1.1e-5 | 0 | 0.50 | 0.21 (T_det) |
| data: the stored next command | 0.011 | 7.5e-5 | 2.9e-5 | 4.2e-4 | 1.6 | 2.1e-5 | 0 | 0.45 | 0.55 (planner branch) |
| zero: freeze | 0.011 | 7.5e-5 | 2.6e-5 | 3.9e-4 | 1.5 | 2.5e-5 | 0 | 0.43 | 0.40 (freeze branch) |
| stoch_norm: stoch, action inputs scaled to unit size | 0.059 | 2.7e-4 | 6.5e-5 | 5.5e-4 | 2.1 | 9.6e-6 | 0 | 0.63 | 0.66 (T_stoch) |
| stoch_mlp (exploratory, MLP readout, stopped at 5,000) | 0.0005 | 7.6e-5 | 3.1e-4 | – | 16 at 5,000 | 4.0e-5 | 0 | 0.55 | 0.64 (T_stoch) |
| the vanilla round run's last checkpoint (79,872 steps, reference) | | | | | | 1.5e-5 | 0 | 0.42 | 0.87 (T_stoch) |

**Verdict, by the rule fixed in advance: "not the target; the data and the one-tick time step".**
- No variant reaches a clear spread (the largest, stoch_norm, is 5.5e-4, 36× below 0.02). det, data and zero stay
  within 1.6× of the stochastic control, so the action used in the target is not what makes the critic blind. stoch_norm
  is 2.1× the control but not clear, so the input scale is not enough on its own either.
- T5 sign test: no final checkpoint has a single pair with a Q gap of 1e-3 (score 0.00, bar 0.70). Counting any gap,
  the signs agree with the truth in 0.43-0.63 of the pairs. stoch_norm (0.63, Kendall τ 0.24) is the only one above
  chance, with gaps of about 1e-5.
- The actor stays at zero in every variant: mean |tanh μ| 0.007-0.024, against the bar of 0.05. The criterion's
  other branch, σ < 0.9, is met by every variant and means nothing, because the tanh-Gaussian's maximum-entropy σ is
  0.874 (check 1). I noticed this on 10-31 at 00:30, from the interim numbers and before the 20,000-step numbers
  existed.

**What else the 20,000 steps show.**
- Every target loses the action in the same first 250 steps (0.011 → 7.5e-5), while the target critics (τ = 0.005) are
  still near their initialization. The stochastic control reproduces the original run (batch action sensitivity
  3.7e-5 to 4.9e-5, vs 4.1e-5 to 6.3e-5 in the original log at the same steps).
- After that the spread grows slowly in every variant, about 10× from 5,000 to 20,000 steps. Longer training does not
  close the gap, though: the vanilla run's own last checkpoint, trained 4× longer, sits at the same level on the same
  T5 states (16 random actions: 3.1e-4, vs 2.2e-4 to 4.4e-4 here). Over the four branch first actions Q spreads 1e-5
  to 2.5e-5 (the largest gap at any state is 7e-4). The truth spreads 0.49 over the four whole continuations, and about
  0.11 when each first tick is followed by the same deterministic actor (section 3). So Q is still thousands of times
  too flat, even as a critic of one-tick actions.
- The critics do learn the value of the state under their own continuation: Q against its own truth has an error of
  0.21 for det (one tick, then the deterministic near-freeze), 0.40 for zero (freeze) and 0.55 for data (the planner).
  At about 2,000 steps these were 0.71, 0.70 and 0.87. What they never learn is the difference between actions.
- Readout: in every softmin variant the minimum sits on a readout query in 100 % of the states, at every step. The link
  heads, which carry the action, sit a median 150-930 h units above it and move 0.23-0.45 over the 16 actions, but never
  set Q.
- The MLP readout made the action reach Q faster: 16× the softmin control at 5,000 steps (3.1e-4 vs 2.0e-5). It was
  still 64× below 0.02, and the softmin variants caught up with it by 20,000 steps.

**What check 4 does not separate.** The rule reaches "data and time step" by elimination. Check 4 changed only the
target, and all five variants also share the data, the one-tick action and (except the short MLP run) the softmin
readout. So it rules out the target and the input scale. It cannot tell "the data has no action contrast at a state and
one tick changes little" from "the readout cannot carry the action". Section 7 lists a test that would separate them.

(Run note: from about 05:20 to 07:15 on 10-31 the project's SSD storage pool was full, and every 330 MB save failed.
The 20,000-step saves of stoch, zero and data and det's last 2,500 steps were lost, then resumed from the last state
file, which is written atomically and so was intact. The repeated steps were recomputed with the same recipe.)

## 6. Answers to the three candidate causes

- (i) Q(s, stop) > Q(s, slow down): **no**. The critic does not rank stop against slow-down at all (a coin flip, with
  differences below the numerical noise).
- (ii) the actor does not learn argmax Q: **true, but only because Q has no argmax**. The actor sits at the entropy-only
  optimum with a state-independent mean of about 0, which the filter turns into a freeze.
- (iii) the unimodal Gaussian: **not the limit now**. The saving action in T5 (the nominal direction, slower) is a single
  mode; the actor's form cannot matter until the critic ranks actions.
- The outside analysis's mechanism (Q learns "a, then random flailing", a loop through the stochastic target): **not
  supported**. Q sits far above the random-actor truth, and every target variant loses the action in the same first
  250 steps and is still blind at 20,000 (check 4's verdict).

## 7. What the numbers point to for the design (my reading)

1. **One 40 ms tick is too small an action for the critic to learn from this data.** The true effect of one different
   tick is small (median 0.03 in V_0.95), almost every stored action is the planner's (random ticks are 0.9 % of T5's
   planner ticks), and the critic fits y from the state within 250 steps. What matters in T5 is a whole continuation
   (0.75 apart). Make the filter's decision a temporally extended action: for example a speed scale s ∈ [0, 1] applied
   along the nominal for the next 0.5-1 s, or an action chunk. With s·a_nom, "slow down" is one number and the freeze
   is s = 0, so the actor's job becomes easy and stays near the nominal.
2. **Give the critic action contrast in the data.** The branching tool from check 2 (exact save/restore, many
   continuations from one state) generates exactly the counterfactual data the buffer lacks: the same state, different
   interventions, their real outcomes. The 500 safe demos add the slow, careful versions of each task.
3. **Let the action reach the readout.** Today Q comes from readout queries that see the action only second-hand while
   the action-carrying link heads run away above the minimum, and the residual stream grows to ~10⁴ (bound it, e.g. a
   norm after the trunk blocks). An MLP readout alone reached only 16× the softmin at 5,000 steps, still 64× below a
   clear spread, so this is needed but not sufficient.
4. **Warn earlier, and don't freeze.** In T5 a freeze saves only if it starts 0.6 s or more before the brake, a slow-down
   until about 0.3 s; the current critic turns negative at a fixed phase of the drag, 0.2-0.7 s in, also in drags that
   never tip. A freeze, and "freeze 3 ticks then resume at full speed", tip the block by themselves. With longer actions
   (point 1) one decision covers more time, which also lengthens what the backup sees per step without changing γ.
5. **Measure the action, not only the sentence.** The batch metrics (pair rank accuracy 0.98) never tested the action.
   Add the branch ranking (check 3) to every evaluation.
6. **One test separates the data from the readout before a redesign** (proposed, not run). Use check 2's branch tool to
   record counterfactual episodes: from about 100 T5 drag states (and T1 / T7 states, for contact rules), run the four
   continuations and store every tick with its tokens, as a collection episode would. Then retrain the critic on the
   normal data plus these episodes, once with the softmin readout and once with the MLP readout, and score the branch
   ranking on held-out branch states. If both rank the branches, the data was the cause. If only the MLP does, the
   readout matters too. If neither does, one 40 ms tick is too short an action, which argues for point 1 first.

## 8. Limits

- One task (T5) and one robot; 12 episodes, 11-15 planner-tipping ticks per lead-time bin.
- PhysX's contact warm start is not saved by pack(); a restore shifts the live trajectory by up to 3 mm after one tick
  and flipped the outcome at 4 of 117 ticks, so every branch (and the episode's own continuation) starts from a restore.
- Actor continuations were cut at 60-125 ticks (bounds the V_0.95 error by 0.01-0.09) and run at 15 of the 117 drag
  ticks (6 of 12 episodes).
- Check 4 changed only the target continuation (plus, in two side variants, the input scale and the readout). The data,
  the one-tick action and the readout were shared, so its verdict comes by elimination (section 5).
- The sign test on T5 decided no pair at all, so it says nothing about which way a sensitive critic would rank; it only
  shows that none of these critics ranks.
- The retrains use all the data from step 0; the round run's ring grew over its rounds. The stochastic control still
  matches the original run's action sensitivity at the same steps.

## 9. 2026-11-02: the T1-only model is just as blind (it is not from training five tasks in one model)

Question (Yuxuan): is the critic blind to the action because one model learns five tasks? Test: the check-1 tool
(`actor_stats.py`) on the final checkpoint of `rounds_t1_ur5_g095_1023`, the collision-avoidance (T1) only run. It has
the same settings and the same 79,872 steps as the five-task run `rounds_ur5_g095_1023`; only `task_choices` differs.
The five-task final checkpoint was scored again on the same number of T1 states. Job
`runs/lang/diag_1030/actor_stats/jobs/t1only.sbatch`, outputs `actor_stats/t1only/` and `actor_stats/vanilla_t1states/`,
table `scripts/lang/diag_1030/compare_t1only.py`.

What is measured: how much Q(s, a) changes when only the first command a changes, at a fixed state s and sentence:
16 random commands spread over the whole per-tick box (each joint up to its full per-tick limit), and the planner's
own command scaled by -1 to 1.5 (c·a_ref; c = 0 is a freeze). For scale, the planner's command alone moves the links a
median 2.2 mm per tick, and the farthest link 12.8 mm (p90 36 mm). Q itself is in h units, where an avoid rule's
h = 1 is 5 cm of clearance.

T1 planner ticks (128 states), final checkpoints:

| | T1-only | five-task |
|---|---|---|
| Q change over 16 random commands, median | 5.3e-5 | 1.3e-4 |
| same, largest over all states | 5.7e-3 | 5.4e-3 |
| Q change between states (std of Q over the 128 states) | 0.61 | 0.64 |
| Q change over c·a_ref (freeze ... 1.5× planner), median | 3.5e-6 | 1.3e-5 |
| states where Q(freeze) − Q(half speed) is larger than the TF32 rounding noise | 0% | 1.6% |
| readout minimum on a query head (not on an arm link) | 100% | 100% |
| lowest link head above Q, median (h units) | 8,910 | 1,780 |
| body token size, median / the action's part of it | 14,000 / 1.0 | 7,850 / 0.93 |
| actor's mean command, share of the per-tick limit | 0.2% | 1.7% |
| actor entropy (maximum 8.204) | 8.20 | 8.20 |

The actor's own ticks in T1 (64 states) give the same picture (T1-only: median change 4.1e-5, 0% above the noise, 100%
on a query head). A freshly initialized critic of the same architecture changes by 0.012 over the 16 commands (section
4), about 200× more.

Reading: in the T1-only model too, the largest change any command makes is under 1% of how much Q changes from one
state to the next, so the critic is blind to the action. Even for "don't touch X", where an arm link is the natural
place for the minimum, the readout minimum sits on a query head in every state. The link heads, the only heads that see
the command, sit about 9,000 h units above it (5× farther than in the five-task model), so they never set Q and get no
gradient. The actor again sits at the entropy-only optimum, a near-freeze. Training many tasks in one model is
therefore not the cause; what both runs share is the data, the one-tick action and the softmin readout (section 5's
caveat stands). Not run: the over-training curve of the T1-only run (not needed for this question) and a T1 branch
ground truth.
