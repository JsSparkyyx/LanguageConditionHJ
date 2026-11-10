# Why the critic ignores the action: data, time step or injection (plan)

Written 2026-11-10 with Yuxuan, for the code agent. Read `docs/lang_filter/DIAG_ACTOR_2026-10-30.md` first.

## 1. The problem and the three hypotheses

The trained critic Q(s, a) barely changes when the action a changes. On the tray-sliding task (`tray_transport`),
Q differs by about 1e-5 between first actions whose true outcomes differ by 0.49, so the actor gets no signal and
freezes. The diagnosis already ruled out the Bellman target, scaling the action inputs alone, and training five
tasks in one model. Three hypotheses are left. They can combine, so they are tested crossed.

1. **Data.** The buffer holds almost one action per state, because the planner is deterministic (random ticks are
   0.9%). Nothing in it shows that a different action from the same state gives a different outcome. Episodes at
   different speeds do not help: they are different states, and the critic can read the speed from the frames.
   What is missing is the same state near the boundary (true value close to 0) continued with different motions.
2. **Time step.** At 25 Hz one action changes the true outcome very little (median 0.03), so even with contrast
   in the data the signal may be too small. A 0.5 s action chunk makes one decision's effect much larger.
3. **Injection.** The action enters the critic as raw numbers (link translation about 0.002 m per tick) through
   one Linear layer, added to tokens of size about 1e4, and the readout sees it only through those tokens.

Out of scope: the actor, language and grounding, other tasks, the residual policy. This is a critic-only
experiment on one task and one robot (ur5-wsg). γ = 0.95 per tick everywhere.

## 2. Definitions

- **Tick**: one control step, 40 ms. **h_t**: the named rule's safety margin at tick t on the usual scale
  (1 safe, 0 harm, −1 badly harmed). Here the rule is "keep the tall block upright".
- **True value of a continuation**: the backward recursion the branch tool already uses, on that continuation's
  h trace, y_T = h_T at the last tick T and y_t = (1 − γ) h_t + γ min(h_t, y_{t+1}). Called V95 below.
- **Decision**: what the critic scores. Two lengths, H = 1 tick and H = 12 ticks (0.48 s, "chunk").
- **Speed scale c**: the planner's own path played at c times its rate for the H ticks of the decision, then the
  planner resumes at full rate from where its plan cursor stands. c = 1 is the planner itself, c = 0 a freeze.
- **Branch**: the simulator restored to a saved state and continued with one decision, then the planner, until
  everything has settled (the branch tool's rule, cap 300 ticks).

## 3. What gets built

### 3.1 Boundary branch data (hypothesis 1)

Extend the branch tool (`scripts/lang/diag_1030/t5_branch.py`, exact save and restore) into a collector.

- **States.** About 100 states from about 25 planner episodes at time scales 0.75 to 1.3, taken in the last
  second before the planner's brake. That is where the earlier truth table puts the boundary (planner value
  0.15 at 0.6 s or more before the brake, −0.10 at 0.4 to 0.6 s).
- **Branches per state (7).** The planner (c = 1), two 1-tick decisions (c = 0 and 0.5), four chunk decisions
  (c = 0, 0.25, 0.5, 0.75).
- **Stored.** Every tick of every branch in the normal shard format (tokens, robot inputs, h, the commanded joint
  step), plus a branch table: state id, episode, tick, decision length H, c, the per-tick commands of the
  decision, V95, min h, whether the block fell.
- **Split.** By episode: 20 episodes for training, 5 held out for scoring.

The continuation after every decision is the planner, so the data follows one Markov rule ("decision, then
planner") and matches the normal buffer, which is all planner.

### 3.2 A critic that takes a chunk (hypothesis 2)

- **Input.** A chunk A = (dθ_1 … dθ_12), the commanded joint steps of 12 ticks. Summarize it at three points,
  the cumulative steps after 4, 8 and 12 ticks: c_k = dθ_1 + … + dθ_k. Per body token, the features are the
  token's own joint entry of c_k, the link translation J_lin·c_k and the link rotation J_ang·c_k for the three
  points, plus the valid flag (22 numbers in place of today's 8). J_lin and J_ang are the link's Jacobians at the
  current state, so this is a linearization over 0.5 s. The post-action link position uses the last point.
- **Rows.** Any tick whose next 12 ticks are in the same episode starts a chunk (sliding window).
- **Target.** Bootstrap at the chunk's end and unroll the one-tick recursion back over the chunk:
  y_{t+12} = mean of the target twins at (s_{t+12}, the stored next chunk), then
  y_{t+k} = (1 − γ) h_{t+k} + γ min(h_{t+k}, y_{t+k+1}) for k = 11 … 0, and the target is y_t. If the episode
  ends inside the chunk, start the recursion at its last tick as the one-tick code does.

### 3.3 Revised injection (hypothesis 3)

In `main/lang/` only (subclass `LangCriticHead`, new config switch, old checkpoints still load).

1. RMSNorm on the body tokens before the action is added.
2. Action features divided by their bounds (own joint step by dθmax times the number of ticks covered,
   translations and rotations by fixed reference sizes measured once at a saturated action, as
   `critic_retrain.py`'s ActionNorm does), then MLP(features → 256 → 256) and a final RMSNorm.
3. The mean action embedding over the valid body tokens is concatenated into the readout MLP's input.

All runs use the MLP readout (the current default), so the readout is not a hidden variable.

## 4. Steps

### Step 1. Free check on the data we already have (no simulator)

Question: can the critic fit action differences at all when they are given directly?

- Data: the 117 drag states with stored encoder inputs (`runs/lang/diag_1030/t5_branch/states/`) and their
  truths (`results.json`), three first actions per state with clearly different commands: the planner's, zero,
  and half speed's first command, each with the V95 of its whole continuation.
- Fit: start from the vanilla final checkpoint, regress Q(s, a) on V95 (Huber loss, no Bellman), everything
  trainable. Two variants: current injection, revised injection. 3 folds by episode (8 train, 4 held out).
- Report: rank agreement among the three actions and Q spread against the true spread, on training and held-out
  states.
- Reading: if the current injection fits and ranks held-out states, the critic can carry the action and
  hypothesis 3 is not the main cause. If only the revised one does, use the revised injection from here on. If
  neither fits even the training states, stop and report before spending simulator time.

Cost: minutes on any GPU. If the state files are gone, skip this step and do the same fit on step 2's data.

### Step 2. Collect the boundary branch data (A40 jobs)

Start it alongside step 1, since the queue is slow. First a smoke job: 2 states, all 7 branches, to measure
seconds per branch and check that a restored branch reproduces the planner's own continuation. Then size the
collection to the budget in section 6 and tell Yuxuan the number of jobs before submitting.

### Step 3. Train the critics (any GPU partition, resumable 1-hour jobs)

Critic only: no actor loss, the next action or chunk in the target is the stored one. Same settings as the
vanilla run otherwise (`scripts/lang/diag_1030/critic_retrain.py` is the starting point), 6,000 steps each,
probe every 250 steps.

| run | decision | injection | data | tests |
|---|---|---|---|---|
| existing | 1 tick | current | normal | reference, already blind (the stored-next-action retrain) |
| A | 1 tick | current | normal + branch | data alone |
| B | 1 tick | revised | normal + branch | data + injection |
| C | chunk | current | normal + branch | data + chunk |
| D | chunk | revised | normal + branch | all three |
| E | chunk | current | normal | chunk alone |
| F | 1 tick | revised | normal | injection alone |

"Normal" is the vanilla run's warmup and ring, restricted to the tray-sliding task. Branch rows are sampled so
that they make up a quarter of each batch.

### Step 4. Score and decide

On the held-out states, score every branch's decision with each critic. Single-tick critics score the 1-tick
decisions, chunk critics the chunk decisions, and both score the planner's.

- **Rank agreement**: over pairs of branches of one state whose V95 differ by more than 0.05, the share where Q
  orders them as the truth does (chance 0.5). Report how many pairs that is: for 1-tick decisions it may be few,
  which is itself hypothesis 2's point.
- **Spread ratio**: the spread of Q over a state's branches divided by the spread of their V95.
- **During training**: the spread of Q over random actions or chunks on fixed probe states, to see whether the
  action survives the first 1,000 steps.

A critic "reads the action" if rank agreement is at least 0.70 and the spread ratio at least 0.25. Fix these
numbers in a file before step 3 starts.

| result | conclusion |
|---|---|
| A passes | data was the main cause, boundary collection is the fix |
| A fails, B passes | data and injection |
| A and B fail, C or D pass | the one-tick step is the limit, chunks are needed (D vs C says whether the injection matters too) |
| E or F pass without branch data | that change alone is enough |
| nothing passes | the critic cannot learn it from this data, report before going further |

Write the results to `docs/lang_filter/ACTION_SENSITIVITY_REPORT.md` in plain words, with the table above filled
in.

## 5. Tools that already exist

- `scripts/lang/diag_1030/t5_branch.py`: exact save and restore, the planner / freeze / half speed / smooth stop
  continuations, settle rule, V95.
- `scripts/lang/diag_1030/critic_retrain.py`: critic retraining with a chosen target action, fixed probe set,
  ActionNorm, resumable with exit code 75.
- `scripts/lang/diag_1030/score_t5.py` and `score_states.py`: scoring checkpoints on branch states, rank agreement.
- Cluster usage: `residual/PLAN.md` section 5.

## 6. Compute

- Step 1: no simulator, minutes of GPU.
- Step 2: A40 only (cuRobo and ray tracing). Target at most 10 one-hour jobs. If the smoke job says 100 states
  do not fit, reduce the number of states, not the branches per state.
- Step 3: six runs of 6,000 steps. They read stored tokens only, so they can use the A100 partitions.
- Tell Yuxuan the job count before each submission round.
