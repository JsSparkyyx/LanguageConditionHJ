# T8 rounds training with a 3-tick decision, the revised action injection and recovery data (Yuxuan, 2026-11-17)

## 1. Why

The T8 boundary check (`scripts/lang/diag_1030/t5_branch.py --task tray_slide`, 2026-11-17) branched a medium drag
0.46 s before the brake. Following the plan, stopping dead, or halving the speed tipped the glass. Slowing to zero over
1 s kept it standing. Q(s, a) of the 1115 T8 runs was the same for all of these, and for 16 random actions, to 4
decimals. The actor therefore gets no gradient from Q and only stops. This is the action-sensitivity problem of
`ACTION_SENSITIVITY_PLAN.md`, `ISSUES.md` issues 1 and 2: the one-tick decision (Yuxuan 2026-11-11: use 3 ticks), the
linear injection, and data with one action per state.

## 2. Symbols

- Tick: one control step, 40 ms (25 Hz). H = `decision_ticks` = 3: one decision covers 3 ticks (120 ms).
- dθ_t: the stored per-tick joint command of row t. a: the actor's per-tick output.
- h_t: the sampled rule's h at row t. γ = 0.95 per tick.

## 3. What changes (`main/lang` only)

- **Critic action** = the joint displacement over the decision: Σ dθ_t..t+2 for a stored row, 3a for the actor (its
  action is executed as 3 equal ticks). `network.stored_decision`, `policy_decision`.
- **Target** (`network.decision_target`): y_3 = mean of the target twins at (s_t+3, 3 a(s_t+3)), then
  y_k = (1 − γ) h_k + γ min(h_k, y_k+1) for k = 2, 1, 0. Example: h = (1, 1, −1), y_3 = 1 gives −0.805: the harm at
  the third tick is not missed. With H = 1 it is CrossSafe's target.
- **Data** (`data.LangSource`): the "next" frames are those of row t+3. A row is usable only when t+3 is in its
  episode, so the last 3 rows of an episode drop out. A batch adds `dtheta_seq` [B, 3, A] and `h_seq` [B, 3].
- **Revised injection** (`critic_injection="revised"`, MLP readout): the bounds come from the run's own data at the
  start (`train.action_reference`, `<run>/action_ref.json`): 3 × one tick's.
- **Filter** (`rollout.LangRolloutController._tick_hook_decision`): every tick it scores Q(s, the planner's
  3-tick decision) and engages when Q < 0 (unchanged). An engaged block queries the actor once and holds its action
  for the 3 ticks (`hj_hold_ticks` = 3). The release test runs at the next hook. The buffer still stores each tick's
  own command.
- **The planner's 3-tick decision in the filter** (`rollout.nominal_decision`, fixed 2026-11-17 before any run used
  it for a result): the sum of the 3 per-tick commands the planner is about to issue, because that is what training
  reads from the stored rows. A stored planner command is the target at the tick's end minus the live joint state
  at its start, so it holds the planned step plus the arm's tracking lag. In the T8 data the arm realizes about 80%
  of each command, so the lag is about a quarter of a step, and the sum over 3 ticks holds it 3 times. The filter
  knows only the current lag, lag = pos[i − 1] − q, and uses it for all 3 ticks:
  decision = (pos[i + 3·10 − 1] − q) + 2·lag. On the stored nominal episodes this is within 1-2% of the summed
  commands (median; 7-8% at the 90th percentile of the fast rows). The first version used the endpoint alone,
  pos[i + 29] − q, which is 20-43% smaller than what the critic was trained on. Example: a cruise of 0.06 rad per
  tick with a lag of 0.02 rad gives stored commands of 0.08 rad and a decision of 0.24 rad; the endpoint alone
  would be 0.20 rad.
- Probes, the offline evaluation and the counterfactual-sentence trace use the same decision.
- Old configs read `decision_ticks = 1`; nothing changes for them.

## 4. Data and runs

- Warmup: `data/lang_t8_rec`: the first T8 warmup (60 episodes, kicks 0.001) plus the recovery warmup (60 episodes,
  no kicks: `recover_glass` with 12, 8 and 5 cm slow-downs, `recover_can`). Validation: the T8 set (20 episodes).
- Rounds: T8 only, ur5-wsg, γ 0.95, no kicks, 100 rounds × 4 episodes × 2048 steps, probes every 10 rounds.
  Nominal mix: medium 0.4, fast 0.2, slow 0.15, recover_glass 0.15, recover_can 0.1 (the plain recoveries only, so
  "in the way" stays right).
- Two runs: the argmax actor only; the argmax and the boundary actor.
- The 1115 runs (1-tick, linear injection) are stopped; their checkpoints stay as the reference.

## 5. Checks

- Unit tests: `tests/lang/test_decision_ticks.py`.
- Per round: the filter's harm and intervention rates, the actor's mean |a| when engaged (does it still only stop?),
  `critic/action_sens_rel`.
- The boundary check on the new checkpoints, with 3-tick first actions: does Q rank stop / half / slow like the
  measured outcome?
