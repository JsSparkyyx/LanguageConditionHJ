# T8-only rounds training with a second, boundary-seeking actor (Yuxuan, 2026-11-15)

Request: "a task only for T8; the nominal controller randomly chooses a velocity setting per rollout; in addition to
the argmax actor, train an actor whose actions keep the arm near 0.1 > h > 0, so it stays near the boundary; in
online training, each training rollout randomly chooses one actor."

## 1. Symbols

- s: the state (camera tokens of frames t, t-2 and 0, robot tokens, the sentence). a: a per-tick joint displacement.
- h: the named rule's failure value in h units, `clip(margin / h_unit, -1, 1)`. For T8 the rule is `upright`
  (45° limit): h = (45° - tilt) / 45°. Example: tilt 9° gives h = 0.8; tilt 40.5° gives h = 0.1.
- Q(s, a): the twin critic, min(Q1, Q2), trained toward `y = (1-γ) h + γ min(h, Q_targ(s', π(s')))`, γ = 0.95. It
  estimates the discounted worst future h when a is taken now and the argmax actor π acts afterwards.
- π (the **argmax actor**, the current one): maximises Q(s, π(s)) - α log π. It is the filter's safe actor and the
  only actor in the critic's target.
- β (the **boundary actor**, new): same architecture, its own weights and optimiser. It is trained so that
  Q(s, β(s)) lies in the band [0, 0.1].

## 2. What "near the boundary" means here (the band is on Q, not on h)

For T8, h itself is no use as the band: the glass tips over long before 40.5° (its tipping angle is about 10-15°).
So h stays near 1 while the glass stands and then drops to -1 within a few ticks. A state with 0 < h < 0.1 (tilt
40.5-45°) is a glass already falling.

Q has the band instead. With h = 1 until a fall k ticks ahead and h = -1 after, the target recursion gives
Q = 1 - 2·0.95^k. Q = 0 at k ≈ 13.5 ticks (0.54 s) and Q = 0.1 at k ≈ 15.5 ticks. Q in [0, 0.1] therefore means:
"the safest continuation still keeps the glass up, but only just". That is the filter's switching surface
(engage at Q(s, a_nom) < 0). Data there is what the filter's decision depends on, and the argmax actor alone rarely
goes there, because it moves away from the boundary as fast as it can.

## 3. Nominal: a random velocity per rollout (already in the code)

Every T8 episode draws its nominal behaviour from its seed (`config.episode_risk`, `LangConfig.behaviour_mix`,
bd4d0b3): `slow` 0.25 (0.15 m/s, nothing falls), `medium` 0.5 (0.45 m/s, the glass falls), `fast` 0.25 (0.70 m/s,
both fall). Each drag has one speed, and the speed varies between episodes. The same seed always gets the same
behaviour, so a resumed or re-evaluated episode replays it. No code change; another mix is a `--set
behaviour_mix=...` (the two recover_* behaviours are available).

## 4. The boundary actor

- **Loss** (one more actor update in each grad step, after the argmax actor's, on the same detached encoder
  features; the critics frozen, as for π):

  `L_β = mean( relu(Q(s, β(s)) - 0.1) + relu(0 - Q(s, β(s))) + α log β + 1e-3 · pen )`

  The hinge is zero inside the band and has slope 1 outside it, the same gradient scale as π's -Q. α (the entropy
  weight, same schedule as π) keeps β stochastic inside the band. `pen` is the same pre-tanh penalty as π's.
  Example: Q(s, β(s)) = 0.4 gives loss 0.3 and pushes β toward actions with lower Q; Q = -0.2 gives loss 0.2 and pushes
  toward safer ones; Q = 0.05 gives no hinge loss.
- **Critic target unchanged**: y keeps the argmax actor π at s'. β never enters a target, so Q stays the value of
  the safest continuation, and β only changes which states and actions the data covers.
- **Not deployed**: the filter in evaluation (`eval_closed`, `eval_offline`) uses π only. β is a data-collection
  policy.

## 5. Rollouts: one actor per training episode

- Rounds collection is unchanged apart from the actor: 80% of episodes are filter episodes
  (`filter_episode_frac`). Each filter episode uses β with probability `boundary_actor_frac` (0.5) and π
  otherwise. The draw comes from the episode's seed on its own stream, so a resume repeats it.
- The filter's semantics stay those of CrossSafe: engage when Q(s, a_nom) < 0, the chosen actor drives the 3-tick
  hold, re-queried every tick, and the release test runs at each block boundary. With β, an engaged block steers to Q
  in [0, 0.1] instead of as far from the boundary as possible, and the nominal takes over again as soon as
  Q(s, a_nom) >= 0.
  - Example: a `medium` drag (0.45 m/s). Near the pad the brake would tip the glass, and Q(s, a_nom) drops below 0.
    π would stop the arm hard and early. β slows the arm only until Q is back at 0-0.1, then releases.
- The actor never drives the gripper, as before. β acts only while the filter is engaged, never before the tray is
  pinched, because the filter does not engage there.
- Recorded per episode: `actor` = "argmax" or "boundary" in `lang_episodes.jsonl` and in `eval/rounds.jsonl`. The
  round summary adds the filter-episode means per actor (`round_overview/actor_<name>/...`). Buffer rows keep
  action_source 1 (actor), and the episode line tells which actor.

## 6. Config and compatibility

- `LangConfig.boundary_actor: bool = False`, `boundary_lo = 0.0`, `boundary_hi = 0.1` (h units),
  `boundary_actor_frac = 0.5`. Off by default: a config without the fields loads with them off, and nothing changes
  for existing runs. β's initial weights come from a forked RNG (seed + 0xB0D1), so a run with β starts with the same
  encoder, critics and argmax actor as one without it.
- Checkpoints and `state.pt` gain `actor_boundary` / `opt_b` only when it is on. Evaluation ignores them.
- Diagnostics: `actor_b/q_mean`, `actor_b/in_band_frac` (share of batch states with Q(s, β(s)) in the band),
  `loss/pi_b`.
- Code only in `main/lang/` (config, network, train, collect, rounds, metrics); no CrossSafe or `envs/` file
  changes.

## 7. Runs (ur5-wsg, GD encoder, γ 0.95, the 1023 rounds settings otherwise)

Random kicks (Yuxuan, 2026-11-15): the nominal's random joint kicks (`perturb_prob`, CrossSafe's 0.05, a 2-tick
random displacement at full joint speed) are set to 0.001 for T8, in the data (`collect --perturb-prob 0.001`) and in
the rounds runs (`--set perturb_prob=0.001`). At 0.05 a kick while the arm held the tray tipped both items: in the
first warmup (kept in `data/lang_t8_kick05`) the glass and the can fell in 5 of 12 slow drags, where the card's robot
check has none, and the can in 10 of 32 medium drags, where it has none; every one of those falls followed a kick
by 2-12 ticks.

1. Validation: 20 T8 episodes, g 26 (the command in `main/lang/collect.py`'s docstring), into
   `data/lang_t8/val/shard_g26`.
2. Warmup: 60 T8 episodes (2 shards x 30, g 40 and 41), nominal only, no masks, into
   `data/lang_t8/warmup/shard_g40`, `shard_g41`. Then the text caches and `check_shards.py`, as in the smoke data.
3. Rounds, `--tasks tray_slide`, two runs from the same warmup:
   - argmax actor only (the reference);
   - argmax + boundary actor (`--set boundary_actor=true`).

   100 rounds x 4 episodes x 2048 steps, probes every 10 rounds on the T8 validation set.
   Compare per round: the filter's harm rate and intervention rate under π (the deployed actor), v_le_h_frac,
   and the probes' rank accuracy between the glass and the can rules.

## 8. Tests (CPU, before any job)

- The β loss: zero gradient inside the band, and the sign of the gradient pushes Q toward the band from either
  side.
- The actor draw: the same seed gives the same actor, and the frequency is about `boundary_actor_frac`.
- Checkpoint and state round trip with β; a config without the fields loads with β off; the existing tests pass.
