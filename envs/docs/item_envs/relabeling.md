# Rule labels and relabeling (vanilla text-conditioned filter)

The filter is conditioned only on a sentence; text conditioning is learned by
relabeling — the same transition trained with different rules of the same
scene, whose h differ. So every tick stores h of **every rule of the
episode**, not only the rule it was collected under. Sentences are not
stored per tick: they are generated at sample time from the rule and a
paraphrase bank. Spec: the user's "Rule labels and relabeling" note
(2026-10-05); this page is what is implemented and where it differs.

## What is stored

### Per tick — in the lockstep sidecar (`item_buffer/`)

The fields live in `ItemSidecarBuffer` (main/item_envs/item_buffer.py), not
in `StepBuffer.FIELDS`: the sidecar is written in lockstep with the
StepBuffer (slot i of either ring is the same transition), so the CEHJ
StepBuffer stays byte-identical for the non-item tasks.

| field | type | meaning |
|---|---|---|
| `n_violating` | int16 [r_max] | beads breaking each bead rule this tick, -1 for other rules (2026-10-07, optional) |
| `h_rules` | float32 [r_max] | h of each rule this tick, the env definition `margins.rule_h` = clip(margin / h_unit, -1, 1); bead rules below 0 count beads (`margins.particle_h`, see below) |
| `fails_soon` | bool [r_max] | min(h_rules[t : t + 25, j]) < 0 within the episode (`FAILS_SOON_H` = 25 ticks = 1 s) |
| `rule_valid` | bool [r_max] | True for columns < the episode's rule count (padding is never sampled) |
| `active_rule` | int16 | the episode's active rule column, -1 = none (h = min over all rules) |
| `t_in_ep` | int16 | tick index in the episode (rows t - k and t - t_in_ep are the previous / first frames) |
| `episode_id` | int32 | StepBuffer's episode_id (copy) |
| `margins` | float32 [r_max] | the raw SI margins (kept) |

StepBuffer `h` = `h_rules[active_rule]`, or the min over all rules when
`active_rule` = -1 — which is the collection default: collection protects
every rule at once and the sentence (the referent) is chosen at training
time. `done` keeps its meaning; a relabeled rule can go
negative before `done` — the HJ target's min(h, ·) handles that.

`fails_soon` is filled AS THE EPISODE RUNS (collect.py has no end-of-episode
hook): a rule negative at tick t marks rows t - 24 ... t of the episode.

### Per episode — `item_buffer/episodes/{episode_id}.json`

`language.episode_record(runner)`; `episodes.jsonl` (logger dicts) is still
written too.

```json
{"episode_id": 1234, "task": "push_zone", "embodiment": "ur5-wsg", "seed": 3,
 "active_rule": 3, "active_instruction": "don't disturb the cylinder",
 "objects": [{"id": 0, "item_id": "push_0", "category": "box", "category_key": "box",
              "synonyms": ["crate"], "color": "yellow", "size": null, "role": "candidate", "kind": "rigid"}, ...],
 "regions": [{"id": 1, "region_id": "zone_red", "kind": "zone", "names": ["zone", "mat", "marked area"],
              "color": "red", "role": null, "nameable": true}, ...],
 "rules": [{"idx": 1, "rule_id": "outside|push_0|zone_red", "consequence": "zone_exclude",
            "subject": {"object": 0}, "reference": {"region": 1}, "params": {"h_unit": 0.05},
            "set": false, "split": "train", "ambiguous": false, "sampleable": true}, ...]}
```

- `rules[idx]` = column `idx` of the per-tick arrays (the episode's rule
  table order).
- Particles: `"subject": {"particles": <object id>}`, `params.material` =
  the material words (beads, rice, beans); containers are `reference`
  regions.
- `sampleable`: a training-split, single-referent rule with at least one
  training sentence in this scene (its referent has a unique name).
- `size` words: T5 / T7 blocks ("tall" / "short", `register_item(size=)`);
  episodes recorded before 2026-10-05 have none.

## h per rule (env definition, 2026-10-05)

One scale for every rule (`margins.rule_h`): 1.0 = comfortably safe, 0 =
harm, -1 = badly harmed, clipped.

Tasks used for training and evaluation (Yuxuan, 2026-10-23): T1, T3, T5,
T6, T7. T4 was dropped on 2026-10-22; T2 was replaced on 2026-10-23 by
the redesigned T6, because its one rule is not text-conditioned. Both stay
built and are kept in the table.

| task | rules in an episode | consequence | h |
|---|---|---|---|
| T1 selective_avoid | per extra object: avoid, stay; set rules (category, all) | `avoid`, `stay` (3 cm, 20 deg) | avoid: (real-shape gap - 1 cm) / 5 cm |
| T2 granular_pour (not used from 2026-10-23) | one | `contain_particles` | safe: depth of the worst bead inside the columns / 1 cm, clipped to [0, 1]; violated: by bead count (below) |
| T3 push_zone | table, per mat (+ pairs), per standing object upright + stay (+ pair) | `zone_include`, `zone_exclude`, `upright` (45 deg), `stay` (1.5 cm, 45 deg) | zones / 5 cm; upright (45 deg - tilt) / 45 deg; stay (1.5 cm - moved, tilt as movement) / 1.5 cm |
| T4 granular_carry (not used from 2026-10-22) | one | `contain_particles` | as T2 |
| T5 tray_transport | per block (2 from 2026-10-22; 2-3 before) | `upright` (45 deg) | (45 deg - tilt) / 45 deg |
| T6 clean_pour | per non-target bowl (from 2026-10-23; per bin / plate before), plus the spill rule | `exclude_particles`, `contain_particles` | safe: distance of the closest bead from entering the column / 5 cm, clipped to [0, 1]; violated: by bead count (below) |
| T7 gap_place | per standing block: upright + stay | `upright` (45 deg), `stay` (2 cm, 45 deg) | as T3 |

Tilt limits (Yuxuan, 2026-10-23): 45 deg in T3, T5 and T7, for `upright`
and for the tilt part of `stay`, where tilt counts as movement (45 deg of
tilt counts like the whole movement limit). Until 2026-10-22 both were
10 deg (in T3, and in T7's stay, since 2026-10-05; before that T3 had
upright 45 deg and stay 3 cm / 20 deg, and T7's stay 20 deg). T1's `stay`
keeps 3 cm / 20 deg. Example: a T7 block
moved 1 cm and tilted 9 deg reads stay h = min((2 - 1) / 2,
(45 - 9) / 45) = min(0.5, 0.8) = 0.5, and upright h = (45 - 9) / 45 = 0.8.

T6 has two designs under the one key `clean_pour`: three bowls from
2026-10-23 (`robotwin_tasks/bowl_pour.py`, regions `bowl_<colour>`) and
bins / plates before (`robotwin_tasks/clean_pour.py`, regions `bin_*` /
`plate_*`). A bowl's column is its footprint, capped 1.2 cm above the
rim and floored half a bead above the inside floor (`Region.z_floor`): a
bead on the table against the bowl's outside wall is not in it.
`main/lang` skips the old design's episodes (`LangConfig.drop_old_t6_rows`).

Bead rules (`contain_particles`, `exclude_particles`; user 2026-10-07): the margin is the hard
minimum over beads (it was a soft-min, T = 1 mm in the T2 / T4 / T6 yaml and 5 mm by default, which
could read < 0 with no bead out; the hard min took effect only at commit deb43e7, review B1), and
once n >= 1 beads break the rule, h = -(0.5 + 0.5 (min(n, 20) - 1) / 19): -0.5 for one bead, -1 at
20 or more, whatever their depth (`margins.particle_h`). The per-tick n is stored as
`n_violating` (int16 [r_max], -1 for other rules; optional, so older buffers still open) and in
`trace.npz` as `violation_counts`. Buffers written before 2026-10-07 hold the old bead h and no
counts, so their T2 / T4 / T6 rows cannot be relabeled.

Differences from the spec's table (all intended): the spec predates the
2026-10-05 changes (real-shape distance + 1 cm buffer, new T3 / T7 limits,
the shared h scale — the spec's suggested `h_gain` 3 for T5 is no longer
needed: the env already reads 1.0 upright, so `h_gain` defaults to 1); the
rule tables hold more rules than the spec's candidates (T1 stay, T3 table +
pair rules, T6 spill rule, T7 upright, set rules), hence `r_max` = 64, not
8. Set / pair rules are recorded but never sampled (no paraphrases).

Consequence names: `avoid`, `upright`, `stay` as in the rules; `outside` ->
`zone_exclude`, `inside` -> `zone_include`, `particles_inside` ->
`contain_particles`, `particles_outside` -> `exclude_particles`
(`language.SPEC_CONSEQUENCE`).

## Language

- `main/item_envs/language/paraphrases.yaml`: 5 plain sentences per named
  consequence, 4 train / 1 held out (2026-10-20; before, 30-31 per consequence),
  split by TEMPLATE (a held-out template is never used in training with any
  object); zone_include has none (never named), and from 2026-10-23 neither
  has contain_particles (the spill rule: recorded, never named); every
  rules.py template and every splits.yaml held-out template are in it
  (test). `heldout_words`: referent nouns never used in training
  (2026-10-23: plastic bottle, goblet, milk, fruit, cube, serving tray,
  little balls, and container for T6's bowls; earlier lists had flask,
  glass, pillar, saucer, grains, ...).
- `language.describe(obj, scene)`: "the {color} {noun}", "the {noun}", "the
  {size} {noun}", "the {size} {color} {noun}", the noun from the category
  word and its synonyms — only phrases NO OTHER object of the scene answers
  to (two cups: "the cup" is invalid, "the red cup" is required).
  `describe_region`: "the red zone", "the green bowl" (T6 from 2026-10-23;
  "the green bin" before), "the table" — unique among regions and not an
  object's name.
- `language.sentences(rule, scene, split)`, `sample_sentence(...)`.
  Splits: `train`, `heldout_template`, `heldout_word`.

Nameability: before 2026-10-05 only 128 of 287 rules of 30 T1 episodes
were sampleable — set rules, and scenes with several uncolored objects of
one category (three plain cups: none has a unique name). T1 now draws its
candidates from 10 types without repeating a category or a host object's
(user), so every candidate is nameable. Every T2 / T4 / T5 / T6 / T7 rule
is sampleable; T3 95 of 164 (the pair rules are not). From 2026-10-23 the
spill rule (T2, T4, T6) has no sentences, so it is never sampleable: T2
and T4 have no sampleable rule, and T6 names only its bowl rules.

## Text cache

`scripts/item_envs/build_text_cache.py --episodes <item_buffer>/episodes
--out <file>` (or `--metas <glob>` for saved probe runs): every sentence
of every usable rule in the three splits, embedded once with the frozen
HoloBrain BERT + `text_feat_map` (`language/text_encoder.HoloBrainText`:
robo_orchard_lab's bip3d BertModel forward with the checkpoint's own
weights and its model.config.json settings; identical to the earlier
test-only pipeline, max |diff| 0.0). Output: `{"index": {sentence: i}, "emb": fp16 [N, 24, 256], "mask":
bool [N, 24], "split": [...]}`. All 2181 calibration episodes give 9852
sentences (6080 train, 1708 held-out template, 2064 held-out word), 7-12
tokens each. Note: mean-pooled, "the green block" and "the yellow block"
are 0.988 similar — the filter needs the per-token embeddings (kept).

## Sampling (`main/item_envs/relabel.py`)

`RelabelSampler(item_buffer_root, split="train", harmful_frac=0.5,
pair_frac=0.5, h_gain=None, k_prev=1, text_cache=None).sample(B, rng)`:
row t + its next row (or terminal), a sampleable rule (harmful = 
`fails_soon` / benign about 50/50 when both exist), a sentence (train
templates + train words), h = clip(gain x h_rules[t, rule]), h_next at
t + 1, `prev_row` (t - k, clamped to the episode start), `first_row`;
`pair_frac` of the batch is the same transition twice with a harmful and a
benign rule. A text-cache miss raises. `benign_rules(episode_id)`: the
rules whose `fails_soon` is never set (benign-scene intervention rate).

Evaluation splits (`RelabelSampler(split=...)`): `train` (seen templates
and referents), `heldout_template`, `heldout_word`, `heldout_item` (rules
naming a splits.yaml held-out color / category — never sampled in
training — with the training templates), and benign scenes
(`benign_rules`).

## Checked end to end (2026-10-05, `runs/diag/relabel/`)

collect.py's collection path, 2 T7 ur5 + 2 T1 franka episodes into fresh
buffers: t_in_ep / episode_id per row right; `fails_soon` equal to a
recomputation from `h_rules` on every row and rule; sign(h_rules) =
sign(margins) everywhere; size words in the T7 records; the sampler draws
paired harmful / benign rules on one tick ("don't bump the green brick",
h -0.15 / "keep the green block on its base", h 1.0); the text cache built
on the GPU (1512 sentences from the two T7 episodes). Driver for one item
task (collect.py's CLI schedules only `cfg.task_choices`):
`cfg.task_choices = (task,)` then `collect(cfg, out)`.
