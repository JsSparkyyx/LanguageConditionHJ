# Response to the code review, round 4 (at commit 6504fcd)

Code at 0e97772 and after. Grounding runs and numbers: `docs/lang_filter/grounding_1008.md`.

| item | status | where |
|---|---|---|
| 1. V3 scored before the CKPT_PARTS fix is invalid | done. v3_s0 re-exported from state.pt (e_tag norm 2.49) and re-scored; its tagless rows are kept apart as `grounding_noetag_INVALID`. v3_s1 resumes on a5b6055, which writes e_tag. grounding_summary refuses a V3 row whose scored checkpoint has no e_tag (from metrics.json `data.ckpt_parts` when recorded, else from the scored checkpoint file). | `scripts/lang/grounding_summary.py` `run_settings`, `PARTS` |
| 2. V0-V4 comparable | done. V4 / V4b run from the frozen checkout `_exp3` (0e97772) with `--set gamma=0.95 gamma_final=0.95 actor_hidden=0 recall_horizon_ticks=7`: V0's settings, same data, validation, seeds and steps. The summary refuses runs whose gamma, actor_hidden, tf32_train, steps, training tasks or embodiments differ from V0's, and also runs scored on another set (val_roots, scored tasks and embodiments). It writes a per-run settings table (commit, gamma, actor_hidden, tf32_train, steps, recall, switches, aux settings, aux ratio). | `grounding_summary.py` `FIXED`, `SCORED`; `runs/lang_grounding_1008/e2.cmds` |
| 3. V4 loss too weak | done. Prediction and target divided by `aux_unit` = 0.05 m, Huber delta 1 in those units; `aux/ratio` = λ·L_aux / L_critic logged. λ is set from a 600-step calibration run at λ = 1: `scripts/lang/aux_calibrate.py` takes the median ratio over steps 400-600 and sets λ = 0.2 / ratio. No V4 had run with the old scale (the queued lines were commented out first). The summary treats a V4 whose median ratio over steps 400-600 is below 0.05 as not tried. V4b (`aux_attention_loss`, default off) is implemented as specified: q1's head-averaged scene attention in the last encoder block, renormalized within frames t and 0, times the squared distance to the named item's centre in aux_unit, with its own calibrated weight. | `main/lang/train.py` `_grad_step`; `network.py` `keep_q_attn`; `config.py` |
| 4. E0b reading | done. grounding_1008.md and the page judge by the paired difference any - named and its interval: T1 +0.12 [0.02, 0.20] overall, +0.17 [0.03, 0.29] before contact, +0.13 [0.04, 0.21] when no harm is near; T7 +0.07 [0.02, 0.14] overall, inconclusive per stratum; every correlation weak (at most 0.33). The page derives the sentence from the numbers (sign, intervals, strata). | `grounding_page.py` `seeing_reading` |
| 5. Page numbers | done. Section 2 reads every number from the metrics files and states the set under the table (T1 / T7: the ur5 grounding set; T3 / T5: `val20_e0` once scored, until then the round-3 set, both robots, stated). Section 4 reason 2 is built from word_attention.json (both critics' last blocks; the start token constant, the end token not). Section 6 fallback: "neither the text fixes nor V4". | `grounding_page.py` `section2`, `special_reading`, `section6` |
| 6. recall horizon from gamma | done. `recall_horizon_ticks = None` → round(ln 0.5 / ln γ): 7 at 0.9, 14 at 0.95; saved configs keep their value. The offline tables print the horizon used. | `config.py` `recall_ticks` |
| 7. Actor head | done. New runs default to the MLP (actor_hidden 256). actor_hidden is printed in the offline tables' title, the closed-loop table header and metadata, and report.py groups offline and closed-loop tables by it (from the run's config when metrics.json lacks it). | `eval_offline.py` `model_line`; `eval_closed.py`; `report.py` |

## Audit of the grounding code (2026-10-08 night)

Five readers covered the data and relabeling, the text path, the evaluation and pairs, the probes and traces, and
round-4 compliance; a skeptic re-checked each finding. Confirmed and fixed:
- The summary's bootstrap order followed Python's string hashing, so intervals changed between runs: keys
  are now sorted. The "T1 + T7" group now filters its tasks explicitly.
- The page printed an interim outcome ("run V4 next") as a conclusion: the summary now writes a status
  (pending, final_a, final_b, final_slots, final_other, inconclusive_oracle) and the page words the header
  and section 6 from it.
- Zone rules: their subject is the pushed box but the sentence names the zone, so V3 / V4 centres marked the
  box. zone_exclude now targets the mat's centre. zone_include gets no target, because the table's frame
  sits at z = 0 and the top at 0.74 m. T1 / T7 results are unaffected.
- `bead_h_of` could not report old-h shards when drop_stale_bead_rows is off: fixed.
- Page claims not backed by measurement removed (the start token in "many heads"; "the last critic block" =
  only the second critic), the task-1 "already harmed" bullet now uses the different-object pairs (0.56,
  wrong in 44%), the frozen [SEP] is a sentence summary, not constant (only [CLS] is isolated by the
  sub-sentence mask).

Found, with a bearing on the results (open):
- **V3 does not work as a ceiling.** Zeroing its learned tag changes 30 of 1735 primary outcomes (0.580 with
  the tag vs 0.570 without). Every variant ranks its own training batches' pairs at 0.95-0.97, against
  0.56-0.61 on validation. The model fits its 60 training scenes (30 T1 + 30 T7 ur5 episodes) without
  needing the tag. The summary now checks this: V3 must beat V0's interval and its zeroed tag must change
  at least 5% of the primary outcomes, or the outcome is "inconclusive_oracle", not "the bottleneck is not
  grounding". Being measured tonight: the training-episode scores, V3 s1's zeroed tag, V0's validation
  learning curve, and V0 trained on both robots' T1 / T7 episodes (120 scenes).
- **Referent phrases that are not unique to a reader** (needs a decision from Yuxuan: it is a change under
  `envs/`). `language._answers` rejects a phrase only if another object has exactly that noun, so "the box"
  passes next to a "coffee box" and "the bottle" next to a "pill bottle". 153 of the 1279 T1 already-harmed
  referent pairs use such a phrase (none in T7). They are not why the score is low: rank accuracy on them is
  similar to the clean pairs (V0 s0 0.58 vs 0.56). Fix: a head-noun match in `_answers`, then regenerate the
  sampleable flags and text caches.
- V3 training probes (lang/text_sens, rule_std, examples.md) score swapped sentences with the sample's own
  tag. Training and the primary metric are unaffected; those probe numbers for V3 are not comparable.
