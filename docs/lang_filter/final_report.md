# Vanilla language-conditioned filter: overnight implementation report (2026-10-06/07)

Status per phase, what was built, every deviation from `VANILLA_PLAN.md` and why, and the decisions
left for the human. Detailed reports: `recon.md` (Phase 0), `collect_report_warmup.md` /
`collect_report_val.md` (Phase 1), `phase2.md` (model), `phase3.md` (sampler, trainer, sanity, offline
run), `phase4.md` (rounds), `phase5.md` (closed loop).

<!-- STATUS -->

## 1. What was built

| file | content |
|---|---|
| `main/lang/__init__.py` | `main.__path__` bootstrap for `main.item_envs` |
| `main/lang/config.py` | `LangConfig(FrozenConfig)` (Section 11 defaults; the four per-episode env overrides declared), canonical index tuples, the Section 4.2 seed / id helpers with bound asserts, bank hash |
| `main/lang/scene_roles.py` | `referent`, `in_way` (T1 placements + second-corridor rule, T3 elements, T5 tall block, T6 hit container + beads + unclear diagonal, T7 hit flanker), `designation`, `ref_rows`, `mask_region_rows` |
| `main/lang/collect.py` | the one episode loop (`run_episodes`, Section 7.1) for warmup, validation, recon and rounds; shard schedule (Section 7.2); warmup / validation / recon CLIs with episode-boundary resume; per-episode scene metadata |
| `main/lang/rollout.py` | `LangRolloutController` (status, s0 capture outside collect mode, timing, rule trace, Q_cf), `LangPolicyShim`, `set_instruction`, `wire_item_eval` |
| `main/lang/network.py` | the model of Section 5 (`phase2.md`) |
| `main/lang/text.py` | `TextStore`, the `bert_raw` / `sentence_generic` backends and their cache builder |
| `main/lang/data.py` | `LangSampler` (Section 4.4), `PrefetchSampler` |
| `main/lang/train.py` | `LangTrainer` (Section 6.1), offline mode, checkpoints, `--resume`, sanity checks (Section 6.4), CLI |
| `main/lang/rounds.py` | rounds mode (Sections 6.2, 7.3, 7.4) |
| `main/lang/metrics.py` | Section 8.1 scalars, Section 8.2 probes (attention recomputed from `GeoCrossAttention`'s own projections), `LangRoundMetrics` |
| `main/lang/video.py` | `LangPanelVideoWriter` (Section 8.4) |
| `main/lang/eval_offline.py` | offline language probe (Section 9.1) |
| `main/lang/eval_closed.py` | closed-loop grid (Section 9.2), sharded parallel children |
| `main/lang/report.py` | one table per research question |
| `main/lang/recon.py`, `phase4_checks.py`, `collect_report.py` | Phase 0 checks, Phase 4 simulator checks, the Phase 1 data report incl. the delay audit |
| `scripts/lang/` | `in_container.sh`; `chain.sh` / `chain2.sh` (chained 1-hour interactive jobs; A40 for simulation, A100 for torch-only stages); `gpu_stage.sbatch`, `par_gpu.sh`, `par_cmds.sh`; `stage.sbatch`, `long.sbatch`; `matrix.sh` (prints only); `run_rounds.sh`; `crosssafe_regression.py`; `profile_step.py` |
| `tests/lang/` | 71 CPU tests (seed scheme, scene roles on records of all 7 tasks, schedule, shard resume, model, sampler, trainer resume, metrics, offline probe, closed-loop aggregates, shim history) |

## 2. Deviations from the plan, and why

1. **Compute path.** The batch partition (`gpuA40x4`) was estimated to start in two weeks, so everything
   runs as chained 1-hour interactive jobs (one per partition at a time). Interactive jobs may take a
   whole 4-GPU node, so collection runs 5 shards per A40. Training-only stages run on the A100
   interactive partition, which has its own one-job limit. A100s have no ray-tracing cores, so all
   rendering stays on A40.
2. **Resumable shards.** A shard (14 episodes, ~70 min) does not fit one hour. Shards resume at episode
   boundaries: the interrupted episode's rows are rolled back on both rings (`discard_to`) and the slot
   is collected again with the same seed (same scene). The plan's rule "a crashed shard is deleted and
   collected again" exists to avoid id collisions and overwritten records. Neither can happen this way,
   because the ids and the records are the same scene's.
3. **No per-tick segmentation masks in warmup** (`--no-masks`): ~30% of the per-tick cost; `ref_mask`
   only comes from validation data (probes, offline probe), which keeps the masks.
4. **Check 4 tolerance**: read as relative 1e-3 (fp16 storage of features up to |150|; `recon.md`).
5. **Closed-loop nominal runs** use the env's standalone runner without rendering (same controller
   and physics, no model reads the cameras), ~5x faster than a rendered controller run.
6. **The CrossSafe 20-step regression** uses a recon StepBuffer (no CrossSafe dataset on Delta).
7. **`eval_closed`** shards each embodiment's work list and runs children in parallel; the full grid is
   ~1120 filtered episodes per checkpoint (~90 h of simulation), which one child per embodiment
   cannot finish.

<!-- DEVIATIONS_MORE -->

## 3. Decisions reserved for the human

Each item names the evidence and a recommendation; nothing below was decided for you.

1. **gamma (0.9 now).** Delay audit (170 nominal traces): the median time from the start of the risky
   motion to the first violation is 0.6 s (T7) to 3.4 s (T3), against the ~0.4 s (10-tick) horizon of
   gamma = 0.9. The T3 numbers are inflated: its risky segment is the whole push. T5 tipping (0.80 s)
   and T7 scraping (0.60-0.64 s) are the clean cases. Recommendation: try 0.95 (horizon ~20 ticks) as
   one extra offline run against 0.9; with 0.9 the critic can only warn ~7 ticks ahead.
2. **Warmup size (30 episodes per task and embodiment = 420).** Collection runs at ~1 episode per
   minute per A40 in 1-hour interactive slots, about 3 slots of 3-4 GPUs. The label balance in
   `collect_report_warmup.md` decides whether T4 / T7 (few harmful rows) need more.
3. **T1 near-path candidates.** Designated *unclear* by the plan (never named), but violated in 59-79% of
   nominal runs (78-100% on the two stacking hosts, which rarely have an on-path candidate). Promoting
   `near_path` to harm-designated would give T1's stacking hosts harm runs; keeping it leaves them
   benign-only.
4. **Text feature scale.** HoloBrain word features have RMS 9 (outliers to 150) and pass the identity
   `in_proj` unnormalized into keys / values (`phase2.md`). Training is stable so far; a fixed scale or an
   RMSNorm on `w'` would be a one-line change, to try only if probes show the text attention saturating.
5. **Batch size / precision.** Unchanged (B = 64, fp32). TF32 would cut the step 15% on H200, 29% on
   A100: small enough to leave as is.
6. **Evaluation scene counts.** The full grid (20 scenes, 2 axes) is ~90 h of simulation per checkpoint
   (~4.5 h on a 4-GPU node). Recommendation: 10 scenes per combo on `train` + `heldout_template` for the
   first comparison.
7. **Matrix runs to launch.** `bash scripts/lang/matrix.sh` prints all of them; nothing beyond `full` was
   launched. Each offline run is ~2.6 h on an H200 at 20k steps (~15 units).
8. **Env-side issues (envs/ is read-only for this work).** `place_container_plate` fails its grasp
   (`target_pose cannot be None`, RoboTwin `_base_task.py:1212`) in ~30% of episodes (recorded as
   `status: exception`, kept).

