# A VLM as the failure function (2026-10-26)

**Question.** The filter learns V(s) ≈ the lowest h ahead, where h is the failure function (1 safe, 0 at the edge,
−1 harmed). So far h came from the simulator (distances, tilts, bead counts). When there is no simulator h, can a
vision-language model (VLM) looking at the scene give an h that is good enough to train the filter?

## Decisions (Yuxuan, 2026-10-26)

| what | decision |
|---|---|
| VLMs | the two in `VLM/`: Qwen2.5-VL-32B-Instruct (`qwen25vl32b`) and RoboBrain2.0-32B (`robobrain2_32b`), one run each |
| robot, γ | ur5-wsg only, γ = 0.95 |
| tasks | T1 `selective_avoid`, T5 `tray_transport`, T6 `clean_pour` (three-bowl design) |
| VLM image | the agent view, the observer camera of the episode videos (the three encoder cameras are sometimes covered by the arm) |
| prompts | Yuxuan's for T1 and T5, one in the same style for T6 (`main/lang/vlm_prompts.yaml`) |
| T1 rules | only "don't touch X" (avoid) is named and trained: the T1 prompt asks about touch, bump, move and tilt together, so it has no separate answer for "don't move X" (stay) |
| what the VLM sees | two images per question: the episode's first frame and the current one, asked every 5 ticks (0.2 s) |
| where it runs | its own GPUs, not the training A40 (48 GB; a 32B model in bf16 needs ~66 GB): Yuxuan said A100; every Delta A100 has 40 GB, so Qwen runs on one H200 and RoboBrain on two A100s, as 1-h interactive jobs (batch A100 jobs were about a week out on 10-26) |
| rounds | bigger, so the runs wait for labels less often: 25 rounds × 16 episodes × 8192 steps (the same 400 episodes and 204,800 steps as the 10-23 runs) |
| reference | a third run with the simulator's h on the same data, tasks and rules: any difference comes from h alone |
| model | otherwise the 10-23 vanilla run's: HoloBrain-GD encoder, softmin critic, MLP actor |

## How an answer becomes h

Symbols: t = tick (25 per second) in the episode; a labelled tick k is a multiple of 5; s_k ∈ {−1, 0, 1} is the
VLM's score at tick k; h(t) is the h a training sample at tick t gets.

1. During collection the agent-view frame is saved at ticks 0, 5, 10, ... (`agent_view/<episode>/t0010.jpg`).
2. For each rule with a prompt and each saved tick k, the VLM gets [frame 0, frame k] and the rule's prompt with the
   item's name filled in ("the tall red block", "the blue bowl", "the wine glass"), and answers JSON:
   `{"score": -1, "explanation": "The arm bumps the green box."}`.
3. h(t) = s_k of the nearest labelled tick k (ties go to the earlier one; the last few ticks take the last score).
   The scale needs no conversion: −1 = harmed, 0 = uncertain (on the edge), 1 = safe, the same scale as the
   simulator's h.
4. fails_soon (used to balance harmful and benign rules in a batch) is recomputed from this h as before: some
   h < 0 within the next 25 ticks.

Example: answers at ticks 0, 5, 10, 15 = 1, 1, −1, −1. Ticks 0-2 get 1 (nearest is 0), 3-7 get 1 (nearest 5),
8-12 get −1, 13 onwards −1. The HJ target y = (1 − γ) h + γ min(h, V(next)) then pulls V below 0 from about tick 8 back.

An answer without a readable score is asked again once with sampling; if it still has none, it counts as 0.

## Data and files

- Warmup: the 90 T1 / T5 / T6 warmup episodes of `data/lang_1023` replayed with the same seeds (same layouts,
  re-simulated) and frames: `data/lang_vlm_1026/warmup/shard_g0..14` (chain stage `vlm1026_data`,
  `runs/lang_vlm_1026_data/`). Validation stays `data/lang_1023/val`, scored with the simulator's h.
- Labels: `<root>/vlm_labels/<vlm>/<episode>.json`, one file per episode (`vlm_prompts.py` docstring), written by the
  labeler. They record the prompt hash; labels made with other prompts are refused. The simulator's h stays in the
  item sidecar, untouched, for measuring.
- A run's own episodes: the ring gets frames too (`agent_view_every = 5`); the run's labeler labels each episode
  as it ends, and before each training phase the run waits for the last ones (`hsource.wait_labels`).

## Code (all in `main/lang/`, no CrossSafe or `envs/` file changed)

| file | role |
|---|---|
| `vlm_prompts.yaml`, `vlm_prompts.py` | prompts, the item phrase, the answer parser, label-file paths |
| `agent_view.py` | saves the frames during collection (`LangRolloutController._agent_view_tick`) |
| `vlm_label.py` | the labeler (`serve`, `agree`); interactive chains (`runs/lang/rounds_vlm_1026/start_labelers.sh`, `scripts/lang/vlm/labeler_watch.sh`) or a batch job (`scripts/lang/vlm/label.sbatch`) run it |
| `hsource.py` | `HSourceSampler`: swaps the VLM's h into the training sampler, restricts the rules, waits for labels |
| `config.py` | `h_source` ("sim" or "vlm:<name>"), `rule_consequences` ("all" or "vlm_prompts"), `agent_view_every` |

## What is measured

- How often the VLM agrees with the simulator: `python -m main.lang.vlm_label agree --vlm <name> --roots ...`
  (per task: same sign of h, the share each calls harmed, recall of the simulator's harmed ticks), and every round
  in the run's log (`vlm/<task>/agree`, `recall`, ...).
- The filter itself, with the simulator's h, as for the other runs: closed-loop success, collision / rule-broken
  rate and safe success, VLM runs against the simulator-h reference.

## Jobs

`runs/lang/rounds_vlm_1026/presubmit.sh` (training jobs on A40) and `start_labelers.sh` (the labelers); the training jobs wait for
`data/lang_vlm_1026/warmup/LABELS_<vlm>` (written by the labeler once every warmup episode is labelled), the
reference for `data/lang_vlm_1026/READY`.

A labeler runs only while there is work: the run touches `ring/vlm_labels/<vlm>/WANTED` while it collects and
waits, the labeler stops after 5 idle minutes without a fresh marker, and `labeler_watch.sh` (login node) holds the
chain stage while nothing is pending, so the GPUs are not held while the run trains (about 2/3 of each round).

## Labeler comparison (2026-10-27)

Five VLMs labelled the same 90 warmup episodes (30 per task, the careless nominal), with the same prompts and frames;
scored against the simulator's h by `scripts/lang/vlm/compare_vlms.py` (results `runs/lang/vlm_1026/compare/compare_5.json`).
Per (episode, rule) pair: **flagged** = the VLM calls it harmed at some tick, of the pairs the simulator calls harmed;
**false alarms** = the same, of the pairs the simulator never calls harmed; **flagged right** = of the pairs the VLM
flags, the share the simulator calls harmed.

| VLM (commit) | T1 touch: flagged / false alarms (flagged right) | T5 tip | T6 beads in bowl | s / question | GPU memory |
|---|---|---|---|---|---|
| Qwen2.5-VL-32B (7cfb30d7) | 83% / 32% (0.74) | 78% / 0% (1.00) | 88% / 18% (0.79) | 0.55-0.70 (2 x A100) | 66 GB |
| RoboBrain2.0-32B (892c7248) | 67% / 27% (0.74) | 91% / 14% (0.81) | 92% / 32% (0.69) | 0.49-0.78 (2 x A100) | 66 GB |
| RoboBrain2.5-8B-NV (7ae187cc) | 76% / 19% (0.82) | 61% / 27% (0.58) | 96% / 94% (0.44) | 0.19 (1 x A100) | 17.5 GB |
| Qwen3-VL-8B (0c351dd0) | 71% / 27% (0.75) | 0% / 0% | 100% / 100% (0.43) | 0.19 | 17.5 GB |
| Qwen2.5-VL-7B (cc594898) | 2% / 0% | 0% / 0% | 0% / 6% | 0.19 | 16 GB |

- Only the two 32B models are usable on all three tasks. Qwen-32B is the more precise (no T5 false alarm), RoboBrain-32B
  catches more (T5 91%, T6 92%).
- RoboBrain2.5-8B is the best T1 labeler, but in T6 it reports beads in the named bowl as soon as the pour starts
  (median first "failure" at tick 195): it does not tell the bowls apart.
- Qwen3-VL-8B never sees a tipped block and calls every bowl harmed; Qwen2.5-VL-7B calls nearly everything safe.
- Not tested yet: 4-bit versions of the 32B models on one A40 (the two-A40 job plan).


## 8-bit labelers and the one VLM run (2026-10-28)

Score = (detected + (1 - false alarm)) / 2 per task, averaged over T1, T5, T6 (90 warmup episodes):

| labeler | T1 | T5 | T6 | average |
|---|---|---|---|---|
| Qwen2.5-VL-32B, bf16 | 0.755 | 0.891 | 0.854 | 0.833 |
| RoboBrain2.0-32B, bf16 | 0.698 | 0.889 | 0.800 | 0.796 |
| RoboBrain2.0-32B, vLLM's own fp8 (two A100s) | 0.712 | 0.902 | 0.781 | 0.798 |
| RoboBrain2.0-32B, 4-bit NF4 | 0.534 | 0.769 | 0.775 | 0.693 |
| RoboBrain2.5-8B / Qwen3-VL-8B / Qwen2.5-VL-7B | | | | 0.655 / 0.574 / 0.494 |

- Yuxuan's decisions (2026-10-28): one VLM run only, the 8-bit Qwen if it scores higher than the 8-bit RoboBrain,
  else the RoboBrain (by the average score; within 0.01 is a tie and goes to RoboBrain, which answers "uncertain"
  less often); the HoloBrain GD model, gamma 0.95, the 10-26 settings; one job with two A40s, one GPU training and
  one labelling.
- vLLM's own fp8 loads the 64 GB bf16 model before converting it, so it cannot load on one 48 GB A40.
  `scripts/lang/vlm/to_fp8.py` saves the 8-bit copy once (llm-compressor's FP8_DYNAMIC format: per-channel scales,
  the vision tower kept bf16) on `/work/hdd/bifq/VLM_fp8`; entries `qwen25vl32b_fp8saved`, `robobrain2_32b_fp8saved`.
  Both are scored on one A40 (the demo chain's next job, `runs/lang/vlm_1026/compare/fp8saved_a40.sh`), which is also the
  check that they fit.
- `runs/lang/rounds_vlm8_1028/decide_and_submit.sh` (a login-node watcher) scores both when their labels are done,
  writes `decision.json` and submits `presubmit.sh <winner>`: run `runs/lang/rounds_ur5_vlm_<winner>_g095_1028`, one
  2-A40 job (`train_and_label.sbatch`), code = worktree `LanguageConditionHJ_rvlm8` (branch `lang-vlm8-1028`: the
  sim-h reference run's 774d5e6 plus the labeler's 8-bit entries), so only h differs from the reference run.
  Nothing is submitted if the winner scores below 0.75 (a broken copy).
- Result (2026-10-29, the saved 8-bit copies, 90 warmup episodes; one-A40 check: both load on one A40, Qwen labelled
  14 episodes there at ~0.8 s per question, the rest on two A100s): Qwen 8-bit 0.851 (T1 0.777, T5 0.921, T6 0.854;
  T1 detected 90 % / false alarm 35 %, T5 87 / 3, T6 88 / 18, "uncertain" 6 % / 0 % / 35 %), RoboBrain 8-bit 0.764
  (T1 0.686, T5 0.835, T6 0.770; T5 false alarms 24 %, T6 38 %). Winner Qwen (the 8-bit copy also scores above the
  bf16 Qwen, 0.833); `rounds_ur5_vlm_qwen25vl32b_fp8saved_g095_1028` submitted as job 23319949 (two A40s).
