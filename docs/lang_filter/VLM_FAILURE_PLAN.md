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

