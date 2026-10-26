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
| where it runs | one A100 per VLM (a 32B model needs ~66 GB; the training A40 has 48 GB) |
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
| `vlm_label.py` | the labeler (`serve`, `agree`); `scripts/lang/vlm/label.sbatch` runs it on an A100 |
| `hsource.py` | `HSourceSampler`: swaps the VLM's h into the training sampler, restricts the rules, waits for labels |
| `config.py` | `h_source` ("sim" or "vlm:<name>"), `rule_consequences` ("all" or "vlm_prompts"), `agent_view_every` |

## What is measured

- How often the VLM agrees with the simulator: `python -m main.lang.vlm_label agree --vlm <name> --roots ...`
  (per task: same sign of h, the share each calls harmed, recall of the simulator's harmed ticks), and every round
  in the run's log (`vlm/<task>/agree`, `recall`, ...).
- The filter itself, with the simulator's h, as for the other runs: closed-loop success, collision / rule-broken
  rate and safe success, VLM runs against the simulator-h reference.

## Jobs

`runs/lang/rounds_vlm_1026/presubmit.sh` (training jobs on A40, labelers on A100); the training jobs wait for
`data/lang_vlm_1026/warmup/LABELS_<vlm>` (written by the labeler once every warmup episode is labelled), the
reference for `data/lang_vlm_1026/READY`.

Cost note: each labeler holds its A100 for the whole run (about a week), idle while its run trains. A labeler that
runs only while its run collects would need the run to request it; not done.
