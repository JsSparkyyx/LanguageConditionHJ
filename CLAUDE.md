# LanguageConditionHJ: project instructions for Claude Code

@AGENTS.md

## Current direction: item-centric, language-conditioned safety filter

Design decisions and their reasons (read before changing the method, data, or environments):

@docs/lang_filter/DESIGN_CONTEXT.md

Plans and docs (open when working on that part, not imported):
- `docs/lang_filter/VANILLA_PLAN.md`: current implementation plan (vanilla text-conditioned filter).
- `envs/docs/item_envs/`: the item environments (implemented): `tasks_overview.md`, `relabeling.md`,
  `task_cards/`, `final_report.md`.

## Repo layout

- `main/`: CrossSafe (with the Phase-5 item hooks in `main/train/collect.py` and `rollout.py`).
- `envs/main/item_envs/`: the item environments, imported as `main.item_envs`. Code run from the repo
  root appends `<repo>/envs/main` to `main.__path__` first (`collect.is_item_task`, `main/lang/__init__.py`).
  Env tests run from `envs/`.
- `main/lang/`: the language-conditioned filter (new code goes here).
- GPU and simulator jobs run inside `/work/nvme/bifq/robot-sim_cu128.sif` on Delta
  (`scripts/lang/in_container.sh`).

## Working rules

- Do not edit CrossSafe files (`main/network`, `main/train`, `main/envs`) or anything under `envs/`
  without asking. Compose and subclass in `main/lang/`.
- No rule-type or task-type input to the model. The sentence is the only rule description.
- Keep CrossSafe semantics identical (Q(s, a_nom) < 0 engages the filter, 3-tick hold), with γ = 0.95
  (Yuxuan, 2026-10-08; runs keep the γ of their saved config).
- No external detector or segmenter in the deployed filter (offline labeling and ablations only).
- When explaining design choices, define every symbol and give a concrete example.