---
name: register-ldm-task
description: Scaffold, implement, register, and verify a new domain task under this repository's tasks/ directory. Use when adding a new LDM task, domain adapter, task manifest, task-specific dependency checker, mock/real experiment configs, or when making an existing task conform to the manifest-driven task contract.
---

# Register An LDM Task

Add a domain adapter through the repository's manifest-driven task seam. Keep
the runner interface small and domain implementation local to the task.

Read [references/task-contract.md](references/task-contract.md) before editing.
Treat `tasks/README.md` as the authoritative human-facing contract when it is
present in the repository.

## Workflow

1. Inspect `tasks/README.md`, `ldm_tts/task_registry.py`, the closest existing
   task, and the user's domain requirements.
2. Select a lowercase Python identifier for `task_id`. Confirm the directory
   and `config/<task_id>/` do not already exist.
3. Run the non-overwriting scaffolder:

   ```bash
   python scripts/scaffold_task.py <task_id> --description "<one-line description>"
   ```

4. Replace every generated semantic placeholder. Keep
   `tasks/<task_id>/ldm_task/procedure.py` as the stable adapter and implement
   candidate generation, objectives, model calls, acquisition, evaluation, and
   mock control flow under `tasks/<task_id>/core/`. Do not report completion
   while any `replace_me` or "Replace with" placeholder remains.
5. Keep importable domain code in `core/`, versioned inputs in `resources/`,
   auxiliary CLIs in `scripts/`, optional external environment specs in
   `environments/`, and generated artifacts in ignored `runs/`. Use shared
   `ldm_tts` modules for runner contracts, acquisition scoring, task-space
   descriptions, response parsing, trajectory records, and generic search-loop
   behavior.
6. Define the task's fine-tuning collection boundary. When the runtime produces
   validated model actions, create `DataCollectionSink.from_env` with a
   run-local `runs/.../ldm_data` default and append canonical `ldm-2.0` IR only
   after parsing/validation succeeds. Keep provenance and evaluator outcomes in
   `collection`, never in model-visible state. Do not collect rejected attempts,
   random fallbacks, or task actions whose inference contract differs from the
   training target. If the task cannot produce trainable actions yet, document
   that decision in its README and tests.
7. Add a task-local `dependencies.py` hook only when the task has meaningful
   model, binary, artifact, dataset, accelerator, or evaluator prerequisites.
   Declare it in `task.json`. Keep its module imports lightweight so it can
   diagnose missing optional packages. Omit the hook for dependency-free tasks.
8. Finish the mock config and tests first. Mock execution must avoid remote
   models, external evaluators, GPUs, large downloads, and secrets.
9. Add real configs only after mock execution passes. Put endpoint URLs, model
   names, and non-secret defaults in config; source credentials from environment
   variables. Document a staged first real run in the task README.
10. Run the required verification sequence from the repository root:

   ```bash
   python scripts/validate_tasks.py --task <task_id>
   uv run --locked --project tasks/<task_id> python -m pytest tasks/<task_id>/tests
   python scripts/check_task_dependencies.py config/<task_id>/mock.yaml --no-optional
   python scripts/run_ldm_tts.py config/<task_id>/mock.yaml --dry-run
   python scripts/run_ldm_tts.py config/<task_id>/mock.yaml
   python -m pytest -q tests
   git diff --check
   ```

11. Scan for the task ID in shared dispatch conditionals. Registration must not
    require a new task-name branch in `ldm_tts.runner` or
    `ldm_tts.dependency_checks`.

## Interface Rules

- Register through `tasks/<task_id>/task.json`; do not edit a central task map.
- Keep the conventional module at
  `tasks.<task_id>.ldm_task.procedure` and define `main(argv)`.
- Keep the conventional module shallow: it delegates into `core/` and does not
  contain the task implementation.
- Define `parse_args` and `describe_ldm_task` for consistency and inspection.
- Keep `describe_ldm_task` faithful to runtime behavior; it is not decorative
  metadata.
- Exercise mock and real execution through the shared config runner, not a
  parallel task-specific launcher.
- Prefer `ldm_tts.acquisition` for Mean, EI, LCB, UCB, and EHVI scoring. Add
  task-local acquisition behavior only when the domain algorithm requires more
  than posterior scoring, and document that distinction.
- Import collection APIs from `ldm_tts.data`. Instantiate the sink once per run
  or recorder, and use the run directory's `ldm_data/` child as the default.
- Build IR from the accepted parsed action, not the raw first response. Preserve
  stable run/task provenance so `finetune/prepare_dataset.py` can group related
  records and prevent train/evaluation leakage.
- Add a mock integration test with `LDM_DATA_COLLECTION_ENABLED=1` that validates
  the emitted IR and confirms private provenance/outcomes do not enter the
  rendered instruction.
- Never put API keys, tokens, credentials, downloaded models, run outputs, or
  virtual environments into tracked files.

## Existing Task Repair

When making an existing task conform, run `python scripts/validate_tasks.py`
first. Fix missing manifest fields, package markers, procedure functions,
tests, configs, and dependency-hook references in place. Preserve the task's
stable ID and user-facing config paths unless the user explicitly requests a
breaking migration.
