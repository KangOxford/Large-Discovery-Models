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

4. Replace every generated semantic placeholder in
   `tasks/<task_id>/ldm_task/procedure.py`. Implement the real candidate space,
   objectives, model response contract, acquisition rule, and mock control
   flow. Do not report completion while any `replace_me` or "Replace with"
   placeholder remains.
5. Keep domain code, prompts, scorers, model clients, and heavy dependencies
   inside `tasks/<task_id>/`. Use shared `ldm_tts` modules for runner contracts,
   acquisition scoring, task-space descriptions, response parsing, trajectory
   records, and generic search-loop behavior.
6. Add a task-local `dependencies.py` hook only when the task has meaningful
   model, binary, artifact, dataset, accelerator, or evaluator prerequisites.
   Declare it in `task.json`. Keep its module imports lightweight so it can
   diagnose missing optional packages. Omit the hook for dependency-free tasks.
7. Finish the mock config and tests first. Mock execution must avoid remote
   models, external evaluators, GPUs, large downloads, and secrets.
8. Add real configs only after mock execution passes. Put endpoint URLs, model
   names, and non-secret defaults in config; source credentials from environment
   variables. Document a staged first real run in the task README.
9. Run the required verification sequence from the repository root:

   ```bash
   python scripts/validate_tasks.py --task <task_id>
   uv run --project tasks/<task_id> python -m pytest tasks/<task_id>/tests
   python scripts/check_task_dependencies.py config/<task_id>/mock.yaml --no-optional
   python scripts/run_ldm_tts.py config/<task_id>/mock.yaml --dry-run
   python scripts/run_ldm_tts.py config/<task_id>/mock.yaml
   python -m pytest -q tests
   git diff --check
   ```

10. Scan for the task ID in shared dispatch conditionals. Registration must not
    require a new task-name branch in `ldm_tts.runner` or
    `ldm_tts.dependency_checks`.

## Interface Rules

- Register through `tasks/<task_id>/task.json`; do not edit a central task map.
- Keep the conventional module at
  `tasks.<task_id>.ldm_task.procedure` and define `main(argv)`.
- Define `parse_args` and `describe_ldm_task` for consistency and inspection.
- Keep `describe_ldm_task` faithful to runtime behavior; it is not decorative
  metadata.
- Exercise mock and real execution through the shared config runner, not a
  parallel task-specific launcher.
- Prefer `ldm_tts.acquisition` for Mean, EI, LCB, UCB, and EHVI scoring. Add
  task-local acquisition behavior only when the domain algorithm requires more
  than posterior scoring, and document that distinction.
- Never put API keys, tokens, credentials, downloaded models, run outputs, or
  virtual environments into tracked files.

## Existing Task Repair

When making an existing task conform, run `python scripts/validate_tasks.py`
first. Fix missing manifest fields, package markers, procedure functions,
tests, configs, and dependency-hook references in place. Preserve the task's
stable ID and user-facing config paths unless the user explicitly requests a
breaking migration.
