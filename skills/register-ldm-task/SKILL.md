---
name: register-ldm-task
description: Scaffold, implement, register, scientifically qualify, and production-check an LDM domain task in this repository. Use when adding or repairing a task adapter, task manifest, experiment.json benchmark contract, metric roles, official evaluation budget, campaign profile, dependency checker, mock/real config, GP-guided search, endpoint preflight, durable budget/status reporting, or staged real-run qualification.
---

# Register And Qualify An LDM Task

Build a domain adapter through the manifest-driven task seam, then qualify its
scientific and operational contract before calling it production-ready.

Read [references/task-contract.md](references/task-contract.md) before editing.
Read [references/qualification.md](references/qualification.md) before adding a
real config or launching an external evaluator. Treat `tasks/README.md` as the
authoritative human-facing repository contract when present.

## Establish The Contract

Before scaffolding, discover or ask for:

- the candidate domain and its parser/validator boundary;
- each reservoir-expansion action and whether it emits candidates, configures a
  generator, edits a candidate, or updates the expansion schema;
- the surrogate representation, dimension policy, encoder, and version;
- the benchmark source URL, immutable commit, and task path;
- reported, optimized, and diagnostic metrics with directions;
- one expensive evaluation and its official per-candidate limits;
- search, LLM-attempt, expensive-evaluation, and baseline budgets;
- required datasets, artifacts, binaries, accelerators, and seed observations;
- resume expectations, comparison axis, and required run artifacts.

Do not infer an official budget from a smoke run. Record unknowns explicitly and
keep `experiment.json` at `qualification: draft` until primary-source evidence
and real evaluator checks support `qualified`.

## Implement Registration

1. Inspect `tasks/README.md`, `ldm_tts/registration/registry.py`, the closest task, and
   the domain benchmark.
2. Select a lowercase Python `task_id`. Confirm `tasks/<task_id>/` and
   `config/<task_id>/` do not already exist.
3. Run the non-overwriting scaffolder:

   ```bash
   python scripts/scaffold_task.py <task_id> --description "<one-line description>"
   ```

4. Replace every semantic placeholder. Keep `ldm_task/procedure.py` shallow;
   put candidate, prompt, reservoir-expansion, surrogate-adapter, and evaluator
   code in `core/`, versioned inputs in `resources/`, and outputs in ignored
   `runs/`.
5. Complete `experiment.json`. Keep registration identity in `task.json`; keep
   scientific provenance, metric roles, evaluator settings, limits, and named
   runner-enforced campaign profiles in `experiment.json`.
6. Implement the campaign through `ldm_tts.engine.LDMEngine`. Supply task-owned
   `ReservoirExpander`, `CandidateDomainAdapter`, and `CandidateEvaluator`
   adapters. Add `SurrogateEncoder` and `AcquisitionSelector` only for
   surrogate-guided methods. Use `CampaignRuntime` for budgets, events,
   checkpoints, status, and summaries; use `ProposalClient` for model transport.
   Reuse `ldm_tts.optimization.search`, `ldm_tts.optimization.gp`, and
   `ldm_tts.optimization.acquisition`
   behind those adapters before adding task-local infrastructure.
7. Define the fine-tuning collection boundary after response parsing and
   validation. Append canonical `ldm-2.0` IR through
   `DataCollectionSink.from_env`; keep provenance/outcomes out of model-visible
   state and never collect rejected attempts or incompatible fallback actions.
8. Add a lightweight `dependencies.py` hook only for meaningful external
   prerequisites. Never import optional heavy dependencies at module import.

## Qualify In Stages

Finish each stage before starting the next:

1. `registered`: manifest, layout, draft experiment contract, and imports.
2. `mock_verified`: deterministic mock run and collection test.
3. `contract_verified`: candidate parser plus CPU/GPU tensor, parameter, and
   evaluator assembly checks.
4. `seed_evaluated`: one official-budget seed evaluation with source commit,
   metrics, and artifacts recorded.
5. `ldm_tiny_verified`: endpoint preflight, one generated reservoir, one
   acquisition-selected candidate, and one real evaluation.
6. `campaign_qualified`: named contract profile, durable resume, budget/status
   files, comparison budget, and monitored detached launch.

Use the exact gates and expected artifacts in
[references/qualification.md](references/qualification.md). Registration is not
the same as campaign qualification; report both states explicitly.

## Required Verification

Run from the repository root:

```bash
python scripts/validate_tasks.py --task <task_id>
uv run --locked --project tasks/<task_id> python -m pytest tasks/<task_id>/tests
python scripts/check_task_dependencies.py config/<task_id>/mock.yaml --no-optional
python scripts/run_ldm_tts.py config/<task_id>/mock.yaml --dry-run
python scripts/run_ldm_tts.py config/<task_id>/mock.yaml
python -m pytest -q tests
git diff --check
```

After the seed and tiny LDM gates justify changing the contract to `qualified`,
also run `python scripts/validate_tasks.py --task <task_id> --require-qualified`.
The normal validator accepts honest draft scaffolds; the strict form is the
campaign-readiness gate.

Before a real launch, also verify the selected `contract_profile` appears in the
runner dry run, endpoint preflight succeeds, `status.json` and `budget.json` are
created, and the first selected candidate enters the intended evaluator rather
than a standalone benchmark agent.

## Interface Rules

- Register only through `task.json`; do not add a central task-name branch.
- Use the canonical terms in `docs/concepts.md`: candidate domain, reservoir,
  reservoir expansion, expansion schema, and surrogate representation.
- Keep `experiment.json` versioned, strict, secret-free, and source-pinned.
- Put task CLI options under config `args`, environment values under `env`, and
  select enforced production settings with top-level `contract_profile`.
- Define `main(argv)`, `parse_args`, and a runtime-faithful `describe_ldm_task`.
- Make the deterministic mock execute at least one complete `LDMEngine` round
  and assert that `events.jsonl`, `checkpoint.json`, and `summary.json` exist.
- Count LLM calls, valid search states, selected candidates, expensive attempts,
  successful evaluations, benchmark jobs, and outer iterations separately.
- Use expensive evaluations, not wall time or generated states, as the default
  fair-comparison x-axis unless the benchmark specifies otherwise.
- Snapshot the active experiment contract into every qualified run.
- Preflight real model endpoints before iteration 1. Pause resumably when the
  circuit opens; do not exhaust the candidate budget on identical timeouts.
- Keep credentials in environment variables or ignored protected files. Never
  write them to configs, logs, manifests, prompts, or command arguments.

## Existing Task Repair

Run validation first. Preserve the stable task ID and config paths. Add
`experiment.json` without changing schema-version-1 `task.json`, migrate generic
GP/budget/endpoint behavior to shared modules where practical, and retain
compatibility exports when callers already import task-local names.
