# Built-In Task Run Reference

## Contents

- [Common Rules](#common-rules)
- [nanoGPT](#nanogpt)
- [Small Molecule](#small-molecule)
- [Antibody](#antibody)

## Common Rules

Treat each task README, especially its **Minimal First Real Run** section, as
the source of truth for commands and artifacts. This reference identifies the
right configs and migration state; it intentionally does not duplicate volatile
real-run recipes.

Use identical `--set` overrides for dependency checks, runner dry-runs, and
execution. Runner `--dry-run` validates config resolution and any selected
experiment profile. A task-level `args.dry-run=true`, zero-iteration mode, or
similar option enters the task adapter and may write diagnostic artifacts.

The built-in tasks emit `LDMTaskSpec` and reuse shared `ldm_tts` components, but
they currently retain task-specific or compatibility execution loops. Do not
expect the complete `LDMEngine` artifact and resume contract unless the executed
task path actually constructs `LDMEngine`.

When a config selects `contract_profile`, do not override locked budget or
method arguments. Use a checked-in smoke profile, or follow a task README that
explicitly clears `contract_profile` for a diagnostic run. Such a run is not a
qualified execution of the named profile.

All task adapters accept OpenAI-compatible URL, model, and key settings. Keep
authenticated keys in environment variables, never tracked YAML or literal
command arguments.

## nanoGPT

Files:

```text
tasks/nanogpt/README.md
config/nanogpt/mock_best_of_n.yaml
config/nanogpt/real_operation_tool_best_of_n.yaml
config/nanogpt/real_operation_tool_fixed_best_of_n.yaml
```

Model variables: `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY`. The
historical `TTS_LLM_URL`, `TTS_LLM_MODEL`, and `TTS_LLM_API_KEY` aliases remain
accepted.

The real config selects a profile that locks `method`, `iterations`, and
`warmup`. Follow `tasks/nanogpt/README.md` when making a zero-iteration or tiny
run; its diagnostic recipe explicitly clears `contract_profile` before changing
those values. Real evaluation requires prepared data/tokenizer artifacts and
the task's training dependency group.

The current nanoGPT workflow is a compatibility runtime built around its
task-specific search engine. Inspect its run directory and
`model_based_summary.json`/`summary.json` rather than assuming all shared-engine
artifacts exist.

## Small Molecule

Files:

```text
tasks/small_molecule/README.md
config/small_molecule/mock_m1_stratified_oversample.yaml
config/small_molecule/real_m1_seed_analog.yaml
```

Model variables: `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY`. Evaluator
variables include `VINA_BIN` and `G12D`; ReaSyn paths are needed only by methods
that generate analogues.

The real profile locks `budget`, `batch-size`, and `acq`. Follow
`tasks/small_molecule/README.md` for contract and tiny runs; it explicitly clears
`contract_profile` before reducing the budget. `--no-optional` may omit ReaSyn
checks only when the selected direct method cannot call ReaSyn.

The current loop uses the shared compatibility `run_budgeted_search` API and
task-specific trajectory files. Inspect the task README and run summary for its
actual resume and artifact contract.

## Antibody

Files:

```text
tasks/antibody/README.md
tasks/antibody/resources/default_config.yaml
config/antibody/mock_ei.yaml
config/antibody/real_cpu_smoke.yaml
config/antibody/real_lcb.yaml
```

Model variables: `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY`. Real runs
also require an Absolut installation configured through `ABSOLUT_PATH` or the
selected task config.

Use `real_cpu_smoke.yaml` for the first real proposal and evaluation. It carries
the matching `real_cpu_smoke` contract profile and already fixes the one-run
budget, initialization count, parallel budget, and CPU device. Do not recreate
that smoke run by overriding `real_lcb.yaml`; reserve `real_lcb.yaml` for the
larger unprofiled run described by the task README.

The current antibody workflow is task-specific. Inspect its resolved run
directory, decision trajectory, results, and summary as documented by the task;
do not infer `LDMEngine` lifecycle semantics from its emitted `LDMTaskSpec`.
