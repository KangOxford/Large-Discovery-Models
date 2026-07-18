# LDM-TTS

LDM-TTS is a shared Large Discovery Model test-time search codebase. It uses
one config runner and one lightweight abstraction layer to run LLM-guided,
Bayesian-optimization-style search across three domains:

| Task | Optimizes | Task Guide and Real Run |
| --- | --- | --- |
| `nanogpt` | Training-code and hyperparameter operations for nanoGPT-style pretraining. | [nanogpt/README.md](nanogpt/README.md) |
| `small_molecule` | SMILES candidates for docking and activity objectives. | [small_molecule/README.md](small_molecule/README.md) |
| `antibody` | CDRH3 amino-acid sequences for antigen binding. | [antibody/README.md](antibody/README.md) |

The shared code keeps orchestration, config loading, task-space specs, response
parsing, trajectory metadata, and common tests in one place. Task adapters keep
domain-specific dependencies such as training data, Vina, ReaSyn, and Absolut
behind task boundaries.

## Quick Start

Start from the repository root:

```bash
cd /path/to/LDM_merge
```

Install the task environments you plan to use:

```bash
uv sync --project nanogpt
uv sync --project small_molecule
uv sync --project antibody
```

List configs and preview the mock suite:

```bash
python scripts/run_ldm_tts.py --list
python scripts/run_ldm_tts.py config/suites/mock_all.yaml --dry-run
```

Run fast mock experiments:

```bash
uv run --project nanogpt python scripts/run_ldm_tts.py config/nanogpt/mock_best_of_n.yaml
uv run --project small_molecule python scripts/run_ldm_tts.py config/small_molecule/mock_m1_stratified_oversample.yaml
uv run --project antibody python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml
```

Mock configs are the safest first check. They exercise the merged runner,
task-space specs, response parsing, and trajectory plumbing without requiring
real LLM endpoints or domain-specific external tools.

## Environment Setup

Real experiments usually need CUDA visibility and an OpenAI-compatible LLM
endpoint. Set these before editing or running real configs:

```bash
export CUDA_VISIBLE_DEVICES=0
export LLM_BASE_URL=http://127.0.0.1:52308/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL_NAME=Qwen3.5-9B
```

nanoGPT also accepts its historical LLM variable names:

```bash
export TTS_LLM_URL=$LLM_BASE_URL
export TTS_LLM_API_KEY=$LLM_API_KEY
export TTS_LLM_MODEL=$LLM_MODEL_NAME
```

Small-molecule real runs need additional task dependency paths:

```bash
export VINA_BIN=/path/to/vina
export G12D=small_molecule/activity_modeling/best_g12d_model.joblib
export REASYN_REPO=/path/to/ReaSyn
export REASYN_PYTHON=/path/to/ReaSyn/.venv/bin/python
```

Antibody real runs need `antibody/bo/config.yaml` updated with the local
Absolut path:

```yaml
bbox:
  tool: Absolut
  path: /path/to/Absolut
```

Task configs can set these values directly under `env:` and mirror them into
`args:` with `${VAR}` references. Config values win when they are passed as
explicit task CLI arguments.

## Dependency Checks

Before running task-relevant real experiments, run the dependency checker on
the exact config you plan to use:

```bash
python scripts/check_task_dependencies.py config/nanogpt/real_operation_tool_best_of_n.yaml
python scripts/check_task_dependencies.py config/small_molecule/real_m1_seed_analog.yaml
python scripts/check_task_dependencies.py config/antibody/real_lcb.yaml
```

The checker reads the same YAML configs as the runner and reports `OK`,
`WARN`, `FAIL`, or `SKIP` for each dependency. It checks lightweight things
only: configured LLM settings, CUDA visibility, file paths, Vina executability,
ReaSyn checkout/checkpoints, nanoGPT data artifacts, antigen inputs, and
Absolut path.

If a config mentions optional dependencies that the selected method will not
use, such as ReaSyn paths in a direct-only small-molecule run, add
`--no-optional`.

Use overrides exactly as with the runner:

```bash
python scripts/check_task_dependencies.py config/small_molecule/real_m1_seed_analog.yaml \
  --set args.vina-bin=/path/to/vina \
  --set args.llm-url=$LLM_BASE_URL
```

The small-molecule real workflow also performs an early runtime preflight for
Vina and the activity model before starting a search.

## Config-Driven Runs

Experiments are YAML files under `config/`. A config selects the task,
algorithm label, mode, environment variables, and task CLI arguments.

Minimal shape:

```yaml
name: small_molecule_mock_m1
task: small_molecule
algorithm: m1_stratified_direct_llm_oversample_sir
mode: mock
env:
  G12D: small_molecule/activity_modeling/best_g12d_model.joblib
args:
  mock: true
  budget: 8
  batch-size: 1
  nn-model-path: ${G12D}
```

Important fields:

| Field | Meaning |
| --- | --- |
| `name` | Human-readable run name. |
| `task` | One of `nanogpt`, `small_molecule`, or `antibody`. |
| `algorithm` | Bookkeeping label for the run style. |
| `mode` | Usually `mock` or `real`. |
| `env` | Environment variables set for the run. |
| `args` | CLI options passed to the task workflow, without the leading `--`. |
| `runner` | Optional task module or working-directory override. |

Useful commands:

```bash
python scripts/run_ldm_tts.py config/small_molecule/mock_m1_stratified_oversample.yaml --dry-run
python scripts/run_ldm_tts.py config/suites/mock_all.yaml
```

Override config values with dotted paths:

```bash
python scripts/run_ldm_tts.py config/nanogpt/mock_best_of_n.yaml \
  --set args.iterations=5 \
  --set args.run-name=nanogpt_mock_iter5
```

Config values support:

- `null` to omit an optional CLI argument and let the task default apply
- runner placeholders such as `{repo_root}` and `{task_dir}`
- environment references in `args`, such as `vina-bin: ${VINA_BIN}`
- repository-root expansion for values starting with `small_molecule/`,
  `nanogpt/`, `antibody/`, `config/`, `ldm_tts/`, or `scripts/`

Suite configs contain an `experiments` list and run the listed configs
sequentially.

## Codebase Architecture

The codebase has three layers:

| Layer | Where | Responsibility |
| --- | --- | --- |
| Shared runner | `ldm_tts.runner`, `scripts/run_ldm_tts.py` | Load configs, build commands, run suites, and provide dry-runs. |
| Shared contracts | `ldm_tts/` | Describe task spaces, parse LLM JSON, define operation-space helpers, serialize trace shapes, and provide common scoring utilities. |
| Task adapters | `<task>/ldm_task/procedure.py` | Own prompts, LLM calls, candidate generation, scoring, acquisition, resume behavior, and output writing. |

Key shared modules:

| Module | Purpose |
| --- | --- |
| `ldm_tts.spaces` | `LDMTaskSpec`, candidate spaces, objectives, response spaces, and acquisition specs. |
| `ldm_tts.response` | Shared LLM JSON extraction and validation helpers. |
| `ldm_tts.parameter_space` | nanoGPT operation-schema primitives and validators. |
| `ldm_tts.bo` | Lightweight BO records and protocols. |
| `ldm_tts.trace_schema` | Task-neutral candidate and round trace shapes. |
| `ldm_tts.trajectory` | Atomic JSON and JSONL trajectory writers. |
| `ldm_tts.dependency_checks` | Config-aware preflight checks for task dependencies. |

The shared package should remain dependency-light. Heavy domain dependencies
such as RDKit, torch, gpytorch, Vina, ReaSyn, and Absolut should stay inside
task packages or task setup instructions.

## Outputs And Logs

Common run artifacts:

| Artifact | Meaning |
| --- | --- |
| `summary.json` | Task-level run summary. |
| `model_based_summary.json` | nanoGPT model-based search summary. |
| `ldm_task_spec.json` | Serialized task-space contract for the run. |
| `config.json` | Trajectory config snapshot. |
| `rounds.jsonl` or task-specific JSONL logs | Per-round candidates, decisions, scores, and diagnostics. |
| `model_based_buffer.jsonl` | nanoGPT evaluated-state buffer for GP fitting and resume. |
| `vina_cache/` | Small-molecule docking and receptor-preparation cache. |

Generated runs, caches, scratch files, plots, notebooks, local virtual
environments, and `.env` files should stay out of git.

## Customization

Start from the closest YAML file under `config/`, then edit `env` and `args`.
Run both the dependency checker and runner dry-run before launching a real
experiment:

```bash
python scripts/check_task_dependencies.py config/small_molecule/real_m1_seed_analog.yaml
python scripts/run_ldm_tts.py config/small_molecule/real_m1_seed_analog.yaml --dry-run
```

To add a new task, provide:

1. A workflow module, usually `<task>/ldm_task/procedure.py`.
2. A `parse_args(...)` function or compatible CLI.
3. A `describe_ldm_task(...) -> LDMTaskSpec` helper.
4. Mock mode for fast local checks.
5. Configs under `config/<task>/`.
6. Focused tests under `tests/`.
7. Optional dependency checks in `ldm_tts.dependency_checks`.

See the task guides for domain-specific customization:

- [nanoGPT task guide](nanogpt/README.md)
- [Small-molecule task guide](small_molecule/README.md)
- [Antibody task guide](antibody/README.md)

