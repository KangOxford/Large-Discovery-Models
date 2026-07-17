# LDM-TTS

This repository is organized around one shared Large Discovery Model
test-time search implementation and three task workflows. The task folders no
longer carry separate task-local script trees; each task exposes only the workflow
module needed by the common runner.

| Path | Purpose |
| --- | --- |
| `ldm_tts/` | Shared LDM-TTS primitives, scoring helpers, trajectory logs, and the config runner. |
| `config/` | Experiment configs and suites for all tasks and algorithms. |
| `scripts/run_ldm_tts.py` | Thin CLI shim for `ldm_tts.runner`. |
| `nanogpt/ldm_task/` | nanoGPT task workflow, train targets, operation schemas, and search strategies. |
| `small_molecule/ldm_task/` | Small-molecule workflow adapter for the acquisition-tilted LDM loop. |
| `antibody/ldm_task/` | AntBO-style workflow adapter for CDRH3 optimization. |
| `tests/` | Repository-level tests for shared LDM-TTS behavior. |

Task-local README files and legacy experiment notes have intentionally been
removed. This root README is the canonical setup and run guide.

## Architecture

The repository now has three layers:

1. `ldm_tts.runner` loads experiment configs, resolves the task, builds CLI
   arguments, and calls the configured task workflow module in-process.
2. Each task implements one procedure module at `<task>/ldm_task/procedure.py`.
   That module owns task-specific setup, proposal generation, scoring, resume
   behavior, and environment integration.
3. `config/` selects the task, algorithm label, mode, and task arguments. The
   same launcher can dry-run or execute any supported task.

Default task modules:

| Task | Workflow module | Default working directory |
| --- | --- | --- |
| `nanogpt` | `nanogpt.ldm_task.procedure` | `nanogpt/` |
| `small_molecule` | `small_molecule.ldm_task.procedure` | `small_molecule/` |
| `antibody` | `antibody.ldm_task.procedure` | `antibody/` |

Generated run outputs should live under each task's `ldm_runs/` directory.

## Housekeeping Policy

Track only the LDM-TTS runtime, minimal task workflow modules, small
configuration files, focused tests, and required model artifacts. Generated
runs, caches, scratch files, notebooks, plots, PDFs, and historical experiment
scripts stay out of git.

Ignored generated paths include:

- `**/ldm_runs/`
- `**/ldm_docs/`
- `**/progress.png`
- `antibody/cache/`, `antibody/outputs/`, and `antibody/temp/`
- `small_molecule/output/`
- Python caches, local virtual environments, and `.env` files

## Config-Driven Experiments

List available configs:

```bash
python scripts/run_ldm_tts.py --list
```

Print the execution plan for one experiment without running it:

```bash
python scripts/run_ldm_tts.py config/small_molecule/mock_m1_stratified_oversample.yaml --dry-run
```

Run one experiment:

```bash
python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml
```

Run one smoke experiment for each task:

```bash
python scripts/run_ldm_tts.py config/suites/mock_all.yaml
```

Override config values from the command line with dotted paths:

```bash
python scripts/run_ldm_tts.py config/nanogpt/mock_best_of_n.yaml \
  --set args.iterations=5 \
  --set args.run-name=nanogpt_mock_iter5
```

Config fields:

| Field | Meaning |
| --- | --- |
| `task` | One of `nanogpt`, `small_molecule`, or `antibody`. |
| `algorithm` | Human-readable algorithm label for bookkeeping. |
| `mode` | Usually `mock` or `real`. |
| `args` | CLI options passed to the task workflow, without the leading `--`. |
| `env` | Optional environment variables for that run. |
| `runner` | Optional overrides for `cwd` or `module`. Most configs should not need this. |

Use `null` for an optional CLI argument when you want the runner to omit it and
let the task workflow use its default. Put deployment paths and credentials in
`env:` when legacy helpers also read environment variables. If a task exposes
the same setting as a CLI flag, keep the matching `args:` entry explicit so the
dry-run plan shows the complete experiment command.

String values can use runner placeholders such as `{repo_root}` and
`{task_dir}`. Values in `args:` can also reference config environment variables
with shell-style syntax, for example `vina-bin: ${VINA_BIN}` or
`nn-model-path: ${G12D}`.

Relative values that start with a repository directory such as
`small_molecule/`, `nanogpt/`, `antibody/`, `config/`, `ldm_tts/`, or
`scripts/` are expanded from the repository root before the task runs.

Suite configs contain an `experiments` list of config paths and run them
sequentially. Paths inside `args` are resolved by the task workflow from that
task's working directory.

## nanoGPT Task

Install the nanoGPT task environment:

```bash
cd nanogpt
uv sync
```

Optional full-training setup downloads or prepares data for `train.py`:

```bash
uv run prepare.py
```

Fast local smoke run through the shared launcher:

```bash
python scripts/run_ldm_tts.py config/nanogpt/mock_best_of_n.yaml
```

The mock config targets:

- `nanogpt/ldm_task/mock_train.py`
- `nanogpt/ldm_task/operation_schema_mock_train.json`
- output under `nanogpt/ldm_runs/`

Real code-search runs usually target `nanogpt/ldm_task/real_train.py` or
`nanogpt/train.py`, use `nanogpt/ldm_task/operation_schema_real_train.json`,
and evaluate candidates with:

```bash
--eval-command "uv run python {train_path}"
```

For OpenAI-compatible LLM endpoints, configure `--llm-url`,
`--llm-model-name`, and optionally `--api-key`.

## Small-Molecule Task

Install the molecule task environment:

```bash
uv sync --project small_molecule
```

Fast mock run without external services:

```bash
uv run --project small_molecule python scripts/run_ldm_tts.py config/small_molecule/mock_m1_stratified_oversample.yaml
```

Real runs need:

- `LLM_BASE_URL` and `LLM_API_KEY` in config `env:`
- matching `llm-url` and `api-key` entries in config `args:` for the current
  small-molecule workflow CLI
- AutoDock Vina in `PATH` or `--vina-bin /path/to/vina`
- Meeko's receptor-preparation dependencies in the small-molecule uv
  environment. In particular, `gemmi` is required by recent Meeko releases.
- ReaSyn checkout via `REASYN_HOME`, `REASYN_REPO`, or `--reasyn-repo`
- ReaSyn Python dependencies either in the small-molecule uv environment or in
  a dedicated ReaSyn interpreter selected by `--reasyn-python`,
  `REASYN_PYTHON`, `REASYN_BIN`, or `<REASYN_REPO>/.venv/bin/python`. A venv
  `bin/` directory is also accepted and resolved to `bin/python`.
- ReaSyn AR and EB checkpoints, by default under
  `data/trained_model/nv-reasyn-ar-166m-v2.ckpt` and
  `data/trained_model/nv-reasyn-eb-174m-v2.ckpt`
- The NN activity model artifact at
  `activity_modeling/best_g12d_model.joblib`

The real starter config is:

```bash
uv run --project small_molecule python scripts/run_ldm_tts.py config/small_molecule/real_m1_seed_analog.yaml
```

Deployment-specific paths for Vina, the G12D activity model, ReaSyn, and the
trajectory root live in `config/small_molecule/real_m1_seed_analog.yaml`. The
config uses `env:` for compatibility with legacy helpers and explicit `args:`
entries such as `llm-url`, `api-key`, `vina-bin`, `nn-model-path`,
`reasyn-repo`, and `trajectory-dir` for the task workflow itself. Local
OpenAI-compatible servers often accept `EMPTY` as the API key placeholder; use
the real key if your endpoint enforces authentication.

Vina uses a cache under `<trajectory-dir>/vina_cache` by default. If you are
moving from an older run that already has a prepared receptor and docking cache,
set `args.vina-cache-dir` to that existing cache directory instead of moving the
whole trajectory back under `TTS/`. For example, a repository-root-relative path
such as `small_molecule/TTS/runs/<old-run>/vina_cache` is accepted. When a round
has candidates but cannot select any, inspect `selection_results.failed_evaluations`
in `rounds.jsonl`; it records per-objective scoring diagnostics from Vina and
the NN scorer.

The small-molecule workflow runs from the `small_molecule/` directory. A
relative `nn-model-path: activity_modeling/best_g12d_model.joblib` is valid, but
`nn-model-path: small_molecule/activity_modeling/best_g12d_model.joblib` is
clearer because the runner expands it from the repository root. Prefer
`nn-model-path: ${G12D}` with
`G12D: small_molecule/activity_modeling/best_g12d_model.joblib` in `env:`, or
use an absolute path.

`uv run --project small_molecule ...` uses `small_molecule/pyproject.toml` and
the uv-managed environment under `small_molecule/.venv`. ReaSyn generation uses
that same interpreter only when no separate ReaSyn interpreter is configured or
auto-detected. The selection order is:

1. `args.reasyn-python`
2. `env.REASYN_PYTHON`
3. `env.REASYN_BIN`
4. `<REASYN_REPO>/.venv/bin/python`
5. the current `small_molecule/.venv` interpreter

If ReaSyn raises an import error such as `No module named 'omegaconf'`, install
or sync that dependency in the interpreter selected by the list above. The
small-molecule uv config includes `omegaconf`, but a dedicated ReaSyn venv must
also contain ReaSyn's own dependencies.

Use the actual Python executable when possible:

```yaml
env:
  REASYN_PYTHON: /mnt/data0/dock-project/ReaSyn/.venv/bin/python
```

Outputs are written under `small_molecule/ldm_runs/` unless the config
overrides `trajectory-dir`.

Plot the Pareto hypervolume curve for a completed small-molecule run:

```bash
uv run --project small_molecule python small_molecule/plots/plot_pareto_hv.py small_molecule/ldm_runs/case2_mock_m1
```

The plotter writes `pareto_hv.png`, `pareto_hv.pdf`,
`pareto_hv_hypervolume.csv`, and `pareto_hv_summary.csv` under the run's
`plots/` directory. Pass an absolute run directory if the outputs live outside
this checkout.

## Antibody Task

Install the antibody task environment:

```bash
uv sync --project antibody
```

Fast mock run without Absolut or an LLM endpoint:

```bash
uv run --project antibody python scripts/run_ldm_tts.py config/antibody/mock_ei.yaml
```

Real runs need:

- Absolut installed and configured in `antibody/bo/config.yaml` under
  `bbox.path`
- `LLM_BASE_URL` and `LLM_API_KEY`, or the equivalent CLI flags
- An antigen via `--antigen` or an antigen list via `--antigens-file`

The real starter config is:

```bash
uv run --project antibody python scripts/run_ldm_tts.py config/antibody/real_lcb.yaml
```

Outputs are written under `antibody/ldm_runs/` unless the config overrides
`out-dir`.

`antibody/environment.yaml` is retained as the legacy conda environment for
the original pinned CUDA/PyTorch/DGL AntBO stack. Prefer `uv` for the unified
LDM-TTS runner; fall back to conda only if you need to reproduce that exact
legacy environment or your platform cannot resolve the GP dependencies through
`uv`.

`antibody/cache/init_dataset/` and `antibody/cache/init_dataset.zip` are legacy
AntBO custom-init data. They can exist locally if you need that old path, but
they are not tracked by this LDM-TTS-focused repository.

## Quick Checks

Run the shared core test from the repository root:

```bash
python -m pytest tests/test_ldm_tts_core.py
```

Run focused task checks when changing task code:

```bash
uv run --project small_molecule python -m pytest tests/test_tilted_loop.py tests/test_tilted_methods_m1.py
```

```bash
uv run --project antibody python -m pytest tests/bo/ldm
```
