# Testing And Coverage

Validate task registration and conventional adapter layouts before running the
separate test lanes:

```bash
uv sync --locked --group dev
uv run --locked python scripts/validate_tasks.py
```

The repository has separate test lanes because the task packages use different
Python and scientific dependency stacks.

Coverage commands use `pytest-cov`. It is declared by the root and each
task-specific development group.

## Shared and NanoGPT Core

From the repository root:

```bash
uv run --locked python -m pytest
uv run --locked python -m pytest tests tasks/nanogpt/tests \
  --cov --cov-config=.coveragerc --cov-report=term-missing
```

The coverage command measures branch coverage for the shared `ldm_tts`
package and the dependency-free NanoGPT search engine, search methods, mock
trainer, and single-search wrapper. `.coveragerc` enforces a minimum total of
80%. GPU training code, remote model clients, and domain adapters are verified
in their package environments instead of being counted as uncovered core code.

## Antibody

Use the locked antibody environment documented in
`tasks/antibody/pyproject.toml`, then run:

```bash
uv sync --locked --project tasks/antibody --group dev
cd tasks/antibody
uv run --locked --group dev python -m pytest
uv run --locked --group dev python -m pytest \
  --cov --cov-config=.coveragerc --cov-report=term-missing
```

The acquisition search tests import PyTorch, which is a declared runtime
dependency of the antibody package. The coverage configuration measures all
of `core/ldm` with branch coverage and enforces a minimum total of 80%.

## Small Molecule

Install the package's scientific dependencies and run its suite from the
repository root. Prefer a standalone CPython rather than an Anaconda base
interpreter; some Anaconda macOS installations crash in the native `readline`
module before pytest collection begins.

```bash
uv sync --locked --project tasks/small_molecule --group dev
uv run --locked --project tasks/small_molecule \
  python -m pytest tasks/small_molecule/tests
cd tasks/small_molecule
uv run --locked --group dev python -m pytest \
  --cov --cov-config=.coveragerc --cov-report=term-missing
```

The small-molecule coverage configuration measures the unit-testable algorithm
core and service adapters with branch coverage and enforces a minimum total of
80%. It excludes the activity-model training script, docking implementation,
ReaSyn analogue client, dependency preflight, and top-level workflow. Those
operational boundaries are exercised by the mock workflow and opt-in real
integration lanes instead of being counted as uncovered unit-testable code.

The real ReaSyn, Vina, model-service, and GPU integration tests remain skipped
unless their documented environment variables and external assets are
available. Unit tests mock those process and service boundaries.
