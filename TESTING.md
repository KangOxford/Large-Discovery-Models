# Testing and Coverage

Validate task registration and conventional adapter layouts before running the
separate test lanes:

```bash
python scripts/validate_tasks.py
```

The repository has separate test lanes because the task packages use different
Python and scientific dependency stacks.

Coverage commands use `pytest-cov`. It is declared by the antibody and
small-molecule development groups; install it in the root test environment if
it is not already available.

## Shared and NanoGPT Core

From the repository root:

```bash
python -m pytest
python -m pytest --cov --cov-config=.coveragerc --cov-report=term-missing
```

The coverage command measures branch coverage for the shared `ldm_tts`
package and the dependency-free NanoGPT search engine, search methods, mock
trainer, and single-search wrapper. `.coveragerc` enforces a minimum total of
80%. GPU training code, remote model clients, and domain adapters are verified
in their package environments instead of being counted as uncovered core code.

## Antibody

Use the antibody environment documented in `tasks/antibody/pyproject.toml`, then run:

```bash
cd tasks/antibody
python -m pytest
python -m pytest --cov --cov-config=.coveragerc --cov-report=term-missing
```

The acquisition search tests import PyTorch, which is a declared runtime
dependency of the antibody package. The coverage configuration measures all
of `bo/ldm` with branch coverage and enforces a minimum total of 80%.

## Small Molecule

Install the package's scientific dependencies and run its suite from the
repository root. Prefer a standalone CPython rather than an Anaconda base
interpreter; some Anaconda macOS installations crash in the native `readline`
module before pytest collection begins.

```bash
uv sync --project tasks/small_molecule --group dev
uv run --project tasks/small_molecule python -m pytest tasks/small_molecule/tests
cd tasks/small_molecule
uv run python -m pytest --cov --cov-config=.coveragerc --cov-report=term-missing
```

The small-molecule coverage configuration measures all of `strbo_v1` with
branch coverage and enforces a minimum total of 80%.

The real ReaSyn, Vina, model-service, and GPU integration tests remain skipped
unless their documented environment variables and external assets are
available. Unit tests mock those process and service boundaries.
