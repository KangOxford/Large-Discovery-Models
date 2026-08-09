# Registering LDM Tasks

Each directory under `tasks/` is a domain adapter behind the shared runner
interface. A task registers itself by adding a versioned `task.json` manifest;
no shared Python registry or dependency-check dispatch table needs editing.

## Standard Layout

```text
tasks/<task_id>/
├── task.json                 # registration manifest
├── README.md                 # domain setup and run tutorial
├── QUICKSTART.md             # validated clean-room workflow
├── pyproject.toml            # isolated task dependencies
├── __init__.py
├── ldm_task/
│   ├── __init__.py
│   ├── procedure.py          # stable shared-runner adapter
│   └── dependencies.py       # optional dependency-check adapter
├── core/                     # private task implementation
│   └── __init__.py
├── resources/                # versioned schemas, seeds, inputs, models
│   └── README.md
├── scripts/                  # optional maintenance/training CLIs
├── environments/             # optional Conda or external-tool specs
├── tests/                    # task-local tests
│   └── test_procedure.py
└── runs/                     # generated runtime artifacts; Git-ignored

config/<task_id>/
├── mock.yaml                 # local, service-free smoke run
└── real.yaml                 # real model and evaluator settings
```

`ldm_task` is the external seam. The shared runner and dependency checker know
only `procedure.main(argv)` and the optional manifest hook. Keep implementation
out of this package so the interface remains small and stable.

The remaining directories have one ownership rule each:

- `core/` contains importable search, surrogate, evaluator, and model-client code.
- `resources/` contains versioned non-generated inputs required at runtime.
- `scripts/` contains auxiliary CLIs that are not runner entrypoints.
- `environments/` contains optional environment specifications beyond `pyproject.toml`.
- `tests/` verifies the task interface and implementation.
- `runs/` contains all generated artifacts and must never be committed.

Do not create alternate task entrypoints or implementation packages beside
these directories. Add domain-specific subpackages inside `core/` and organize
resource types inside `resources/` instead.

## Scaffold A Task

From the repository root:

```bash
python scripts/scaffold_task.py protein_design \
  --description "Optimize protein candidates against structure objectives."
```

The command creates `tasks/protein_design/` and
`config/protein_design/mock.yaml`. It never overwrites an existing task. The
generated mock adapter runs immediately, but deliberately contains semantic
placeholders that must be replaced before a real run.

## Registration Manifest

`task.json` uses schema version 1:

```json
{
  "schema_version": 1,
  "task_id": "protein_design",
  "description": "Optimize protein candidates against structure objectives.",
  "dependency_checker": "tasks.protein_design.ldm_task.dependencies:check_dependencies"
}
```

Rules:

- `task_id` must match the directory name and be a lowercase Python identifier.
- `description` must be a non-empty one-line domain description.
- `dependency_checker` is optional. When present, it must use
  `python.module:function` notation.
- Module and working-directory paths are convention-derived and cannot be
  overridden by the manifest:
  `tasks.<task_id>.ldm_task.procedure` and `tasks/<task_id>`.
- Unknown manifest fields and schema versions fail registration.

The runner discovers manifests when a process starts. Adding the directory and
manifest is sufficient to make a task ID available to configs.

## Procedure Interface

`ldm_task/procedure.py` must define:

```python
def main(argv: list[str] | None = None) -> int | None:
    ...
```

The shared runner changes into the task directory, applies config environment
variables, imports the procedure module, and calls `main(plan_argv)`. The
adapter owns argument parsing, task-contract description, execution dispatch,
and result exit status. Candidate generation, model calls, surrogate fitting,
acquisition, and evaluators belong in `core/`. Versioned schemas and seed
inputs belong in `resources/`.

For consistency and inspectability, also define:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ...

def describe_ldm_task(...) -> LDMTaskSpec:
    ...
```

`describe_ldm_task` is task-internal and may accept domain-specific prepared
objects. It must describe the candidate space, measured objectives, model
response contract, and acquisition rule using the shared `ldm_tts.spaces`
types. Keep domain dependencies and encodings inside the task implementation.

Every task must provide a deterministic `mode: mock` config that avoids remote
models, external evaluators, GPUs, and large datasets. This is the contract test
used before real dependencies are introduced.

## Optional Dependency Hook

A task-specific hook receives the already-resolved runner plan and returns
shared `DependencyCheck` records:

```python
from typing import Any

from ldm_tts.dependency_checks import (
    DependencyCheck,
    ok,
    plan_check_context,
)


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    task, args, env, cwd, mode = plan_check_context(plan)
    return [ok(task, "adapter", "Task dependency hook loaded.", str(cwd))]
```

Declare the hook in `task.json`. Keep domain checks in the task module; reuse
shared helpers for LLM settings, paths, and CUDA when useful. If the manifest
omits the hook, the dependency checker returns a warning rather than blocking
the task. Keep `dependencies.py` import-light: import optional scientific or
model packages inside the check function so the hook can report that they are
missing instead of failing during module import.

## Config Contract

Create configs under `config/<task_id>/` using the registered `task_id`:

```yaml
name: protein_design_mock
task: protein_design
algorithm: mean
mode: mock
args:
  mock: true
  iterations: 1
```

Task CLI flags belong under `args`, environment variables under `env`, and
literal positional arguments under `extra_args`. A config normally does not
need `runner.cwd` or `runner.module`; those escape hatches are intended for
temporary experiments, not registration.

## Required Verification

Run these checks in order:

```bash
python scripts/validate_tasks.py --task protein_design
uv run --locked --project tasks/protein_design python -m pytest tasks/protein_design/tests
python scripts/check_task_dependencies.py config/protein_design/mock.yaml --no-optional
python scripts/run_ldm_tts.py config/protein_design/mock.yaml --dry-run
python scripts/run_ldm_tts.py config/protein_design/mock.yaml
```

Before adding a real config, replace every generated placeholder, document the
model endpoint and evaluator requirements in the task README, and add a staged
first-real-run sequence: endpoint probe, dependency check, zero-iteration or
dry contract run, then a tiny evaluated run.

## Registration Checklist

- The manifest validates without warnings or errors.
- `main(argv)` runs through the shared runner rather than a separate launcher.
- `ldm_task/` contains only the adapter files accepted by task validation.
- Importable implementation lives under `core/`; versioned inputs live under `resources/`.
- Every generated file is written beneath `runs/` or another explicitly ignored temporary path.
- `describe_ldm_task` matches actual candidates, objectives, response parsing,
  and acquisition behavior.
- Acquisition scoring uses `ldm_tts.acquisition` unless a documented domain
  algorithm requires additional task-local behavior.
- Mock tests cross the same procedure interface as real runs.
- Secrets are supplied through environment variables or ignored local files.
- Generated runs, caches, model downloads, and virtual environments remain
  untracked.
