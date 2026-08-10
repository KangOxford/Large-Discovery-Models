# Task Contract Reference

## Registration

Required layout:

```text
tasks/<task_id>/
├── task.json
├── README.md
├── pyproject.toml
├── __init__.py
├── ldm_task/
│   ├── __init__.py
│   ├── procedure.py
│   └── dependencies.py
├── core/
│   └── __init__.py
├── resources/
│   └── README.md
└── tests/

config/<task_id>/
└── mock.yaml
```

Manifest schema version 1 accepts only:

```json
{
  "schema_version": 1,
  "task_id": "example_task",
  "description": "One-line domain description.",
  "dependency_checker": "tasks.example_task.ldm_task.dependencies:check_dependencies"
}
```

`dependency_checker` is optional. The directory name and `task_id` must match.
The module and working directory are inferred as
`tasks.<task_id>.ldm_task.procedure` and `tasks/<task_id>`.

## Procedure

Required external interface:

```python
def main(argv: list[str] | None = None) -> int | None:
    ...
```

Recommended task-local interfaces:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ...

def describe_ldm_task(...) -> LDMTaskSpec:
    ...
```

The runner applies config environment variables, changes to the task directory,
imports the conventional module, and calls `main(argv)`. The task owns all
domain execution behind that interface. Keep `ldm_task/procedure.py` as a thin
adapter; put importable search, model, surrogate, and evaluator implementation
under `core/`, and versioned runtime inputs under `resources/`.

## Dependency Hook

Use this exact callable shape:

```python
from typing import Any

from ldm_tts.dependency_checks import DependencyCheck, plan_check_context


def check_dependencies(
    plan: dict[str, Any], *, include_optional: bool = True
) -> list[DependencyCheck]:
    task, args, env, cwd, mode = plan_check_context(plan)
    ...
```

Return only `DependencyCheck` objects. Use `ok`, `warn`, `fail`, and `skip`
from `ldm_tts.dependency_checks`. Never print or return unmasked credentials.

## Fine-Tuning Data Collection

Tasks that accept model-generated actions must expose an opt-in runtime
collection path through the public `ldm_tts.data` module:

```python
from ldm_tts.data import DataCollectionSink

sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
```

Append only after the task parser and validator have accepted the action. Build
the canonical `ldm-2.0` IR from the accepted payload, not from an unvalidated
response transcript. Store run IDs, task-specific source IDs, selection results,
and evaluator outcomes under `collection.provenance` or `collection.outcome` so
the renderer cannot leak them into the prompt.

Do not collect rejected attempts, deterministic/random fallbacks, or a different
semantic response type under an existing dataset contract. For example, direct
sequence proposals and search-policy DSL updates require separate action
contracts. Use a run-local ignored `ldm_data/` directory unless
`LDM_DATA_COLLECTION_DIR` explicitly selects an aggregate campaign directory.

The mock task test must enable `LDM_DATA_COLLECTION_ENABLED=1`, execute at least
one accepted action, validate the emitted IR, and verify that collection-only
metadata is absent from rendered SFT instructions. If collection is inapplicable,
the task README must state why and identify the future accepted-action boundary.

## Completion Gates

- `scripts/validate_tasks.py --task <task_id>` has no errors.
- `ldm_task/` contains only the runner and dependency-check adapters.
- Importable implementation and static inputs live in `core/` and `resources/`.
- Task tests pass in the task environment.
- Mock dependency check passes without external systems.
- Mock runner dry-run resolves the registered module and task directory.
- Mock runner execution succeeds.
- Shared tests and `git diff --check` pass.
- The mock collection test emits valid `ldm-2.0` IR, or the task documents why
  its response contract is not collectable.
- No scaffold placeholders, task-name dispatch branches, secrets, or generated
  artifacts remain.
