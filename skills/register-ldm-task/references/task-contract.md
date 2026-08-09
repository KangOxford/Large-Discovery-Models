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
│   └── procedure.py
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
domain execution behind that interface.

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

## Completion Gates

- `scripts/validate_tasks.py --task <task_id>` has no errors.
- Task tests pass in the task environment.
- Mock dependency check passes without external systems.
- Mock runner dry-run resolves the registered module and task directory.
- Mock runner execution succeeds.
- Shared tests and `git diff --check` pass.
- No scaffold placeholders, task-name dispatch branches, secrets, or generated
  artifacts remain.
