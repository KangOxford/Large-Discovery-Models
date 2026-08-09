"""Deterministic scaffolding for manifest-registered LDM tasks."""

from __future__ import annotations

import json
from pathlib import Path

from ldm_tts.task_registry import REPO_ROOT, TASK_ID_PATTERN


class TaskScaffoldError(ValueError):
    """Raised when a task skeleton cannot be created without overwriting files."""


def scaffold_task(
    task_id: str,
    *,
    description: str,
    repository_root: Path = REPO_ROOT,
) -> tuple[Path, ...]:
    """Create a conventional task skeleton and mock experiment config."""

    task_id = str(task_id).strip()
    description = str(description).strip()
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise TaskScaffoldError(
            "task_id must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores"
        )
    if not description:
        raise TaskScaffoldError("description must not be empty")

    repository_root = Path(repository_root).resolve()
    task_root = repository_root / "tasks" / task_id
    config_root = repository_root / "config" / task_id
    files = _task_files(task_id, description, task_root, config_root)
    conflicts = sorted(path for path in files if path.exists())
    if task_root.exists() or conflicts:
        conflict_text = ", ".join(str(path) for path in conflicts) or str(task_root)
        raise TaskScaffoldError(
            f"Refusing to overwrite existing task files: {conflict_text}"
        )

    created: list[Path] = []
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return tuple(created)


def _task_files(
    task_id: str,
    description: str,
    task_root: Path,
    config_root: Path,
) -> dict[Path, str]:
    package_name = task_id.replace("_", "-")
    manifest = {
        "schema_version": 1,
        "task_id": task_id,
        "description": description,
    }
    return {
        task_root / "task.json": json.dumps(manifest, indent=2) + "\n",
        task_root / "__init__.py": f'"""{description}"""\n',
        task_root / "ldm_task" / "__init__.py": '"""Shared-runner adapter for this task."""\n',
        task_root / "ldm_task" / "procedure.py": _procedure_template(task_id),
        task_root / "tests" / "__init__.py": "",
        task_root / "tests" / "test_procedure.py": _test_template(task_id),
        task_root / "README.md": _readme_template(task_id, description),
        task_root / "pyproject.toml": _pyproject_template(package_name, description),
        config_root / "mock.yaml": _config_template(task_id),
    }


def _procedure_template(task_id: str) -> str:
    return f'''#!/usr/bin/env python3
"""Procedure adapter for the ``{task_id}`` task."""

from __future__ import annotations

import argparse
import json

from ldm_tts.spaces import (
    AcquisitionSpec,
    CandidateSpaceSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ResponseSpaceSpec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the {task_id} LDM task.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def describe_ldm_task(args: argparse.Namespace) -> LDMTaskSpec:
    return LDMTaskSpec(
        task="{task_id}",
        candidate_space=CandidateSpaceSpec(
            name="replace_me",
            kind="replace_me",
            dimension=None,
            representation="Replace with the task candidate representation.",
        ),
        objectives=(
            ObjectiveSpec(
                name="objective",
                direction="maximize",
                description="Replace with the measured task objective.",
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="proposal",
                output_kind="json",
                description="Replace with the model response contract.",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="mean",
            objective_names=("objective",),
            score_direction="maximize",
            selection_rule="Replace with the task selection rule.",
        ),
        metadata={{"mock": bool(args.mock)}},
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.mock and not args.dry_run:
        raise SystemExit("Implement real task execution before running without --mock.")
    print(json.dumps({{
        "task": "{task_id}",
        "iterations": max(0, args.iterations),
        "mock": bool(args.mock),
        "ldm_task_spec": describe_ldm_task(args).to_dict(),
    }}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _test_template(task_id: str) -> str:
    return f'''from tasks.{task_id}.ldm_task.procedure import main, parse_args


def test_mock_procedure(capsys) -> None:
    assert parse_args(["--mock", "--iterations", "0"]).iterations == 0
    assert main(["--mock", "--iterations", "0"]) == 0
    assert '"task": "{task_id}"' in capsys.readouterr().out
'''


def _readme_template(task_id: str, description: str) -> str:
    return f'''# {task_id}

{description}

## Mock Run

From the repository root:

```bash
uv sync --project tasks/{task_id} --group dev
uv run --project tasks/{task_id} python scripts/run_ldm_tts.py config/{task_id}/mock.yaml
```

Replace the generated placeholder candidate space, objective, response contract,
and execution loop before adding a real-run config.
'''


def _pyproject_template(package_name: str, description: str) -> str:
    escaped_description = description.replace('"', '\\"')
    return f'''[project]
name = "ldm-tts-{package_name}"
version = "0.1.0"
description = "{escaped_description}"
requires-python = ">=3.10"
dependencies = []

[dependency-groups]
dev = [
    "pytest>=7.0",
]

[tool.uv]
package = false
'''


def _config_template(task_id: str) -> str:
    return f'''name: {task_id}_mock
task: {task_id}
algorithm: mean
mode: mock
description: Local contract smoke test for {task_id}.
args:
  mock: true
  iterations: 1
'''


__all__ = ["TaskScaffoldError", "scaffold_task"]
