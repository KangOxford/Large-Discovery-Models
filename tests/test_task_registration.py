from __future__ import annotations

import json
from pathlib import Path

import pytest

import ldm_tts.task_registry as task_registry
from ldm_tts.task_registry import (
    TASK_DEFINITIONS,
    TaskRegistrationError,
    discover_task_definitions,
    load_task_manifest,
    validate_task_layout,
)
from ldm_tts.task_scaffold import TaskScaffoldError, scaffold_task


def test_builtin_tasks_are_discovered_from_manifests() -> None:
    assert set(TASK_DEFINITIONS) == {"antibody", "nanogpt", "small_molecule"}
    for task_id, definition in TASK_DEFINITIONS.items():
        assert definition.relative_root == Path("tasks") / task_id
        assert definition.module == f"tasks.{task_id}.ldm_task.procedure"
        assert definition.manifest_path == Path("tasks") / task_id / "task.json"
        assert definition.dependency_checker


def test_builtin_task_layouts_have_no_validation_errors() -> None:
    for definition in TASK_DEFINITIONS.values():
        issues = validate_task_layout(definition)
        assert [issue for issue in issues if issue.level == "error"] == []


def test_scaffolded_task_is_discoverable_and_valid(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    created = scaffold_task(
        "protein_design",
        description="Optimize protein candidates.",
        repository_root=tmp_path,
    )

    assert len(created) == 9
    definitions = discover_task_definitions(tmp_path)
    definition = definitions["protein_design"]
    assert definition.module == "tasks.protein_design.ldm_task.procedure"
    assert definition.dependency_checker is None
    assert validate_task_layout(definition, repository_root=tmp_path) == []


def test_scaffolder_never_overwrites_existing_task(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    scaffold_task("custom", description="Custom task.", repository_root=tmp_path)

    with pytest.raises(TaskScaffoldError, match="Refusing to overwrite"):
        scaffold_task("custom", description="Replacement.", repository_root=tmp_path)


@pytest.mark.parametrize("task_id", ["BadName", "has-hyphen", "9starts_with_digit"])
def test_scaffolder_rejects_non_package_task_ids(tmp_path: Path, task_id: str) -> None:
    (tmp_path / "tasks").mkdir()
    with pytest.raises(TaskScaffoldError, match="task_id"):
        scaffold_task(task_id, description="Invalid.", repository_root=tmp_path)


def test_manifest_rejects_unknown_fields_and_directory_mismatch(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "tasks" / "wrong_directory"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "task.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "task_id": "different",
        "description": "Mismatch.",
        "extra": True,
    }), encoding="utf-8")

    with pytest.raises(TaskRegistrationError, match="Unknown task manifest"):
        load_task_manifest(manifest, repository_root=tmp_path)

    manifest.write_text(json.dumps({
        "schema_version": 1,
        "task_id": "different",
        "description": "Mismatch.",
    }), encoding="utf-8")
    with pytest.raises(TaskRegistrationError, match="directory name"):
        load_task_manifest(manifest, repository_root=tmp_path)


def test_manifest_rejects_invalid_dependency_hook(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "tasks" / "custom"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "task.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "task_id": "custom",
        "description": "Custom.",
        "dependency_checker": "not a hook",
    }), encoding="utf-8")

    with pytest.raises(TaskRegistrationError, match="dependency_checker"):
        load_task_manifest(manifest, repository_root=tmp_path)


def test_registry_surfaces_discovery_errors_before_lookup(monkeypatch) -> None:
    error = TaskRegistrationError("broken task manifest")
    monkeypatch.setattr(task_registry, "TASK_DISCOVERY_ERROR", error)

    with pytest.raises(TaskRegistrationError, match="broken task manifest"):
        task_registry.get_task_definition("nanogpt")
