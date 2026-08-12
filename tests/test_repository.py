from __future__ import annotations

from pathlib import Path

import pytest

from ldm_tts.repository import (
    REPOSITORY_ROOT_ENV,
    is_repository_root,
    resolve_repository_root,
)


def _make_checkout(path: Path) -> Path:
    (path / "tasks").mkdir(parents=True)
    (path / "config").mkdir()
    return path


def test_is_repository_root_requires_tasks_and_config(tmp_path: Path) -> None:
    assert not is_repository_root(tmp_path)
    (tmp_path / "tasks").mkdir()
    assert not is_repository_root(tmp_path)
    (tmp_path / "config").mkdir()
    assert is_repository_root(tmp_path)


def test_explicit_repository_root_has_precedence(tmp_path: Path) -> None:
    explicit = _make_checkout(tmp_path / "explicit")
    cwd = _make_checkout(tmp_path / "cwd")
    source = tmp_path / "installed" / "ldm_tts" / "repository.py"

    assert resolve_repository_root(
        source_file=source,
        cwd=cwd,
        environ={REPOSITORY_ROOT_ENV: str(explicit)},
    ) == explicit.resolve()


def test_checkout_cwd_precedes_source_fallback(tmp_path: Path) -> None:
    cwd = _make_checkout(tmp_path / "cwd")
    source = tmp_path / "installed" / "ldm_tts" / "repository.py"

    assert resolve_repository_root(source_file=source, cwd=cwd, environ={}) == cwd.resolve()


def test_source_tree_is_fallback_for_library_imports(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "ldm_tts" / "repository.py"

    assert resolve_repository_root(
        source_file=source,
        cwd=tmp_path / "elsewhere",
        environ={},
    ) == source_root.resolve()


def test_invalid_explicit_repository_root_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=REPOSITORY_ROOT_ENV):
        resolve_repository_root(
            source_file=tmp_path / "ldm_tts" / "repository.py",
            cwd=tmp_path,
            environ={REPOSITORY_ROOT_ENV: str(tmp_path / "missing")},
        )
