"""Repository-root discovery for checkout-oriented commands."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT_ENV = "LDM_REPO_ROOT"


def is_repository_root(path: Path) -> bool:
    """Return whether *path* contains the LDM task and config roots."""
    return (path / "tasks").is_dir() and (path / "config").is_dir()


def resolve_repository_root(
    *,
    source_file: Path,
    cwd: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve the checkout used by manifest-driven commands.

    An explicit ``LDM_REPO_ROOT`` is authoritative and must name a valid
    checkout. Otherwise the current working directory is preferred when it is
    a checkout, followed by the source-tree root used during development.
    """
    env = os.environ if environ is None else environ
    explicit = str(env.get(REPOSITORY_ROOT_ENV, "")).strip()
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not is_repository_root(root):
            raise RuntimeError(
                f"{REPOSITORY_ROOT_ENV} does not point to an LDM checkout: {root}"
            )
        return root

    working_root = Path.cwd().resolve() if cwd is None else Path(cwd).resolve()
    if is_repository_root(working_root):
        return working_root

    resolved_source = Path(source_file).resolve()
    package_root = next(
        (parent for parent in resolved_source.parents if parent.name == "ldm_tts"),
        None,
    )
    if package_root is None:
        raise RuntimeError(f"Could not locate the ldm_tts package from {resolved_source}")
    return package_root.parent
