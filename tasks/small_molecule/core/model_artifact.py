"""Integrity helpers for pickle-compatible model artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Mapping


SHA256_CHUNK_SIZE = 1024 * 1024
PUBLISHED_MODEL_SHA256 = "a4c15c1124eced2e8dc80a18fdf94752da106168209d804002b0defbf63986ed"


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact does not match its declared digest."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(SHA256_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_candidates(model_path: Path) -> tuple[Path, Path]:
    """Return supported metadata sidecar paths in precedence order."""
    return (
        model_path.with_name(model_path.stem + "_metadata.json"),
        model_path.with_name(model_path.stem + ".metadata.json"),
    )


def find_metadata_path(model_path: Path) -> Path | None:
    """Find an existing metadata sidecar for *model_path*."""
    return next((path for path in metadata_candidates(model_path) if path.is_file()), None)


def load_metadata_file(path: Path) -> dict[str, Any]:
    """Load a JSON object from *path*."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Model metadata must be a JSON object: {path}")
    return data


def declared_sha256(metadata: Mapping[str, Any]) -> str | None:
    """Return and validate the SHA-256 declared by artifact metadata."""
    artifact = metadata.get("artifact")
    if artifact is None:
        return None
    if not isinstance(artifact, Mapping):
        raise ArtifactIntegrityError("Model metadata field 'artifact' must be an object.")
    value = artifact.get("sha256")
    if value is None:
        return None
    expected = str(value).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ArtifactIntegrityError("Model metadata contains an invalid SHA-256 digest.")
    return expected


def verify_declared_sha256(model_path: Path, metadata: Mapping[str, Any]) -> str | None:
    """Verify *model_path* when *metadata* declares a SHA-256 digest.

    Returning ``None`` means the caller supplied an artifact with no declared
    digest. Such custom artifacts remain caller-trusted for compatibility.
    """
    expected = declared_sha256(metadata)
    if expected is None:
        return None
    actual = sha256_file(model_path)
    if not hmac.compare_digest(actual, expected):
        raise ArtifactIntegrityError(
            f"Model artifact checksum mismatch for {model_path}: "
            f"expected {expected}, got {actual}. Refusing unsafe deserialization."
        )
    return actual
