"""Tests for model artifact integrity verification."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tasks.small_molecule.core.model_artifact import (  # noqa: E402
    ArtifactIntegrityError,
    PUBLISHED_MODEL_SHA256,
    declared_sha256,
    find_metadata_path,
    load_metadata_file,
    metadata_candidates,
    sha256_file,
    verify_declared_sha256,
)


PUBLISHED_METADATA = (
    _PROJECT_ROOT / "resources" / "models" / "best_g12d_model_metadata.json"
)


def test_published_metadata_records_reference_digest() -> None:
    metadata = load_metadata_file(PUBLISHED_METADATA)
    assert metadata["artifact"]["filename"] == "best_g12d_model.joblib"
    assert declared_sha256(metadata) == PUBLISHED_MODEL_SHA256


def test_sidecar_discovery_prefers_training_convention(tmp_path: Path) -> None:
    model = tmp_path / "model.joblib"
    preferred, fallback = metadata_candidates(model)
    fallback.write_text("{}", encoding="utf-8")
    assert find_metadata_path(model) == fallback
    preferred.write_text("{}", encoding="utf-8")
    assert find_metadata_path(model) == preferred


def test_no_sidecar_or_declared_digest_is_allowed_for_custom_models(tmp_path: Path) -> None:
    model = tmp_path / "custom.joblib"
    model.write_bytes(b"caller-trusted")
    assert find_metadata_path(model) is None
    assert declared_sha256({}) is None
    assert declared_sha256({"artifact": {}}) is None
    assert verify_declared_sha256(model, {}) is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"artifact": "not-an-object"},
        {"artifact": {"sha256": "short"}},
        {"artifact": {"sha256": "z" * 64}},
    ],
)
def test_invalid_integrity_metadata_is_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises(ArtifactIntegrityError):
        declared_sha256(metadata)


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "model.joblib"
    model.write_bytes(b"tampered")
    expected = hashlib.sha256(b"original").hexdigest()
    with pytest.raises(ArtifactIntegrityError, match="Refusing unsafe deserialization"):
        verify_declared_sha256(model, {"artifact": {"sha256": expected}})


def test_matching_synthetic_artifact_is_accepted(tmp_path: Path) -> None:
    model = tmp_path / "model.joblib"
    model.write_bytes(b"synthetic model")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    assert sha256_file(model) == expected
    assert verify_declared_sha256(
        model, {"artifact": {"sha256": expected}}
    ) == expected


def test_metadata_must_be_json_object(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_metadata_file(metadata)
