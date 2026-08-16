"""Strict loader for the pinned MLS-Bench provenance contract."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


UPSTREAM_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "upstream_contract.json"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_upstream_contract() -> dict[str, Any]:
    try:
        payload = json.loads(UPSTREAM_CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot load pinned upstream contract {UPSTREAM_CONTRACT_PATH}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("upstream_contract.json must contain a JSON object")
    expected_fields = {
        "source_url",
        "source_commit",
        "task_path",
        "editable_file",
        "embedding_dimension",
        "parameter_budget",
        "sha256",
    }
    if set(payload) != expected_fields:
        raise RuntimeError(
            "upstream_contract.json fields do not match the task provenance contract"
        )
    for field in ("source_url", "source_commit", "task_path", "editable_file"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise RuntimeError(f"upstream_contract.json field {field!r} is invalid")
    if not _COMMIT_PATTERN.fullmatch(payload["source_commit"]):
        raise RuntimeError("upstream_contract.json source_commit must be a full Git SHA")
    if not isinstance(payload["embedding_dimension"], int):
        raise RuntimeError("upstream_contract.json embedding_dimension must be an integer")
    if not isinstance(payload["parameter_budget"], dict):
        raise RuntimeError("upstream_contract.json parameter_budget must be an object")
    hashes = payload["sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("upstream_contract.json sha256 must be a non-empty object")
    for relative, digest in hashes.items():
        if (
            not isinstance(relative, str)
            or not relative.strip()
            or not isinstance(digest, str)
            or not _SHA256_PATTERN.fullmatch(digest)
        ):
            raise RuntimeError("upstream_contract.json contains an invalid SHA-256 entry")
    return payload


UPSTREAM_CONTRACT = _load_upstream_contract()
OFFICIAL_SOURCE_URL = str(UPSTREAM_CONTRACT["source_url"])
OFFICIAL_COMMIT = str(UPSTREAM_CONTRACT["source_commit"])
TASK_PATH = str(UPSTREAM_CONTRACT["task_path"])
_ALL_SHA256 = dict(UPSTREAM_CONTRACT["sha256"])
UPSTREAM_ROOT_SHA256 = {
    relative: digest
    for relative, digest in _ALL_SHA256.items()
    if relative.startswith("src/")
}
UPSTREAM_SHA256 = {
    relative: digest
    for relative, digest in _ALL_SHA256.items()
    if relative not in UPSTREAM_ROOT_SHA256
}


__all__ = [
    "OFFICIAL_COMMIT",
    "OFFICIAL_SOURCE_URL",
    "TASK_PATH",
    "UPSTREAM_CONTRACT",
    "UPSTREAM_CONTRACT_PATH",
    "UPSTREAM_ROOT_SHA256",
    "UPSTREAM_SHA256",
]
