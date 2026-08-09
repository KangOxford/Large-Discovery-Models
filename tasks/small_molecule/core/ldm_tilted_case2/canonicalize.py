"""SMILES canonicalization and hard filters for tilted case2 reservoirs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import CandidateRecord, SourceRecord
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config


@dataclass
class RawCandidate:
    smiles: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    base_support_level: str | None = None
    base_support_value: float = 0.0


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or None when RDKit rejects the input."""
    text = str(smiles or "").strip()
    if not text or "." in text:
        return None
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if "." in canonical:
            return None
        return canonical
    except ImportError:
        return text


def build_candidate_records(
    raw_records: Sequence[RawCandidate],
    sources: Sequence[SourceRecord],
    history: Sequence[tuple[str, Sequence[float | None]]],
    cfg: TiltedLDMCase2Config,
) -> tuple[list[CandidateRecord], dict[str, int]]:
    """Apply validity, length, evaluated and duplicate hard filters."""
    _ = {source.source_id for source in sources}
    drops: Counter[str] = Counter()
    evaluated = _canonical_history_set(history)
    merged: dict[str, CandidateRecord] = {}

    for raw in raw_records:
        canonical = canonicalize_smiles(raw.smiles)
        if canonical is None:
            drops["invalid"] += 1
            continue
        if len(canonical) > cfg.smiles_max_len:
            drops["overlength"] += 1
            continue
        if canonical in evaluated:
            drops["evaluated"] += 1
            continue
        if canonical in merged:
            _merge_candidate(merged[canonical], raw)
            drops["duplicate"] += 1
            continue
        merged[canonical] = CandidateRecord(
            raw_smiles=raw.smiles,
            canonical_smiles=canonical,
            method=cfg.method,
            sources=[raw.source_id],
            occurrence_by_source={raw.source_id: 1},
            base_support_level=raw.base_support_level,
            base_support_value=float(raw.base_support_value),
            metadata=dict(raw.metadata),
        )

    _update_source_valid_counts(sources, merged.values())
    return list(merged.values()), dict(drops)


def _canonical_history_set(history: Sequence[tuple[str, Sequence[float | None]]]) -> set[str]:
    evaluated: set[str] = set()
    for smiles, _scores in history:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            evaluated.add(canonical)
    return evaluated


def _merge_candidate(record: CandidateRecord, raw: RawCandidate) -> None:
    record.occurrence_by_source[raw.source_id] = (
        record.occurrence_by_source.get(raw.source_id, 0) + 1
    )
    if raw.source_id not in record.sources:
        record.sources.append(raw.source_id)
    record.metadata.setdefault("merged_raw_smiles", []).append(raw.smiles)


def _update_source_valid_counts(
    sources: Sequence[SourceRecord],
    records: Sequence[CandidateRecord],
) -> None:
    counts: Counter[str] = Counter()
    for record in records:
        for source_id, count in record.occurrence_by_source.items():
            counts[source_id] += int(count)
    for source in sources:
        source.valid_count = int(counts.get(source.source_id, 0))
