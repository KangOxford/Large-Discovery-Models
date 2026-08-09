import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from strbo_v1.ldm_tilted_case2.base_measure import (
    apply_m1_base_measure,
    q0_effective_support,
    q0_entropy,
)
from strbo_v1.ldm_tilted_case2.candidate_record import CandidateRecord, SourceRecord
from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config


def candidate(smiles, occurrences, *, support=None):
    return CandidateRecord(
        raw_smiles=smiles,
        canonical_smiles=smiles,
        method="test",
        sources=list(occurrences),
        occurrence_by_source=dict(occurrences),
        base_support_level=support,
    )


def test_candidate_record_roundtrip():
    record = candidate("CCO", {"s1": 2}, support="strong")
    record.mu = [1.0, 2.0]
    rebuilt = CandidateRecord.from_dict(record.to_dict())
    assert rebuilt == record

    source = SourceRecord("s1", "direct_llm", None, 1.0, 4, metadata={"x": 1})
    assert SourceRecord.from_dict(source.to_dict()) == source


def test_m1_q0_from_occurrence_counts():
    records = [candidate("CCO", {"s1": 2}), candidate("CCN", {"s1": 1})]
    q0 = apply_m1_base_measure(records)
    np.testing.assert_allclose(q0, np.array([2 / 3, 1 / 3]))
    assert [r.q0_base_mass for r in records] == list(q0)


def test_m1_unique_candidates_uniform():
    records = [candidate("CCO", {"s1": 1}), candidate("CCN", {"s1": 1})]
    q0 = apply_m1_base_measure(records)
    np.testing.assert_allclose(q0, np.array([0.5, 0.5]))


def test_m1_q0_smoothing_reduces_duplicate_dominance():
    records = [candidate("CCO", {"s1": 3}), candidate("CCN", {"s1": 1})]
    q0 = apply_m1_base_measure(records, smoothing=1.0)
    np.testing.assert_allclose(q0, np.array([4 / 6, 2 / 6]))


def test_q0_entropy_and_effective_support():
    q0 = np.array([0.5, 0.5])
    assert math.isclose(q0_entropy(q0), math.log(2.0))
    assert math.isclose(q0_effective_support(q0), 2.0)
