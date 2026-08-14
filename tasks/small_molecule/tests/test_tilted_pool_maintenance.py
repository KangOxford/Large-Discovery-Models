import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import CandidateRecord
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.pool_maintenance import maintain_candidate_pool
from tasks.small_molecule.core.rng import RNG


def _candidate(smiles: str, q0: float) -> CandidateRecord:
    return CandidateRecord(
        raw_smiles=smiles,
        canonical_smiles=smiles,
        method="test",
        sources=["source"],
        occurrence_by_source={"source": 1},
        q0_base_mass=q0,
    )


def test_pool_maintenance_uses_q0_instead_of_first_n_truncation():
    candidates = [_candidate(f"C{i}", 1.0) for i in range(10)]
    candidates.append(_candidate("HIGH_Q0_LAST", 1e9))
    cfg = TiltedLDMCase2Config(
        "m1_stratified_direct_llm_oversample_sir",
        max_candidates_per_round=3,
    )

    kept, metadata = maintain_candidate_pool(candidates, cfg, RNG(0))

    assert len(kept) == 3
    assert "HIGH_Q0_LAST" in {candidate.canonical_smiles for candidate in kept}
    assert metadata["oversample_candidate_count"] == 11
    assert metadata["maintained_candidate_count"] == 3
