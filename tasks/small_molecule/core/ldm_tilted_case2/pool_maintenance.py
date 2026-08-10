"""Reservoir maintenance for oversized case2 candidate pools."""

from __future__ import annotations

import numpy as np

from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import CandidateRecord
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.resampling import gumbel_top_k
from tasks.small_molecule.core.rng import RNG


def maintain_candidate_pool(
    candidates: list[CandidateRecord],
    cfg: TiltedLDMCase2Config,
    rng: RNG,
) -> tuple[list[CandidateRecord], dict[str, int | str]]:
    """Reduce an oversized reservoir to the configured BO pool size."""
    original_count = len(candidates)
    limit = int(cfg.max_candidates_per_round)
    metadata: dict[str, int | str] = {
        "oversample_candidate_count": original_count,
        "maintained_candidate_count": min(original_count, limit),
        "pool_maintenance_method": "none",
    }
    if original_count <= limit:
        return candidates, metadata

    q0 = np.asarray([candidate.q0_base_mass for candidate in candidates], dtype=float)
    indices = gumbel_top_k(q0, limit, rng)
    kept = [candidates[idx] for idx in sorted(indices)]
    metadata["pool_maintenance_method"] = "q0_gumbel_top_k"
    metadata["maintained_candidate_count"] = len(kept)
    return kept, metadata
