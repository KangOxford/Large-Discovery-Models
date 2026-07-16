"""M1 direct LLM reservoir builder."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from strbo_v1.ldm_tilted_case2.base_measure import apply_m1_base_measure
from strbo_v1.ldm_tilted_case2.canonicalize import RawCandidate, build_candidate_records, canonicalize_smiles
from strbo_v1.ldm_tilted_case2.candidate_record import ReservoirBuildResult, SourceRecord
from strbo_v1.ldm_tilted_case2.config import TiltedLDMCase2Config
from strbo_v1.ldm_tilted_case2.prompts import build_m1_prompt, summarize_history
from strbo_v1.ldm_tilted_case2.schemas import parse_m1_direct_smiles
from strbo_v1.ldm_tilted_case2.sources import call_llm_json


LOGGER = logging.getLogger(__name__)


LLM_DIRECT_CHUNK_SIZE = 8
M1_MAX_PARALLEL_LLM_CALLS = 64
M1_MIN_VALID_CANDIDATES = 32
M1_MAX_REFILL_ROUNDS = 2
M1_ONE_STEP_MAX_REFILL_ROUNDS = 8
M1_OVERSAMPLE_MAX_REFILL_ROUNDS = 4
M1_STRATEGIES = (
    "history-guided local mutation: start from pareto_front and balanced_elites, infer plausible edits from the observed history, make one or two conservative changes per molecule, avoid exact copies, and do not predict objectives",
    "objective-balance exploration: start from top_high_activity and top_low_vina parents, infer the tradeoff patterns from history, generate candidates that test limited changes, and do not predict activity or docking",
    "diversity-preserving proposal: sample alternatives from underrepresented regions of the observed history, avoid many near-duplicates or a single repeated motif, and keep candidates valid and length-bounded",
    "elite crossover and scaffold hop: recombine recognizable fragments from different observed elites or propose near-neighbor scaffold hops based only on the supplied history, without assigning scores",
)


class DirectLLMReservoirBuilder:
    def build(self, history, cfg: TiltedLDMCase2Config, llm, rng) -> ReservoirBuildResult:
        _ = rng
        summary = summarize_history(history, minimize=cfg.minimize)
        batches = _call_m1_batches(summary, cfg, llm)
        candidates, drops, sources, direct_items = _records_from_batches(batches, history, cfg)
        refill_rounds = 0
        while (
            history
            and len(candidates) < _min_valid_candidates(cfg)
            and refill_rounds < _max_refill_rounds(cfg)
        ):
            refill_rounds += 1
            refills = _call_refill_batches(summary, cfg, llm, refill_rounds, candidates, drops)
            batches.extend(refills)
            candidates, drops, sources, direct_items = _records_from_batches(batches, history, cfg)

        _attach_rationales(candidates, direct_items)
        apply_m1_base_measure(candidates, smoothing=cfg.m1_q0_smoothing)
        return ReservoirBuildResult(
            candidates,
            sources,
            "\n".join(batch.result.raw_text for batch in batches),
            {
                "direct_batches": [_batch_metadata(batch) for batch in batches],
                "refill_rounds": refill_rounds,
                "min_valid_candidates": _min_valid_candidates(cfg),
                "requested_total": _requested_total(cfg),
            },
            [attempt for batch in batches for attempt in batch.result.attempts],
            drops,
        )


class _DirectBatch:
    def __init__(self, result, source: SourceRecord, strategy: str | None) -> None:
        self.result = result
        self.source = source
        self.strategy = strategy


def _call_m1_batches(summary, cfg, llm) -> list[_DirectBatch]:
    if cfg.method == "m1_llm_one_step":
        return _call_direct_batches(summary, cfg, llm, 1)
    if cfg.method in {
        "m1_stratified_direct_llm_sir",
        "m1_stratified_direct_llm_oversample_sir",
        "m1_stratified_direct_llm_only",
    }:
        return _call_stratified_batches(summary, cfg, llm)
    return _call_direct_batches(summary, cfg, llm, cfg.m1_k_direct_llm)


def _call_direct_batches(summary, cfg, llm, total_count: int) -> list[_DirectBatch]:
    jobs = []
    for idx, chunk_size in enumerate(_chunk_counts(total_count, LLM_DIRECT_CHUNK_SIZE)):
        source = SourceRecord(f"direct_llm_{idx}", "direct_llm", None, 1.0, chunk_size)
        jobs.append((source, None, chunk_size, None))
    return _execute_direct_jobs(summary, cfg, llm, jobs)


def _call_stratified_batches(summary, cfg, llm) -> list[_DirectBatch]:
    jobs = []
    counts = _strategy_counts(cfg.m1_k_direct_llm, len(M1_STRATEGIES))
    for strategy_idx, (strategy, total_count) in enumerate(zip(M1_STRATEGIES, counts)):
        for chunk_idx, chunk_size in enumerate(_chunk_counts(total_count, LLM_DIRECT_CHUNK_SIZE)):
            source_id = f"m1_strategy_{strategy_idx}_{chunk_idx}"
            source = SourceRecord(
                source_id,
                "direct_llm",
                None,
                total_count / max(1, cfg.m1_k_direct_llm),
                chunk_size,
                metadata={"strategy": strategy},
            )
            jobs.append((source, strategy, chunk_size, None))
    return _execute_direct_jobs(summary, cfg, llm, jobs)


def _execute_direct_jobs(summary, cfg, llm, jobs) -> list[_DirectBatch]:
    parallel = _can_parallelize_llm(llm) and len(jobs) > 1
    max_workers = min(M1_MAX_PARALLEL_LLM_CALLS, len(jobs)) if parallel else 1
    requested = sum(chunk_size for _source, _strategy, chunk_size, _feedback in jobs)
    label = _direct_job_label(jobs)
    t0 = time.monotonic()
    LOGGER.info(
        "%s: launching chunks=%d requested=%d workers=%d chunk_size=%d",
        label,
        len(jobs),
        requested,
        max_workers,
        LLM_DIRECT_CHUNK_SIZE,
    )
    if not parallel:
        results = [
            _call_direct_prompt(
                summary,
                cfg,
                llm,
                chunk_size,
                source_id=source.source_id,
                strategy=strategy,
                feedback=feedback,
            )
            for source, strategy, chunk_size, feedback in jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _call_direct_prompt,
                    summary,
                    cfg,
                    llm,
                    chunk_size,
                    source_id=source.source_id,
                    strategy=strategy,
                    feedback=feedback,
                )
                for source, strategy, chunk_size, feedback in jobs
            ]
            results = [future.result() for future in futures]
    returned = sum(len(result.parsed.direct_smiles) for result in results)
    attempts = sum(len(result.attempts) for result in results)
    LOGGER.info(
        "%s: completed chunks=%d returned=%d attempts=%d elapsed=%.1fs",
        label,
        len(results),
        returned,
        attempts,
        time.monotonic() - t0,
    )
    return [
        _DirectBatch(result, source, strategy)
        for result, (source, strategy, _chunk_size, _feedback) in zip(results, jobs)
    ]


def _direct_job_label(jobs) -> str:
    source_ids = [source.source_id for source, _strategy, _chunk_size, _feedback in jobs]
    if source_ids and all(source_id.startswith("m1_refill_") for source_id in source_ids):
        return "M1 LLM refill"
    if any(source_id.startswith("m1_strategy_") for source_id in source_ids):
        return "M1 stratified LLM reservoir"
    return "M1 direct LLM reservoir"


def _can_parallelize_llm(llm) -> bool:
    return llm.__class__.__name__ == "OpenAIChatClient"


def _call_direct_prompt(
    summary,
    cfg,
    llm,
    chunk_size: int,
    *,
    source_id: str,
    strategy: str | None,
    feedback: dict | None = None,
):
    system, user = build_m1_prompt(
        summary,
        cfg,
        sample_count=chunk_size,
        strategy=strategy,
        feedback=feedback,
    )
    timeout = getattr(llm, "timeout", None)
    t0 = time.monotonic()
    LOGGER.debug(
        "M1 LLM chunk start source=%s requested=%d timeout=%s prompt_chars=%d",
        source_id,
        chunk_size,
        timeout,
        len(user),
    )
    try:
        result = call_llm_json(
            llm,
            system,
            user,
            parse_m1_direct_smiles,
            max_retries=cfg.llm_max_retries,
            retry_wait_seconds=cfg.llm_retry_wait_seconds,
            stage="m1_direct",
            source_id=source_id,
        )
    except Exception:
        LOGGER.exception(
            "M1 LLM chunk failed source=%s requested=%d elapsed=%.1fs",
            source_id,
            chunk_size,
            time.monotonic() - t0,
        )
        raise
    LOGGER.debug(
        "M1 LLM chunk ok source=%s requested=%d returned=%d elapsed=%.1fs attempts=%d",
        source_id,
        chunk_size,
        len(result.parsed.direct_smiles),
        time.monotonic() - t0,
        len(result.attempts),
    )
    return result


def _call_refill_batches(summary, cfg, llm, refill_round: int, candidates, drops) -> list[_DirectBatch]:
    feedback = _refill_feedback(candidates, drops, cfg)
    strategy = (
        "reservoir refill: replace invalid, duplicate, overlength, and already-evaluated outputs "
        "with novel valid candidates; infer diversity only from the supplied history and feedback"
    )
    jobs = []
    for idx, chunk_size in enumerate(_chunk_counts(cfg.m1_k_direct_llm, LLM_DIRECT_CHUNK_SIZE)):
        source = SourceRecord(
            f"m1_refill_{refill_round}_{idx}",
            "direct_llm",
            None,
            1.0,
            chunk_size,
            metadata={"strategy": "reservoir refill", "refill_round": refill_round},
        )
        jobs.append((source, strategy, chunk_size, feedback))
    return _execute_direct_jobs(summary, cfg, llm, jobs)


def _records_from_batches(batches, history, cfg):
    direct_items = [item for batch in batches for item in batch.result.parsed.direct_smiles]
    if cfg.method == "m1_llm_one_step":
        direct_items = direct_items[:1]
    raw_records = [
        raw
        for batch in batches
        for raw in _raw_candidates(
            _batch_items_for_records(batch, cfg),
            batch.source.source_id,
        )
    ]
    sources = [batch.source for batch in batches]
    candidates, drops = build_candidate_records(raw_records, sources, history, cfg)
    return candidates, drops, sources, direct_items


def _refill_feedback(candidates, drops, cfg) -> dict:
    return {
        "valid_candidate_count_after_filters": len(candidates),
        "target_min_valid_candidates": _min_valid_candidates(cfg),
        "drop_counts_after_filters": dict(drops),
        "current_valid_smiles": [
            candidate.canonical_smiles
            for candidate in candidates[: min(len(candidates), 64)]
            if candidate.canonical_smiles
        ],
        "instruction": (
            "Do not repeat current_valid_smiles or avoid_exact_smiles. Replace filtered outputs with "
            "new valid SMILES that are meaningfully different from each other, while staying within "
            "the same API-only proposal role."
        ),
    }


def _min_valid_candidates(cfg) -> int:
    if cfg.method == "m1_llm_one_step":
        return 1
    if cfg.method == "m1_stratified_direct_llm_oversample_sir":
        return min(cfg.max_candidates_per_round, max(1, cfg.m1_k_direct_llm))
    return min(M1_MIN_VALID_CANDIDATES, max(1, cfg.m1_k_direct_llm // 2))


def _max_refill_rounds(cfg) -> int:
    if cfg.method == "m1_llm_one_step":
        return M1_ONE_STEP_MAX_REFILL_ROUNDS
    if cfg.method == "m1_stratified_direct_llm_oversample_sir":
        return M1_OVERSAMPLE_MAX_REFILL_ROUNDS
    return M1_MAX_REFILL_ROUNDS


def _requested_total(cfg) -> int:
    if cfg.method == "m1_llm_one_step":
        return 1
    return int(cfg.m1_k_direct_llm)


def _chunk_counts(total_count: int, chunk_size: int) -> list[int]:
    counts: list[int] = []
    remaining = max(0, total_count)
    while remaining > 0:
        current = min(chunk_size, remaining)
        counts.append(current)
        remaining -= current
    return counts


def _strategy_counts(total_count: int, n_strategies: int) -> list[int]:
    base = max(0, total_count) // n_strategies
    remainder = max(0, total_count) % n_strategies
    return [base + (1 if idx < remainder else 0) for idx in range(n_strategies)]


def _raw_candidates(items, source_id: str) -> list[RawCandidate]:
    return [
        RawCandidate(item.smiles, source_id, metadata={"rationale": item.rationale})
        for item in items
    ]


def _batch_metadata(batch: _DirectBatch) -> dict:
    out = dict(batch.result.parsed_json)
    if batch.strategy:
        out["strategy"] = batch.strategy
    out["source_id"] = batch.source.source_id
    out["requested_count"] = batch.source.requested_budget
    return out


def _batch_items_for_records(batch: _DirectBatch, cfg):
    items = batch.result.parsed.direct_smiles
    if cfg.method != "m1_llm_one_step":
        return items
    return items[: max(0, batch.source.requested_budget)]


def _attach_rationales(candidates, items) -> None:
    rationales: dict[str, list[str]] = defaultdict(list)
    for item in items:
        canonical = canonicalize_smiles(item.smiles)
        if canonical is not None:
            rationales[canonical].append(item.rationale)
    for candidate in candidates:
        candidate.metadata["rationales"] = rationales.get(candidate.canonical_smiles or "", [])
