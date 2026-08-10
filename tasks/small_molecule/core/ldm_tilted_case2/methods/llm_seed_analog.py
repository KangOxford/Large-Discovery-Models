"""M1-style LLM seed plus analogue oversampling reservoir builder."""

from __future__ import annotations

import math
from dataclasses import asdict

from tasks.small_molecule.core.ldm_tilted_case2.base_measure import apply_m1_base_measure
from tasks.small_molecule.core.ldm_tilted_case2.canonicalize import RawCandidate, build_candidate_records, canonicalize_smiles
from tasks.small_molecule.core.ldm_tilted_case2.candidate_record import ReservoirBuildResult, SourceRecord
from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.prompts import build_m1_analog_seed_prompt, summarize_history
from tasks.small_molecule.core.ldm_tilted_case2.schemas import parse_seed_plan
from tasks.small_molecule.core.ldm_tilted_case2.sources import call_llm_json, call_reasyn_source


M1_ANALOG_MAX_REFILL_ROUNDS = 3


class LLMSeedAnalogReservoirBuilder:
    def build(
        self,
        history,
        cfg: TiltedLDMCase2Config,
        llm,
        analog_fn,
        rng,
    ) -> ReservoirBuildResult:
        _ = rng
        summary = summarize_history(history, minimize=cfg.minimize)
        seed_results, seeds = _call_seed_results(summary, cfg, llm)
        sources, raw_records = _expand_seed_sources(seeds, cfg, analog_fn)
        candidates, drops = build_candidate_records(raw_records, sources, history, cfg)
        analog_refill_rounds = 0
        seen_seed_smiles = [seed.smiles for seed in seeds]
        while (
            history
            and len(candidates) < cfg.max_candidates_per_round
            and analog_refill_rounds < M1_ANALOG_MAX_REFILL_ROUNDS
        ):
            analog_refill_rounds += 1
            feedback = _analog_refill_feedback(candidates, drops, cfg, seen_seed_smiles)
            refill_results, refill_seeds = _call_seed_results(
                summary,
                cfg,
                llm,
                feedback=feedback,
                avoid_seed_smiles=seen_seed_smiles,
            )
            if not refill_seeds:
                seed_results.extend(refill_results)
                break
            seen_seed_smiles.extend(seed.smiles for seed in refill_seeds)
            seed_results.extend(refill_results)
            refill_sources, refill_raw = _expand_seed_sources(
                refill_seeds,
                cfg,
                analog_fn,
                source_offset=len(sources),
            )
            sources.extend(refill_sources)
            raw_records.extend(refill_raw)
            candidates, drops = build_candidate_records(raw_records, sources, history, cfg)
        apply_m1_base_measure(candidates, smoothing=cfg.m1_q0_smoothing)
        return ReservoirBuildResult(
            candidates,
            sources,
            "\n".join(result.raw_text for result in seed_results),
            {
                "seed_plan": {"seeds": [asdict(seed) for seed in seeds]},
                "seed_batches": [result.parsed_json for result in seed_results],
                "requested_seed_count": cfg.m1_analog_n_llm_seeds,
                "requested_analog_total": cfg.m1_analog_k_total,
                "analog_refill_rounds": analog_refill_rounds,
                "min_valid_candidates": cfg.max_candidates_per_round,
            },
            [attempt for result in seed_results for attempt in result.attempts],
            drops,
        )


def _call_seed_results(
    summary,
    cfg: TiltedLDMCase2Config,
    llm,
    *,
    feedback: dict | None = None,
    avoid_seed_smiles: list[str] | None = None,
):
    results = []
    seeds = []
    seen = {
        canonical
        for smiles in (avoid_seed_smiles or [])
        for canonical in [canonicalize_smiles(smiles)]
        if canonical is not None
    }
    for refill_idx in range(cfg.llm_max_retries + 1):
        result = call_llm_json(
            llm,
            *build_m1_analog_seed_prompt(summary, cfg, feedback=feedback),
            parse_seed_plan,
            max_retries=cfg.llm_max_retries,
            retry_wait_seconds=cfg.llm_retry_wait_seconds,
            stage="m1_analog_seed_plan",
            source_id=f"m1_analog_seed_plan_{refill_idx}",
        )
        results.append(result)
        for seed in result.parsed.seeds:
            canonical = canonicalize_smiles(seed.smiles)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            seeds.append(seed)
            if len(seeds) >= cfg.m1_analog_n_llm_seeds:
                return results, seeds[: cfg.m1_analog_n_llm_seeds]
        missing = cfg.m1_analog_n_llm_seeds - len(seeds)
        feedback = {
            "instruction": f"Need {missing} additional seed molecule(s).",
            "current_seed_smiles": [seed.smiles for seed in seeds],
            "avoid_seed_smiles": list(avoid_seed_smiles or []),
        }
    return results, seeds[: cfg.m1_analog_n_llm_seeds]


def _expand_seed_sources(
    seeds,
    cfg: TiltedLDMCase2Config,
    analog_fn,
    *,
    source_offset: int = 0,
):
    sources: list[SourceRecord] = []
    raw_records = []
    if not seeds:
        return sources, raw_records
    per_seed_budget = max(1, math.ceil(cfg.m1_analog_k_total / len(seeds)))
    for idx, seed in enumerate(seeds):
        sources.append(_source_for_seed(source_offset + idx, seed, len(seeds), per_seed_budget))
    target_generator = getattr(analog_fn, "generate_with_targets", None)
    if callable(target_generator):
        raw_records = _expand_targets_in_batch(seeds, sources, target_generator, per_seed_budget)
        return sources, raw_records
    for idx, seed in enumerate(seeds):
        source = sources[idx]
        records = call_reasyn_source(
            analog_fn,
            [seed.smiles],
            source_id=source.source_id,
            budget=per_seed_budget,
        )
        source.generated_count = len(records)
        raw_records.extend(records)
    return sources, raw_records


def _analog_refill_feedback(candidates, drops, cfg: TiltedLDMCase2Config, seed_smiles) -> dict:
    return {
        "instruction": (
            f"Need additional analogue-expansion seeds because only {len(candidates)} valid "
            f"candidates remain after filters; target_min_valid_candidates={cfg.max_candidates_per_round}."
        ),
        "valid_candidate_count_after_filters": len(candidates),
        "target_min_valid_candidates": cfg.max_candidates_per_round,
        "drop_counts_after_filters": dict(drops),
        "avoid_seed_smiles": list(seed_smiles),
    }


def _source_for_seed(idx, seed, seed_count: int, budget: int) -> SourceRecord:
    return SourceRecord(
        f"m1_analog_seed_{idx}",
        "reasyn",
        seed.smiles,
        1.0 / seed_count,
        budget,
        metadata={"intent": seed.intent},
    )


def _expand_targets_in_batch(seeds, sources, target_generator, per_seed_budget: int):
    seed_by_canonical = {
        canonicalize_smiles(seed.smiles) or seed.smiles: idx
        for idx, seed in enumerate(seeds)
    }
    counts = [0 for _ in sources]
    rows = _iter_target_rows(target_generator([seed.smiles for seed in seeds]))
    out: list[RawCandidate] = []
    for row in rows:
        target = canonicalize_smiles(str(row.get("target", ""))) or str(row.get("target", ""))
        idx = seed_by_canonical.get(target)
        if idx is None or counts[idx] >= per_seed_budget:
            continue
        smiles = str(row.get("smiles", "")).strip()
        if not smiles:
            continue
        source = sources[idx]
        out.append(
            RawCandidate(
                smiles,
                source.source_id,
                metadata={"seed_smiles": source.seed_smiles},
            )
        )
        counts[idx] += 1
    for source, count in zip(sources, counts):
        source.generated_count = count
    return out


def _iter_target_rows(table):
    if hasattr(table, "iterrows"):
        for _idx, row in table.iterrows():
            yield row
        return
    yield from table
