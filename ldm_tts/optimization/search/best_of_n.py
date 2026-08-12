"""Independent-rollout proposal search."""

from __future__ import annotations

from ldm_tts.optimization.search.protocols import SearchEngine, StateT
from ldm_tts.optimization.search.traversal import best_of_n as traverse


async def run(
    engine: SearchEngine[StateT],
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> StateT | None:
    """Sample independent rollout branches and evaluate configured depths."""

    root_evals = 1 if evaluate_root else 0
    evaluated_depths = sum(
        engine.should_evaluate_depth(level, max(1, depth))
        for level in range(1, max(1, depth) + 1)
    )
    total = max(1, breadth) * evaluated_depths
    if max_evaluations is not None:
        total = min(total, max(0, max_evaluations - root_evals))
    engine.start_progress(total + root_evals, label="best_of_n")
    try:
        root = engine.create_seed_state()
        if evaluate_root:
            await engine.evaluate_many([root])
        await traverse(
            engine,
            root,
            breadth=breadth,
            depth=depth,
            beam_width=beam_width,
            max_evaluations=max_evaluations,
        )
        return engine.best_state()
    finally:
        engine.finish_progress()
