from __future__ import annotations

from TTS.search_core import SearchEngine, SearchState


async def run(
    engine: SearchEngine,
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> SearchState | None:
    """Sample independent rollout branches and evaluate at configured depth intervals."""
    root_evals = 1 if evaluate_root else 0
    total = max(1, breadth) * len(engine.evaluation_depths(max(1, depth)))
    if max_evaluations is not None:
        total = min(total, max(0, max_evaluations - root_evals))
    total += root_evals
    engine.start_progress(total, label="best_of_n")
    try:
        return await _run(
            engine,
            breadth=breadth,
            depth=depth,
            max_evaluations=max_evaluations,
            evaluate_root=evaluate_root,
        )
    finally:
        engine.finish_progress()


async def _run(
    engine: SearchEngine,
    *,
    breadth: int,
    depth: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> SearchState | None:
    root = engine.create_seed_state()
    if evaluate_root:
        await engine.evaluate_many([root])

    branch_count = max(1, breadth)
    branch_depth = max(1, depth)
    stop = False
    for branch_index in range(1, branch_count + 1):
        parent = root
        for level in range(1, branch_depth + 1):
            remaining = None if max_evaluations is None else max_evaluations - engine.evaluation_count
            should_evaluate = engine.should_evaluate_depth(level, branch_depth)
            if should_evaluate and remaining is not None and remaining <= 0:
                stop = True
                break
            children = await engine.expand_state(
                parent,
                1,
                search_note=(
                    f"best_of_n branch {branch_index}/{branch_count}, "
                    f"step {level}/{branch_depth}: continue this independent rollout branch."
                ),
            )
            if not children:
                break
            child = children[0]
            if should_evaluate:
                await engine.evaluate_many([child])
            else:
                await engine.defer_evaluation_many(
                    [child],
                    reason=(
                        f"Evaluation deferred at depth {level}; "
                        f"eval_each_num_steps={engine.config.eval_each_num_steps}."
                    ),
                )
            parent = child
        if stop:
            break
    return engine.best_state()
