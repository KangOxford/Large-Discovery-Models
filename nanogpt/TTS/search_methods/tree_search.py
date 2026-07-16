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
    """Expand the full breadth/depth tree, evaluating at configured depth intervals."""
    root_evals = 1 if evaluate_root else 0
    remaining_cap = None if max_evaluations is None else max(0, max_evaluations - root_evals)
    total = estimate_budget(
        breadth,
        depth,
        remaining_cap,
        eval_each_num_steps=engine.config.eval_each_num_steps,
    ) + root_evals
    engine.start_progress(total, label="tree_search")
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

    frontier: list[SearchState] = [root]
    max_depth = max(1, depth)
    for level in range(1, max_depth + 1):
        next_frontier: list[SearchState] = []
        for parent in frontier:
            remaining = None if max_evaluations is None else max_evaluations - engine.evaluation_count
            should_evaluate = engine.should_evaluate_depth(level, max_depth)
            if should_evaluate and remaining is not None and remaining <= 0:
                break
            child_count = breadth if not should_evaluate or remaining is None else min(breadth, remaining)
            children = await engine.expand_state(
                parent,
                child_count,
                search_note=f"tree_search depth {level}: expand this node in the search tree",
            )
            if should_evaluate:
                await engine.evaluate_many(children)
            else:
                await engine.defer_evaluation_many(
                    children,
                    reason=(
                        f"Evaluation deferred at depth {level}; "
                        f"eval_each_num_steps={engine.config.eval_each_num_steps}."
                    ),
                )
            next_frontier.extend(children)
        if not next_frontier:
            break
        frontier = next_frontier

    return engine.best_state()


def estimate_budget(
    breadth: int,
    depth: int,
    max_evaluations: int | None,
    eval_each_num_steps: int = 1,
) -> int:
    total = 0
    parents = 1
    max_depth = max(1, depth)
    interval = max(1, eval_each_num_steps)
    for level in range(1, max_depth + 1):
        parents *= max(1, breadth)
        if level % interval != 0 and level < max_depth:
            continue
        total += parents
        if max_evaluations is not None and total >= max_evaluations:
            return max_evaluations
    return total
