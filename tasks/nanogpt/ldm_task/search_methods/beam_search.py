from __future__ import annotations

from tasks.nanogpt.ldm_task.search_core import SearchEngine, SearchState


async def run(
    engine: SearchEngine,
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> SearchState | None:
    """Expand the current beam, evaluating/pruning at configured depth intervals."""
    root_evals = 1 if evaluate_root else 0
    remaining_cap = None if max_evaluations is None else max(0, max_evaluations - root_evals)
    total = estimate_budget(
        breadth,
        depth,
        beam_width,
        remaining_cap,
        eval_each_num_steps=engine.config.eval_each_num_steps,
    ) + root_evals
    engine.start_progress(total, label="beam_search")
    try:
        return await _run(
            engine,
            breadth=breadth,
            depth=depth,
            beam_width=beam_width,
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
    beam_width: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> SearchState | None:
    root = engine.create_seed_state()
    if evaluate_root:
        await engine.evaluate_many([root])

    beam: list[SearchState] = [root]
    max_depth = max(1, depth)
    for level in range(1, max_depth + 1):
        next_candidates: list[SearchState] = []
        should_evaluate = engine.should_evaluate_depth(level, max_depth)
        for parent in beam:
            remaining = None if max_evaluations is None else max_evaluations - engine.evaluation_count
            if should_evaluate and remaining is not None and remaining <= 0:
                break
            child_count = breadth if not should_evaluate or remaining is None else min(breadth, remaining)
            children = await engine.expand_state(
                parent,
                child_count,
                search_note=f"beam_search depth {level}: improve from a beam parent",
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
            next_candidates.extend(children)

        if not should_evaluate:
            if not next_candidates:
                break
            beam = next_candidates
            continue
        ranked = engine.ranked_states(next_candidates)
        if not ranked:
            break
        beam = ranked[: max(1, beam_width)]

    return engine.best_state()


def estimate_budget(
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
    eval_each_num_steps: int = 1,
) -> int:
    total = 0
    parents = 1
    max_depth = max(1, depth)
    interval = max(1, eval_each_num_steps)
    for level in range(1, max_depth + 1):
        generated = parents * max(1, breadth)
        if level % interval == 0 or level >= max_depth:
            total += generated
            parents = min(generated, max(1, beam_width))
        else:
            parents = generated
        if max_evaluations is not None and total >= max_evaluations:
            return max_evaluations
    return total
