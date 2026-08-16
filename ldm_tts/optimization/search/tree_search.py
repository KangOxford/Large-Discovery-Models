"""Full-tree proposal search."""

from __future__ import annotations

from ldm_tts.optimization.search.protocols import SearchEngine, StateT
from ldm_tts.optimization.search.traversal import tree_search as traverse


async def run(
    engine: SearchEngine[StateT],
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> StateT | None:
    """Expand a full breadth/depth tree and evaluate configured depths."""

    root_evals = 1 if evaluate_root else 0
    remaining_cap = None if max_evaluations is None else max(0, max_evaluations - root_evals)
    total = estimate_budget(
        breadth,
        depth,
        remaining_cap,
        eval_each_num_steps=engine.evaluation_interval,
    )
    engine.start_progress(total + root_evals, label="tree_search")
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


def estimate_budget(
    breadth: int,
    depth: int,
    max_evaluations: int | None,
    eval_each_num_steps: int = 1,
) -> int:
    """Estimate real evaluations performed by full-tree traversal."""

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
