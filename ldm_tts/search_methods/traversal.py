"""Reusable proposal-state traversal kernels."""

from __future__ import annotations

from ldm_tts.search_methods.protocols import SearchEngine, SearchTraversalResult, StateT


async def single_turn(
    engine: SearchEngine[StateT],
    root: StateT,
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
) -> SearchTraversalResult[StateT]:
    """Expand and score one proposal batch from the seed state."""

    del depth, beam_width
    remaining = _remaining_evaluations(engine, max_evaluations)
    if remaining is not None and remaining <= 0:
        return SearchTraversalResult((), ())
    child_count = max(1, breadth) if remaining is None else min(max(1, breadth), remaining)
    children = await engine.expand_state(
        root,
        child_count,
        search_note="single_turn: propose one candidate batch from the seed state",
    )
    await engine.evaluate_many(children)
    states = tuple(children)
    return SearchTraversalResult(states, states)


async def best_of_n(
    engine: SearchEngine[StateT],
    root: StateT,
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
) -> SearchTraversalResult[StateT]:
    """Run independent proposal branches from a common seed state."""

    del beam_width
    generated: list[StateT] = []
    leaves: list[StateT] = []
    branch_count = max(1, breadth)
    branch_depth = max(1, depth)
    stop = False
    for branch_index in range(1, branch_count + 1):
        parent = root
        last_child: StateT | None = None
        for level in range(1, branch_depth + 1):
            remaining = _remaining_evaluations(engine, max_evaluations)
            should_evaluate = engine.should_evaluate_depth(level, branch_depth)
            if should_evaluate and remaining is not None and remaining <= 0:
                stop = True
                break
            children = await engine.expand_state(
                parent,
                1,
                search_note=(
                    f"best_of_n branch {branch_index}/{branch_count}, "
                    f"step {level}/{branch_depth}: continue this independent rollout branch"
                ),
            )
            if not children:
                break
            child = children[0]
            generated.append(child)
            if should_evaluate:
                await engine.evaluate_many([child])
            else:
                await _defer(engine, [child], level)
            parent = child
            last_child = child
        if last_child is not None:
            leaves.append(last_child)
        if stop:
            break
    return SearchTraversalResult(tuple(generated), tuple(leaves))


async def tree_search(
    engine: SearchEngine[StateT],
    root: StateT,
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
) -> SearchTraversalResult[StateT]:
    """Expand the full proposal tree to the configured depth."""

    del beam_width
    frontier: list[StateT] = [root]
    generated: list[StateT] = []
    leaves: list[StateT] = []
    max_depth = max(1, depth)
    branch_count = max(1, breadth)
    for level in range(1, max_depth + 1):
        next_frontier: list[StateT] = []
        should_evaluate = engine.should_evaluate_depth(level, max_depth)
        for parent in frontier:
            remaining = _remaining_evaluations(engine, max_evaluations)
            if should_evaluate and remaining is not None and remaining <= 0:
                break
            child_count = branch_count if not should_evaluate or remaining is None else min(branch_count, remaining)
            children = await engine.expand_state(
                parent,
                child_count,
                search_note=f"tree_search depth {level}/{max_depth}: expand this search-tree node",
            )
            if should_evaluate:
                await engine.evaluate_many(children)
            else:
                await _defer(engine, children, level)
            generated.extend(children)
            next_frontier.extend(children)
        if not next_frontier:
            break
        leaves = next_frontier
        frontier = next_frontier
    return SearchTraversalResult(tuple(generated), tuple(leaves))


async def beam_search(
    engine: SearchEngine[StateT],
    root: StateT,
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
) -> SearchTraversalResult[StateT]:
    """Expand proposal states and retain the best-scoring beam at each scored depth."""

    beam: list[StateT] = [root]
    generated: list[StateT] = []
    leaves: list[StateT] = []
    max_depth = max(1, depth)
    branch_count = max(1, breadth)
    keep_count = max(1, beam_width)
    for level in range(1, max_depth + 1):
        next_candidates: list[StateT] = []
        should_evaluate = engine.should_evaluate_depth(level, max_depth)
        for parent in beam:
            remaining = _remaining_evaluations(engine, max_evaluations)
            if should_evaluate and remaining is not None and remaining <= 0:
                break
            child_count = branch_count if not should_evaluate or remaining is None else min(branch_count, remaining)
            children = await engine.expand_state(
                parent,
                child_count,
                search_note=f"beam_search depth {level}/{max_depth}: expand this beam parent",
            )
            if should_evaluate:
                await engine.evaluate_many(children)
            else:
                await _defer(engine, children, level)
            generated.extend(children)
            next_candidates.extend(children)

        if not next_candidates:
            break
        leaves = next_candidates
        if should_evaluate:
            ranked = engine.ranked_states(next_candidates)
            if not ranked:
                break
            beam = ranked[:keep_count]
        else:
            beam = next_candidates
    return SearchTraversalResult(tuple(generated), tuple(leaves))


def estimate_tree_generated(breadth: int, depth: int) -> int:
    """Return the number of states generated by a full tree traversal."""

    total = 0
    parents = 1
    for _level in range(max(1, int(depth))):
        parents *= max(1, int(breadth))
        total += parents
    return total


def estimate_beam_generated(breadth: int, depth: int, beam_width: int) -> int:
    """Return the number of states generated by a beam traversal."""

    total = 0
    parents = 1
    branch_count = max(1, int(breadth))
    keep_count = max(1, int(beam_width) if int(beam_width) > 0 else branch_count)
    for _level in range(max(1, int(depth))):
        generated = parents * branch_count
        total += generated
        parents = min(generated, keep_count)
    return total


async def _defer(engine: SearchEngine[StateT], states: list[StateT], depth: int) -> None:
    await engine.defer_evaluation_many(
        states,
        reason=(
            f"Evaluation deferred at depth {depth}; "
            f"evaluation_interval={engine.evaluation_interval}."
        ),
    )


def _remaining_evaluations(
    engine: SearchEngine[StateT],
    max_evaluations: int | None,
) -> int | None:
    if max_evaluations is None:
        return None
    return max(0, int(max_evaluations) - int(engine.evaluation_count))

