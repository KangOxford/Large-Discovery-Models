from __future__ import annotations

import math
from dataclasses import dataclass, field

from nanogpt.ldm_task.search_core import SearchEngine, SearchState


@dataclass
class MCTSNode:
    state: SearchState
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value_mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


async def run(
    engine: SearchEngine,
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> SearchState | None:
    """Run a lightweight UCT search over generated train.py states."""
    root_evals = 1 if evaluate_root else 0
    candidate_total = max_evaluations if max_evaluations is not None else max(1, breadth) * max(1, depth)
    if max_evaluations is not None:
        candidate_total = max(0, max_evaluations - root_evals)
    total = candidate_total + root_evals
    engine.start_progress(total, label="mcts")
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
    root_state = engine.create_seed_state()
    if evaluate_root:
        await engine.evaluate_many([root_state])

    root = MCTSNode(root_state)
    if root_state.score is not None:
        _backpropagate(root, _reward(engine, root_state))

    budget = max_evaluations if max_evaluations is not None else max(1, breadth) * max(1, depth)
    budget = max(0, budget - engine.evaluation_count)
    exploration = max(0.1, float(beam_width))

    for simulation in range(budget):
        leaf = _select(root, max_depth=max(1, depth), max_children=max(1, breadth), exploration=exploration)
        if leaf.state.depth >= max(1, depth):
            if leaf.state.score is not None:
                _backpropagate(leaf, _reward(engine, leaf.state))
            continue

        child = await _expand_one(engine, leaf, simulation=simulation + 1)
        if child is None:
            continue
        if engine.should_evaluate_depth(child.state.depth, max(1, depth)):
            await engine.evaluate_many([child.state])
            _backpropagate(child, _reward(engine, child.state))
        else:
            await engine.defer_evaluation_many(
                [child.state],
                reason=(
                    f"Evaluation deferred at depth {child.state.depth}; "
                    f"eval_each_num_steps={engine.config.eval_each_num_steps}."
                ),
            )

    return engine.best_state()


def _select(root: MCTSNode, *, max_depth: int, max_children: int, exploration: float) -> MCTSNode:
    node = root
    while node.state.depth < max_depth and len(node.children) >= max_children and node.children:
        node = max(node.children, key=lambda child: _uct(child, exploration))
    return node


async def _expand_one(engine: SearchEngine, node: MCTSNode, *, simulation: int) -> MCTSNode | None:
    child_index = len(node.children) + 1
    children = await engine.expand_state(
        node.state,
        1,
        search_note=(
            f"mcts simulation {simulation}: expand child {child_index} from a selected node. "
            "Try a distinct, high-upside edit while preserving train.py evaluation output."
        ),
    )
    if not children:
        return None
    child_node = MCTSNode(children[0], parent=node)
    node.children.append(child_node)
    return child_node


def _uct(node: MCTSNode, exploration: float) -> float:
    if node.visits == 0:
        return float("inf")
    parent_visits = node.parent.visits if node.parent is not None else node.visits
    confidence = exploration * math.sqrt(math.log(max(parent_visits, 1) + 1.0) / node.visits)
    return node.value_mean + confidence


def _backpropagate(node: MCTSNode, reward: float) -> None:
    while node is not None:
        node.visits += 1
        node.value_sum += reward
        node = node.parent


def _reward(engine: SearchEngine, state: SearchState) -> float:
    if state.score is None or not math.isfinite(float(state.score)):
        return 0.0
    if state.score >= engine.config.failure_score:
        return 0.0

    scored = [
        float(candidate.score)
        for candidate in engine.states
        if candidate.score is not None
        and math.isfinite(float(candidate.score))
        and float(candidate.score) < engine.config.failure_score
    ]
    if not scored:
        return 0.0
    worst = max(scored)
    best = min(scored)
    if worst <= best:
        return 1.0
    score = float(state.score)
    if engine.config.minimize:
        return (worst - score) / (worst - best)
    return (score - best) / (worst - best)
