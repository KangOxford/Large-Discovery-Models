"""Monte Carlo tree search over task-provided proposal states."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generic

from ldm_tts.optimization.search.protocols import SearchEngine, StateT


@dataclass
class MCTSNode(Generic[StateT]):
    state: StateT
    parent: MCTSNode[StateT] | None = None
    children: list[MCTSNode[StateT]] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value_mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


async def run(
    engine: SearchEngine[StateT],
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> StateT | None:
    """Run a lightweight UCT search through the shared engine interface."""

    root_evals = 1 if evaluate_root else 0
    candidate_total = max_evaluations if max_evaluations is not None else max(1, breadth) * max(1, depth)
    if max_evaluations is not None:
        candidate_total = max(0, max_evaluations - root_evals)
    engine.start_progress(candidate_total + root_evals, label="mcts")
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
    engine: SearchEngine[StateT],
    *,
    breadth: int,
    depth: int,
    beam_width: int,
    max_evaluations: int | None,
    evaluate_root: bool,
) -> StateT | None:
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
        leaf = _select(
            root,
            max_depth=max(1, depth),
            max_children=max(1, breadth),
            exploration=exploration,
        )
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
                    f"evaluation_interval={engine.evaluation_interval}."
                ),
            )

    return engine.best_state()


def _select(
    root: MCTSNode[StateT],
    *,
    max_depth: int,
    max_children: int,
    exploration: float,
) -> MCTSNode[StateT]:
    node = root
    while node.state.depth < max_depth and len(node.children) >= max_children and node.children:
        node = max(node.children, key=lambda child: _uct(child, exploration))
    return node


async def _expand_one(
    engine: SearchEngine[StateT],
    node: MCTSNode[StateT],
    *,
    simulation: int,
) -> MCTSNode[StateT] | None:
    child_index = len(node.children) + 1
    children = await engine.expand_state(
        node.state,
        1,
        search_note=(
            f"mcts simulation {simulation}: expand child {child_index} from a selected node; "
            "try a distinct high-upside proposal"
        ),
    )
    if not children:
        return None
    child_node = MCTSNode(children[0], parent=node)
    node.children.append(child_node)
    return child_node


def _uct(node: MCTSNode[StateT], exploration: float) -> float:
    if node.visits == 0:
        return float("inf")
    parent_visits = node.parent.visits if node.parent is not None else node.visits
    confidence = exploration * math.sqrt(math.log(max(parent_visits, 1) + 1.0) / node.visits)
    return node.value_mean + confidence


def _backpropagate(node: MCTSNode[StateT], reward: float) -> None:
    current: MCTSNode[StateT] | None = node
    while current is not None:
        current.visits += 1
        current.value_sum += reward
        current = current.parent


def _reward(engine: SearchEngine[StateT], state: StateT) -> float:
    return float(engine.reward(state))
