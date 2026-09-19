"""Run a small-molecule RL episode (mock) with the task's own mock proposer.

Demonstrates the LDM RL loop end to end:

    observation -> policy (ExpandingMockCase2LLM) -> candidate proposals
    -> env.step (admit / dedup / evaluate with the task's mock scorers)
    -> improvement reward -> repeat until the budget is exhausted.

Usage:

    python rl/ldm_rl/examples/small_molecule_rl_case.py [iterations]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``ldm_rl`` (rl/) and the repo root (ldm_tts, tasks) importable when run
# as a plain script from the checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RL_ROOT = Path(__file__).resolve().parents[2]
for _path in (_RL_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ldm_rl import EnvConfig  # noqa: E402
from ldm_rl.factories import build_env  # noqa: E402
from tasks.small_molecule.core.workflow import ExpandingMockCase2LLM  # noqa: E402


def main() -> int:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    env = build_env(
        "small_molecule",
        mode="mock",
        config=EnvConfig(
            iterations=iterations,
            reservoir_size=3,
            evaluations_per_round=1,
            reward="improvement",
        ),
    )
    llm = ExpandingMockCase2LLM()

    observation = env.reset()
    trajectory: list[dict] = []
    for round_idx in range(iterations):
        action = llm.chat("system", observation, json_mode=True)
        step = env.step(action)
        row: dict = {
            "round": round_idx,
            "reward": round(step.reward, 6),
            "terminated": step.terminated,
            "truncated": step.truncated,
        }
        evaluated = step.info["evaluated"]
        if evaluated:
            metrics = evaluated[0]["evaluation"]["metrics"]
            row.update(
                {
                    "smiles": evaluated[0]["candidate"]["payload"]["smiles"],
                    "vina": metrics["vina"],
                    "activity": metrics["activity"],
                }
            )
        else:
            row["rejections"] = [item["reason"] for item in step.info["rejections"]]
        trajectory.append(row)
        print(f"round={round_idx} " + json.dumps(row, sort_keys=True))
        observation = step.observation
        if step.done:
            break

    scored = [row for row in trajectory if "vina" in row]
    summary = {
        "status": "ok",
        "task": "small_molecule",
        "mode": "mock",
        "rounds": len(trajectory),
        "total_reward": round(sum(row["reward"] for row in trajectory), 6),
        "best_vina": min(row["vina"] for row in scored),
        "best_activity": max(row["activity"] for row in scored),
        "final_smiles": scored[-1]["smiles"] if scored else None,
    }
    # Single trailing JSON object; the delta-sandbox CLI extracts it as the
    # result summary from stdout.
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
