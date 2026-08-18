"""Serializable episode specifications for the LDM RL environment.

Each Slime rollout sample is one LDM campaign episode. The sample prompt is a
compact JSON episode spec; ``bridge.generate`` parses it, builds the
environment and renders the real policy prompt from ``env.reset()``. Keeping
the spec declarative lets a static prompt-data file drive many tasks and
modes without a custom Slime data source.
"""

from __future__ import annotations

import argparse
import json
import sys

if __package__ in (None, ""):
    # Support `python rl/ldm_rl/episodes.py` from a checkout without the
    # package installed: put <repo>/rl and <repo> on sys.path first.
    import os

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _rl_root = os.path.dirname(_script_dir)
    _repo_root = os.path.dirname(_rl_root)
    for _path in (_rl_root, _repo_root):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from dataclasses import asdict, dataclass, field
from typing import Any

from ldm_rl.env import EnvConfig, REWARD_POLICIES


@dataclass(frozen=True)
class EpisodeSpec:
    """One RL episode: a bounded LDM campaign over a registered task."""

    task: str
    mode: str = "mock"
    iterations: int = 8
    reservoir_size: int = 2
    evaluations_per_round: int = 1
    reward: str = "improvement"
    reward_failure: float = 0.0
    reward_invalid: float = 0.0
    max_empty_reservoir_rounds: int = 3
    seed: int = 0
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("episode task must not be empty")
        if self.mode not in {"mock", "real"}:
            raise ValueError("episode mode must be 'mock' or 'real'")
        if self.reward not in REWARD_POLICIES:
            raise ValueError(
                f"unknown reward policy {self.reward!r}; expected one of {REWARD_POLICIES}"
            )
        EnvConfig(  # validate shared bounds eagerly
            iterations=self.iterations,
            reservoir_size=self.reservoir_size,
            evaluations_per_round=self.evaluations_per_round,
            max_empty_reservoir_rounds=self.max_empty_reservoir_rounds,
            reward=self.reward,
            reward_failure=self.reward_failure,
            reward_invalid=self.reward_invalid,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_env_config(self) -> EnvConfig:
        return EnvConfig(
            iterations=self.iterations,
            reservoir_size=self.reservoir_size,
            evaluations_per_round=self.evaluations_per_round,
            max_empty_reservoir_rounds=self.max_empty_reservoir_rounds,
            reward=self.reward,
            reward_failure=self.reward_failure,
            reward_invalid=self.reward_invalid,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeSpec":
        unknown = set(payload) - {
            "task",
            "mode",
            "iterations",
            "reservoir_size",
            "evaluations_per_round",
            "reward",
            "reward_failure",
            "reward_invalid",
            "max_empty_reservoir_rounds",
            "seed",
            "context",
        }
        if unknown:
            raise ValueError("unknown episode field(s): " + ", ".join(sorted(unknown)))
        if "task" not in payload:
            raise ValueError("episode spec is missing the task field")
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str) -> "EpisodeSpec":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"episode prompt is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("episode prompt must be a JSON object")
        return cls.from_dict(payload)


def make_prompt_rows(specs: list[EpisodeSpec]) -> list[dict[str, str]]:
    """Render Slime prompt-data rows (``prompt`` + empty ``label`` columns)."""

    return [{"prompt": spec.to_json(), "label": ""} for spec in specs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write Slime prompt-data JSONL with LDM episode specs."
    )
    parser.add_argument("--output", required=True, help="JSONL path to write")
    parser.add_argument("--task", required=True, help="registered LDM task id")
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--reservoir-size", type=int, default=2)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument(
        "--reward", choices=REWARD_POLICIES, default="improvement"
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")

    rows = make_prompt_rows(
        [
            EpisodeSpec(
                task=args.task,
                mode=args.mode,
                iterations=args.iterations,
                reservoir_size=args.reservoir_size,
                evaluations_per_round=args.evaluations_per_round,
                reward=args.reward,
                seed=args.seed_offset + index,
            )
            for index in range(args.count)
        ]
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} episode row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
