"""Run a real-mode LDMEnv inside the task venv; JSON-lines over stdio.

Protocol (one JSON object per line, both directions):

    -> {"op": "init", "spec": <EpisodeSpec dict>}
    <- {"observation": "<rendered prompt>"}

    -> {"op": "step", "action": "<policy text>"}
    <- {"observation": ..., "reward": ..., "done": ..., "terminated": ...,
        "truncated": ..., "info": {...}}

    -> {"op": "exit"}

Errors are returned as {"error": "..."} without terminating the process.
"""

from __future__ import annotations

import json
import sys


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()


def main() -> int:
    from ldm_rl.episodes import EpisodeSpec
    from ldm_rl.factories import build_env

    env = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"error": f"invalid json: {exc}"})
            continue
        op = request.get("op")
        try:
            if op == "init":
                spec = EpisodeSpec.from_dict(request["spec"])
                env = build_env(
                    spec.task,
                    mode="real",
                    config=spec.to_env_config(),
                    context=spec.context,
                    seed=spec.seed,
                    **spec.real,
                )
                _emit({"observation": env.reset()})
            elif op == "step":
                if env is None:
                    raise RuntimeError("worker not initialized")
                step = env.step(request["action"])
                _emit(
                    {
                        "observation": step.observation,
                        "reward": step.reward,
                        "done": step.done,
                        "terminated": step.terminated,
                        "truncated": step.truncated,
                        "info": step.info,
                    }
                )
            elif op == "exit":
                break
            else:
                _emit({"error": f"unknown op: {op}"})
        except Exception as exc:  # noqa: BLE001 - keep the worker alive
            _emit({"error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
