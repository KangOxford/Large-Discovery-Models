"""Remote LDMEnv that shells out to the task-venv worker process.

Mirrors the ``LDMEnv`` reset/step surface so ``ldm_rl.bridge.generate`` can
drive a real-mode environment without importing the task's heavy GP runtime
into the Slime process.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class RemoteStep:
    observation: str
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class RemoteLDMEnv:
    def __init__(self, spec: Any, task_python: str, extra_env: dict[str, str] | None = None):
        full_env = os.environ.copy()
        # The worker runs the task venv (torch 2.13 / CUDA 13). Strip the Slime
        # env's CUDA/library overrides (LD_LIBRARY_PATH points at the Slime conda
        # lib and cudart_block, which would shadow libcudart.so.13 that the task
        # venv's torch needs; CUDA_HOME points at the Slime conda prefix).
        for key in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES"):
            full_env.pop(key, None)
        if extra_env:
            full_env.update(extra_env)
        self._proc = subprocess.Popen(
            [task_python, "-m", "ldm_rl.real_env_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=full_env,
        )
        self._send({"op": "init", "spec": spec.to_dict()})
        response = self._recv()
        if "error" in response:
            raise RuntimeError(f"real-env worker init failed: {response['error']}")
        self._observation = response["observation"]

    def reset(self) -> str:
        return self._observation

    def step(self, action_text: str) -> RemoteStep:
        self._send({"op": "step", "action": action_text})
        response = self._recv()
        if "error" in response:
            raise RuntimeError(f"real-env worker step failed: {response['error']}")
        return RemoteStep(
            observation=response.get("observation", ""),
            reward=float(response.get("reward", 0.0)),
            done=bool(response.get("done", False)),
            terminated=bool(response.get("terminated", False)),
            truncated=bool(response.get("truncated", False)),
            info=response.get("info") or {},
        )

    def close(self) -> None:
        try:
            self._send({"op": "exit"})
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass

    def _send(self, payload: dict) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("task-venv worker exited unexpectedly")
        return json.loads(line)
