"""Remote LDMEnv that shells out to the task-venv worker process.

Mirrors the ``LDMEnv`` reset/step surface so ``ldm_rl.bridge.generate`` can
drive a real-mode environment without importing the task's heavy GP runtime
into the Slime process.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
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
        # Keep the worker's stderr instead of discarding it: it runs in a
        # different interpreter, so its traceback is the only evidence of why
        # it died. A temp file rather than a pipe, because nothing drains a
        # stderr pipe during a campaign and a full pipe buffer would deadlock
        # the worker.
        self._stderr_file = tempfile.NamedTemporaryFile(
            prefix="ldm_rl_worker_", suffix=".stderr", mode="w+", delete=False
        )
        self._stderr_path = self._stderr_file.name
        self._proc = subprocess.Popen(
            [task_python, "-m", "ldm_rl.real_env_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
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
        # One env per episode, so leaking the stderr file would leave one per
        # episode behind across a full run matrix.
        try:
            self._stderr_file.close()
        except Exception:
            pass
        if self._stderr_path:
            try:
                os.unlink(self._stderr_path)
            except OSError:
                pass
            self._stderr_path = None

    def _send(self, payload: dict) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(
                "task-venv worker exited unexpectedly"
                + self._stderr_tail()
            )
        return json.loads(line)

    def _stderr_tail(self, n: int = 20) -> str:
        """Last ``n`` lines the worker wrote to stderr, for the exit message.

        The worker is a separate interpreter, so its traceback is the only
        evidence of why it died. Without this the caller sees a bare "exited
        unexpectedly" and has to reproduce the spawn by hand to learn that it
        was, say, a ModuleNotFoundError from a PYTHONPATH that did not reach
        the child.
        """
        if self._stderr_path is None:
            return ""
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass
        try:
            with open(self._stderr_path, "r", errors="replace") as fh:
                tail = fh.read().splitlines()[-n:]
        except OSError:
            return ""
        if not tail:
            return f" (rc={self._proc.returncode}, worker wrote nothing to stderr)"
        body = "\n  ".join(tail)
        return f" (rc={self._proc.returncode}); worker stderr tail:\n  {body}"
