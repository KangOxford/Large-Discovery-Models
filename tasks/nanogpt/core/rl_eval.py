"""Real nanoGPT evaluation: one config -> one measured ``val_bpb``.

This is the environment the RL policy is actually acting in. Evaluating a
proposal means materialising a runnable copy of ``scripts/train.py`` with the
proposed knobs substituted, running it for its full wall-clock ``TIME_BUDGET``
on one GPU, and reading the metric block it prints at the end.

Cost model, because it drives every other design choice here: one evaluation is
~``TIME_BUDGET`` seconds (plus ~30-50s of startup/compile/eval) and occupies a
whole H100. There is no cheaper mode -- the measured within-config std is
~0.0013 bpb, so repeats buy almost nothing, and an analytic surrogate for this
task has already been shown to invert the true ranking. Hence:

* **a cache**, keyed on :func:`rl_knobs.canonical_key`, so a config is never
  measured twice;
* **a GPU claim**, so N concurrent rollout workers each get their own device
  instead of piling onto GPU 0;
* **classified failures**, so an OOM, a divergence and a rejected config are
  distinguishable in the results rather than collapsing to "rc != 0".

The evaluator is deliberately free of any cluster scheduler. Parallelism comes
from running several rollout workers, each holding its own environment and
claiming its own GPU from a shared pool.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from tasks.nanogpt.core import rl_knobs as knobs_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK_ROOT = os.path.dirname(_HERE)
SCRIPTS_DIR = os.path.join(_TASK_ROOT, "scripts")

#: Metrics train.py prints after the "---" separator (lines 621-630).
METRIC_KEYS = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
)

#: Wall-clock slack over TIME_BUDGET before the run is declared hung. Startup
#: (tokenizer load, torch.compile, first-step warmup) plus the final val pass
#: measured 28-51s on an H100; 300s is generous but a hang is expensive.
DEFAULT_TIMEOUT_SLACK = 300.0

FAILURE_KINDS = (
    "rejected_by_trainer",  # train.py's own assert refused the config
    "oom",                  # CUDA OOM
    "loss_diverged",        # train.py printed FAIL (NaN or loss > 100)
    "no_metrics",           # exited 0 but printed no val_bpb
    "crashed",              # anything else non-zero
    "timed_out",
)


@dataclass
class RunOutcome:
    """Result of one real evaluation attempt."""

    canonical_key: str
    config: dict[str, Any]
    time_budget: float
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    failure_kind: str = ""
    error: str = ""
    wall_seconds: float = 0.0
    run_dir: str = ""
    gpu: str = ""
    cached: bool = False

    @property
    def val_bpb(self) -> float | None:
        value = self.metrics.get("val_bpb")
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# train.py materialisation
# ---------------------------------------------------------------------------

_ASSIGNMENT = re.compile(r"^([A-Z_][A-Z0-9_]*)(\s*)=\s*([^#\n]*)(#.*)?$")


def _render(value: Any) -> str:
    return f'"{value}"' if isinstance(value, str) else repr(value)


def patch_train_source(source: str, config: dict[str, Any]) -> str:
    """Rewrite each top-level ``KNOB = ...`` assignment, keeping comments.

    The task's knobs are plain module-level assignments (``train.py`` lines
    433-451), not CLI flags, so a config is applied by rewriting those lines.
    Only names present in ``config`` are touched, which is why the neighbouring
    ``ADAM_BETAS`` tuple on line 444 is left alone.

    Raises ``KeyError`` if a requested knob is not found as a top-level
    assignment -- that means the template moved and silently substituting
    nothing would produce a run that measures the wrong config.
    """

    out: list[str] = []
    seen: set[str] = set()
    for line in source.splitlines(keepends=True):
        match = _ASSIGNMENT.match(line)
        if match and match.group(1) in config:
            name = match.group(1)
            comment = (match.group(4) or "").rstrip()
            pad = " " if comment else ""
            out.append(f"{name} = {_render(config[name])}{pad}{comment}\n")
            seen.add(name)
        else:
            out.append(line)
    missing = set(config) - seen
    if missing:
        raise KeyError(
            "knobs not found as top-level assignments in train.py: "
            f"{sorted(missing)}"
        )
    return "".join(out)


def materialise_run(
    config: dict[str, Any],
    run_dir: str,
    *,
    time_budget: float | None = None,
    scripts_dir: str = SCRIPTS_DIR,
) -> str:
    """Write a runnable ``train.py`` plus its ``prepare.py`` into ``run_dir``.

    ``prepare.py`` is **copied, not symlinked**, because the wall-clock budget
    lives there (``TIME_BUDGET = 300`` at prepare.py:31) rather than in
    ``train.py``, which imports it. Copying is what makes a per-evaluation
    budget possible at all; it costs a few KB and does not duplicate the
    dataset, since ``prepare.py``'s ``CACHE_DIR`` still resolves from
    ``AUTORESEARCH_CACHE_DIR`` and stays shared.
    """

    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(scripts_dir, "train.py")) as handle:
        train_src = handle.read()
    applied = {name: config[name] for name in knobs_mod.KNOB_ORDER}
    target = os.path.join(run_dir, "train.py")
    with open(target, "w") as handle:
        handle.write(patch_train_source(train_src, applied))

    with open(os.path.join(scripts_dir, "prepare.py")) as handle:
        prepare_src = handle.read()
    if time_budget is not None:
        prepare_src = patch_train_source(
            prepare_src, {"TIME_BUDGET": int(time_budget)}
        )
    with open(os.path.join(run_dir, "prepare.py"), "w") as handle:
        handle.write(prepare_src)
    return target


# ---------------------------------------------------------------------------
# Output parsing and failure classification
# ---------------------------------------------------------------------------


def parse_metrics(log: str) -> dict[str, float]:
    """Extract train.py's final metric block."""

    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        match = re.search(rf"^{key}: *(-?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)\s*$",
                          log, re.M)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def classify_failure(log: str, returncode: int) -> tuple[str, str]:
    """Map a failed run onto a :data:`FAILURE_KINDS` value plus a short reason.

    Order matters: an OOM traceback also contains the word "Error", and a
    diverged run also exits non-zero.
    """

    if "assert TOTAL_BATCH_SIZE" in log or (
        "AssertionError" in log and "tokens_per_fwdbwd" in log
    ):
        return "rejected_by_trainer", "TOTAL_BATCH_SIZE not divisible by device batch tokens"
    if "OutOfMemoryError" in log or "CUDA out of memory" in log:
        return "oom", "CUDA out of memory"
    if re.search(r"^FAIL\s*$", log, re.M):
        return "loss_diverged", "train.py fast-fail: loss NaN or > 100"
    if "AssertionError" in log:
        line = next(
            (l.strip() for l in reversed(log.splitlines()) if "assert" in l), "assert"
        )
        return "rejected_by_trainer", f"trainer assertion: {line[:200]}"
    tail = "\n".join(log.strip().splitlines()[-4:])
    return "crashed", f"exit {returncode}: {tail[:400]}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ResultCache:
    """Append-only JSONL cache of :class:`RunOutcome` rows, flock-protected.

    Shared by every concurrent worker, so a config measured by one worker is
    never re-measured by another. Failures are cached too -- a config that OOMs
    or that the trainer rejects will do so again, and paying 20s to rediscover
    that is still 20s of a GPU. ``timed_out`` is the one exception: a timeout
    can be a transient node problem, so it is not treated as final.
    """

    RETRYABLE = frozenset({"timed_out"})

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._loaded_size = -1

    def _load(self) -> dict[str, dict]:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return self._mem
        if size == self._loaded_size:
            return self._mem
        rows: dict[str, dict] = {}
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("canonical_key")
                if not key:
                    continue
                if row.get("failure_kind") in self.RETRYABLE:
                    continue
                rows[key] = row
        self._mem = rows
        self._loaded_size = size
        return rows

    def get(self, key: str) -> RunOutcome | None:
        row = self._load().get(key)
        if row is None:
            return None
        fields = {k: v for k, v in row.items() if k in RunOutcome.__dataclass_fields__}
        outcome = RunOutcome(**fields)
        outcome.cached = True
        return outcome

    def put(self, outcome: RunOutcome) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(outcome.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)
        self._loaded_size = -1  # force reload on next read

    def successes(self) -> list[RunOutcome]:
        """Every measured config, for warm-starting a GP."""

        out = []
        for row in self._load().values():
            if row.get("ok"):
                fields = {
                    k: v for k, v in row.items() if k in RunOutcome.__dataclass_fields__
                }
                outcome = RunOutcome(**fields)
                outcome.cached = True
                out.append(outcome)
        return out


# ---------------------------------------------------------------------------
# GPU pool
# ---------------------------------------------------------------------------


class GpuPool:
    """Exclusive GPU assignment across concurrent evaluators via lock files.

    Why this exists: ``ldm_rl.remote_env.RemoteLDMEnv`` deliberately strips
    ``CUDA_VISIBLE_DEVICES`` from the worker's environment (the task venv's
    torch must not inherit the trainer's CUDA setup), so an evaluator cannot
    learn its device from the usual channel. Pass the pool explicitly instead,
    via ``NANOGPT_EVAL_GPUS`` or the episode's ``real_kwargs``.

    Set the pool to the devices the *trainer is not using*: e.g. actor+sglang on
    0-3, ``NANOGPT_EVAL_GPUS=4,5,6,7``.
    """

    def __init__(self, devices: list[str], lock_dir: str):
        self.devices = devices
        self.lock_dir = lock_dir
        os.makedirs(lock_dir, exist_ok=True)

    @classmethod
    def from_env(cls, lock_dir: str, devices: str | None = None) -> "GpuPool":
        raw = devices if devices is not None else os.environ.get("NANOGPT_EVAL_GPUS", "")
        items = [item.strip() for item in str(raw).split(",") if item.strip()]
        if not items:
            items = ["0"]
        return cls(items, lock_dir)

    @contextmanager
    def claim(self, timeout: float = 3600.0, poll: float = 5.0) -> Iterator[str]:
        """Block until one device is free, then hold it for the duration."""

        deadline = time.time() + timeout
        while True:
            for device in self.devices:
                path = os.path.join(self.lock_dir, f"gpu{device}.lock")
                try:
                    handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                except FileExistsError:
                    if self._stale(path):
                        os.unlink(path)
                    continue
                try:
                    os.write(handle, f"{os.getpid()} {time.time():.0f}\n".encode())
                    os.close(handle)
                    yield device
                    return
                finally:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            if time.time() >= deadline:
                raise TimeoutError(
                    f"no evaluation GPU free after {timeout:.0f}s "
                    f"(pool={self.devices})"
                )
            time.sleep(poll)

    @staticmethod
    def _stale(path: str, max_age: float = 7200.0) -> bool:
        """A lock whose writer died. Ages out rather than checking liveness,
        because the holder may be in another container/namespace."""

        try:
            return time.time() - os.path.getmtime(path) > max_age
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def default_run_command(repo_root: str) -> list[str]:
    """The invocation verified against this repo's train dependency group.

    Real training is an opt-in dependency group (torch 2.9.1+cu128), so it is
    run through ``uv`` against the task project rather than whatever python
    happens to be active.
    """

    return [
        "uv", "run", "--group", "train",
        "--project", os.path.join(repo_root, "tasks", "nanogpt"),
        "python", "-u", "train.py",
    ]


class RealNanogptEvaluator:
    """Measure ``val_bpb`` by running the real trainer.

    Parameters mirror what an episode's ``real_kwargs`` can set, so the whole
    evaluator is configurable from the episode data file.
    """

    def __init__(
        self,
        *,
        output_dir: str,
        repo_root: str | None = None,
        time_budget: float = knobs_mod.DEFAULT_TIME_BUDGET,
        cache_file: str | None = None,
        eval_gpus: str | None = None,
        run_command: list[str] | None = None,
        timeout_slack: float = DEFAULT_TIMEOUT_SLACK,
        gpu_claim_timeout: float = 3600.0,
        autoresearch_cache_dir: str | None = None,
        keep_run_dirs: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.repo_root = os.path.abspath(
            repo_root or os.path.dirname(os.path.dirname(_TASK_ROOT))
        )
        self.time_budget = float(time_budget)
        self.runs_dir = os.path.join(self.output_dir, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)
        self.cache = ResultCache(
            cache_file or os.path.join(self.output_dir, "eval_cache.jsonl")
        )
        self.gpus = GpuPool.from_env(
            os.path.join(self.output_dir, "gpu_locks"), eval_gpus
        )
        self.run_command = run_command or default_run_command(self.repo_root)
        self.timeout_slack = float(timeout_slack)
        self.gpu_claim_timeout = float(gpu_claim_timeout)
        self.autoresearch_cache_dir = autoresearch_cache_dir
        self.keep_run_dirs = bool(keep_run_dirs)
        self.extra_env = dict(extra_env or {})

    # -- public API --------------------------------------------------------

    def evaluate_config(
        self, config: dict[str, Any], time_budget: float | None = None
    ) -> RunOutcome:
        """Measure one config. Returns a cached outcome when one exists."""

        budget = float(self.time_budget if time_budget is None else time_budget)
        clean = knobs_mod.validate(config)
        key = knobs_mod.canonical_key(clean, budget)

        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            knobs_mod.check_hard_constraints(clean)
        except knobs_mod.ProposalError as exc:
            outcome = RunOutcome(
                canonical_key=key,
                config=clean,
                time_budget=budget,
                ok=False,
                failure_kind="rejected_by_trainer",
                error=str(exc),
            )
            self.cache.put(outcome)
            return outcome

        outcome = self._run(key, clean, budget)
        self.cache.put(outcome)
        return outcome

    def measured_history(self) -> list[tuple[dict[str, Any], float]]:
        """``(config, val_bpb)`` for every success so far -- GP warm start."""

        rows = []
        for outcome in self.cache.successes():
            value = outcome.val_bpb
            if value is not None:
                rows.append((outcome.config, float(value)))
        return rows

    # -- internals ---------------------------------------------------------

    def _run(self, key: str, config: dict[str, Any], budget: float) -> RunOutcome:
        # Stable across processes: str.__hash__ is salted by PYTHONHASHSEED, so
        # the same config would otherwise land in a different directory in every
        # worker, which makes a failed run impossible to find from its key.
        tag = "r" + hashlib.sha256(key.encode()).hexdigest()[:16]
        run_dir = os.path.join(self.runs_dir, tag)
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
        materialise_run(config, run_dir, time_budget=budget,
                        scripts_dir=os.path.join(_TASK_ROOT, "scripts"))
        with open(os.path.join(run_dir, "config.json"), "w") as handle:
            json.dump({"canonical_key": key, "config": config,
                       "time_budget": budget}, handle, indent=1, sort_keys=True)

        timeout = budget + self.timeout_slack
        started = time.time()
        try:
            with self.gpus.claim(timeout=self.gpu_claim_timeout) as device:
                completed = self._spawn(run_dir, device, timeout)
                log = completed["log"]
                returncode = completed["returncode"]
                timed_out = completed["timed_out"]
                gpu = device
        except TimeoutError as exc:
            return RunOutcome(
                canonical_key=key, config=config, time_budget=budget, ok=False,
                failure_kind="timed_out", error=str(exc),
                wall_seconds=time.time() - started, run_dir=run_dir,
            )

        wall = time.time() - started
        with open(os.path.join(run_dir, "stdout.log"), "w") as handle:
            handle.write(log)

        if timed_out:
            outcome = RunOutcome(
                canonical_key=key, config=config, time_budget=budget, ok=False,
                failure_kind="timed_out",
                error=f"no exit after {timeout:.0f}s (budget {budget:.0f}s)",
                wall_seconds=wall, run_dir=run_dir, gpu=gpu,
            )
        else:
            metrics = parse_metrics(log)
            if returncode == 0 and "val_bpb" in metrics:
                outcome = RunOutcome(
                    canonical_key=key, config=config, time_budget=budget, ok=True,
                    metrics=metrics, wall_seconds=wall, run_dir=run_dir, gpu=gpu,
                )
            elif returncode == 0:
                outcome = RunOutcome(
                    canonical_key=key, config=config, time_budget=budget, ok=False,
                    metrics=metrics, failure_kind="no_metrics",
                    error="exited 0 without printing val_bpb",
                    wall_seconds=wall, run_dir=run_dir, gpu=gpu,
                )
            else:
                kind, reason = classify_failure(log, returncode)
                outcome = RunOutcome(
                    canonical_key=key, config=config, time_budget=budget, ok=False,
                    metrics=metrics, failure_kind=kind, error=reason,
                    wall_seconds=wall, run_dir=run_dir, gpu=gpu,
                )

        if not self.keep_run_dirs and outcome.ok:
            shutil.rmtree(run_dir, ignore_errors=True)
        return outcome

    def _spawn(self, run_dir: str, device: str, timeout: float) -> dict[str, Any]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        # Two configs in the reference buffer OOMed only ~400-580MB short, i.e.
        # allocator fragmentation rather than a genuinely infeasible config.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.autoresearch_cache_dir:
            env["AUTORESEARCH_CACHE_DIR"] = self.autoresearch_cache_dir
        env.update(self.extra_env)

        try:
            proc = subprocess.run(
                self.run_command,
                cwd=run_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            return {"log": proc.stdout or "", "returncode": proc.returncode,
                    "timed_out": False}
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", "replace")
            return {"log": partial, "returncode": -1, "timed_out": True}
        except FileNotFoundError as exc:
            return {"log": f"could not launch {self.run_command[0]!r}: {exc}",
                    "returncode": 127, "timed_out": False}


__all__ = [
    "DEFAULT_TIMEOUT_SLACK",
    "FAILURE_KINDS",
    "METRIC_KEYS",
    "GpuPool",
    "RealNanogptEvaluator",
    "ResultCache",
    "RunOutcome",
    "classify_failure",
    "default_run_command",
    "materialise_run",
    "parse_metrics",
    "patch_train_source",
]
