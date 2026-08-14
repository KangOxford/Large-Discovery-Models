"""Mock, isolated tensor-contract, and pinned MLS-Bench evaluators."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from ldm_tts.contracts import Candidate, EvaluationResult
from ldm_tts.engine.run_store import atomic_json_write

from tasks.llm_kv_adaptive_quantization.core.candidate import (
    validate_candidate_source,
)


TASK_PATH = "harbor/tasks/mls-bench__llm-kv-adaptive-quantization"
OFFICIAL_COMMIT = "cfd57a7e0139c72753e32e31bca593719b098717"
OFFICIAL_WORKLOADS = (
    "longbench_hotpotqa",
    "longbench_passage_retrieval",
    "longbench_repobench",
    "needlebench_niah",
    "gsm8k",
)
WORKLOAD_LABELS = {
    "longbench_hotpotqa": "longbench-hotpotqa",
    "longbench_passage_retrieval": "longbench-passage-retrieval",
    "longbench_repobench": "longbench-repobench",
    "needlebench_niah": "needlebench-niah",
    "gsm8k": "gsm8k",
}
WORKLOAD_TIMEOUTS = {
    "longbench_hotpotqa": 2 * 3600,
    "longbench_passage_retrieval": 2 * 3600,
    "longbench_repobench": 3 * 3600,
    "needlebench_niah": 2 * 3600,
    "gsm8k": 9 * 3600,
}
OFFICIAL_HARNESS_SHA256 = (
    "6129821b9356b2a8ff00938b9153e7d8bc21f6d03e5930ffe9701655bb7760d9"
)
OFFICIAL_FIXED_HARNESS_SHA256 = (
    "c922c235c1caa09c6cc4be01faadb28fd12cba35852967972257c46dcdbe91dc"
)
METRIC_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)


def qualification_selection_score(final_score: float, compression: float) -> float:
    """Explicit tiny-run signal; not the official five-workload score."""

    quality = min(max(float(final_score) / 100.0, 0.0), 1.0)
    efficiency = min(max(float(compression) / 8.0, 0.0), 1.0)
    return 0.6 * quality + 0.4 * efficiency


def candidate_code(candidate: Candidate) -> str:
    return str(candidate.payload["code"])


class MockQuantizerEvaluator:
    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        source = candidate_code(candidate)
        bit_matches = re.findall(r"self\.bits\s*=\s*min\((\d+)", source)
        bits = int(bit_matches[0]) if bit_matches else 4
        effective_bits = float(max(2, min(bits, 16)))
        compression = 16.0 / effective_bits
        final_score = 88.0 - max(0, 4 - bits) * 4.0
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={
                "selection_score": qualification_selection_score(
                    final_score, compression
                ),
                "final_score": final_score,
                "effective_kv_bits": effective_bits,
                "kv_compression_ratio": compression,
                "runtime_seconds": 0.0,
            },
            resource_usage={"benchmark_jobs": 1},
            metadata={"evaluator": "deterministic_mock"},
        )


class TensorContractEvaluator:
    """Run admitted source in a credential-free subprocess with a timeout."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        device: str = "cpu",
        python_executable: str | Path = sys.executable,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.device = str(device)
        self.python_executable = str(python_executable)

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        started = time.monotonic()
        try:
            source = candidate_code(candidate)
            validate_candidate_source(source)
            with tempfile.TemporaryDirectory(prefix="kv-contract-") as raw_dir:
                directory = Path(raw_dir)
                source_path = directory / "candidate.py"
                source_path.write_text(source, encoding="utf-8")
                completed = subprocess.run(
                    [
                        self.python_executable,
                        "-I",
                        str(Path(__file__).with_name("contract_worker.py")),
                        str(source_path),
                        self.device,
                    ],
                    cwd=directory,
                    env=_isolated_environment(device=self.device),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            if completed.returncode != 0:
                raise ValueError(completed.stderr[-2000:] or "contract worker failed")
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            if payload.get("status") != "ok":
                raise ValueError("contract worker did not report success")
            error = float(payload["mean_absolute_error"])
            compression = float(payload["kv_compression_ratio"])
            synthetic_quality = max(0.0, 100.0 * (1.0 - error))
            return EvaluationResult(
                candidate.candidate_id,
                "succeeded",
                metrics={
                    "selection_score": qualification_selection_score(
                        synthetic_quality, compression
                    ),
                    "final_score": synthetic_quality,
                    "effective_kv_bits": float(payload["effective_kv_bits"]),
                    "kv_compression_ratio": compression,
                    "runtime_seconds": time.monotonic() - started,
                    "state_tensor_elements": float(payload["state_tensor_elements"]),
                },
                resource_usage={"benchmark_jobs": 0},
                metadata={"evaluator": "tensor_contract", "device": self.device},
            )
        except subprocess.TimeoutExpired as exc:
            return EvaluationResult(
                candidate.candidate_id,
                "timed_out",
                error=f"tensor contract exceeded {self.timeout_seconds:.1f}s: {exc}",
                metadata={"evaluator": "tensor_contract"},
            )
        except Exception as exc:
            return EvaluationResult(
                candidate.candidate_id,
                "invalid",
                error=str(exc),
                metadata={"evaluator": "tensor_contract"},
            )


class ContractThenMLSBenchEvaluator:
    def __init__(
        self,
        contract: TensorContractEvaluator,
        benchmark: "MLSBenchEvaluator",
    ) -> None:
        self.contract = contract
        self.benchmark = benchmark

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        contract_result = self.contract.evaluate(candidate)
        if not contract_result.succeeded:
            return contract_result
        return self.benchmark.evaluate(candidate)


class MLSBenchEvaluator:
    """Run official workload commands against one immutable package harness."""

    def __init__(
        self,
        *,
        package_dir: Path,
        upstream_root: Path,
        run_dir: Path,
        workloads: tuple[str, ...],
        devices: tuple[str, ...],
        model_id: str,
        max_examples: int,
        timeout_seconds: float,
        cpu: bool,
        evaluator_python: str | Path = sys.executable,
    ) -> None:
        self.package_dir = Path(package_dir).resolve()
        self.upstream_root = Path(upstream_root).resolve()
        self.run_dir = Path(run_dir)
        self.workloads = tuple(workloads)
        self.devices = tuple(devices)
        self.model_id = str(model_id)
        self.max_examples = int(max_examples)
        self.timeout_seconds = float(timeout_seconds)
        self.cpu = bool(cpu)
        self.evaluator_python = str(evaluator_python)

    @property
    def task_dir(self) -> Path:
        return self.upstream_root / TASK_PATH

    @property
    def harness_file(self) -> Path:
        return self.package_dir / "custom_quant_eval.py"

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        evaluation_dir = self.run_dir / "evaluations" / candidate.candidate_id
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "source_commit": OFFICIAL_COMMIT,
            "workloads": list(self.workloads),
            "jobs": [],
            "status": "running",
        }
        manifest_path = evaluation_dir / "evaluation_manifest.json"
        atomic_json_write(manifest_path, manifest)
        try:
            harness = self.harness_file.read_text(encoding="utf-8")
            if fixed_harness_sha256(harness) != OFFICIAL_FIXED_HARNESS_SHA256:
                raise ValueError(
                    "package harness fixed region does not match the pinned MLS-Bench source"
                )
            harness_copy = evaluation_dir / "custom_quant_eval.py"
            harness_copy.write_text(
                replace_quantizer_class(harness, candidate_code(candidate)),
                encoding="utf-8",
            )
            jobs = self._run_jobs(harness_copy, evaluation_dir)
            manifest["jobs"] = jobs
            failed = [item for item in jobs if item["status"] != "succeeded"]
            if failed:
                manifest["status"] = "failed"
                atomic_json_write(manifest_path, manifest)
                return EvaluationResult(
                    candidate.candidate_id,
                    "failed",
                    error="; ".join(
                        f"{item['workload']}: {item.get('error', item['status'])}"
                        for item in failed
                    ),
                    artifacts={"evaluation_manifest": str(manifest_path)},
                    resource_usage={"benchmark_jobs": len(jobs)},
                )
            flattened = _flatten_official_metrics(jobs)
            selection_score = _mean(
                qualification_selection_score(
                    item["metrics"]["final_score"],
                    item["metrics"]["kv_compression_ratio"],
                )
                for item in jobs
            )
            metrics: dict[str, float] = {
                **flattened,
                "selection_score": selection_score,
                "effective_kv_bits": _mean(
                    item["metrics"]["effective_kv_bits"] for item in jobs
                ),
                "kv_compression_ratio": _mean(
                    item["metrics"]["kv_compression_ratio"] for item in jobs
                ),
                "runtime_seconds": time.monotonic() - started,
            }
            if set(self.workloads) == set(OFFICIAL_WORKLOADS):
                metrics["official_score"] = compute_official_score(
                    self.task_dir, flattened
                )
                metrics["selection_score"] = metrics["official_score"]
            manifest["status"] = "succeeded"
            manifest["metrics"] = metrics
            atomic_json_write(manifest_path, manifest)
            return EvaluationResult(
                candidate.candidate_id,
                "succeeded",
                metrics=metrics,
                artifacts={
                    "evaluation_manifest": str(manifest_path),
                    "harness": str(harness_copy),
                },
                resource_usage={
                    "wall_seconds": time.monotonic() - started,
                    "benchmark_jobs": len(jobs),
                },
                metadata={
                    "workloads": list(self.workloads),
                    "model_id": self.model_id,
                    "source_commit": OFFICIAL_COMMIT,
                },
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            atomic_json_write(manifest_path, manifest)
            return EvaluationResult(
                candidate.candidate_id,
                "failed",
                error=str(exc),
                artifacts={"evaluation_manifest": str(manifest_path)},
                resource_usage={"benchmark_jobs": len(manifest["jobs"])},
            )

    def _run_jobs(self, harness: Path, evaluation_dir: Path) -> list[dict[str, Any]]:
        workers = 1 if self.cpu else min(len(self.workloads), len(self.devices))
        if not self.cpu and workers < len(self.workloads):
            raise ValueError("real evaluation requires one configured device per workload")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._run_job,
                    harness,
                    evaluation_dir,
                    workload,
                    "" if self.cpu else self.devices[index],
                ): index
                for index, workload in enumerate(self.workloads)
            }
            indexed = {
                futures[future]: future.result()
                for future in as_completed(futures)
            }
        return [indexed[index] for index in sorted(indexed)]

    def _run_job(
        self,
        harness: Path,
        evaluation_dir: Path,
        workload: str,
        device: str,
    ) -> dict[str, Any]:
        job_dir = evaluation_dir / "jobs" / workload
        job_dir.mkdir(parents=True, exist_ok=True)
        task_data = job_dir / "_task"
        task_data.mkdir(exist_ok=True)
        (task_data / "task_description.md").write_text(
            f"Pinned MLS-Bench {TASK_PATH} at {OFFICIAL_COMMIT}.\n",
            encoding="utf-8",
        )
        command = [
            self.evaluator_python,
            str(harness),
            "--workload",
            workload,
            "--budget-bits",
            "4",
            "--seed",
            "42",
            "--model-id",
            self.model_id,
            "--max-examples",
            str(self.max_examples),
        ]
        if self.cpu:
            command.append("--cpu")
        env = _evaluation_environment(
            package_dir=self.package_dir,
            task_data=task_data,
            output_dir=job_dir / "output",
            device=device,
        )
        started = time.monotonic()
        timeout = min(self.timeout_seconds, float(WORKLOAD_TIMEOUTS[workload]))
        try:
            completed = subprocess.run(
                command,
                cwd=job_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            status = "succeeded" if completed.returncode == 0 else "failed"
            error = "" if status == "succeeded" else stderr[-2000:]
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            status = "timed_out"
            error = f"workload exceeded {timeout:.1f}s"
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        stdout_path.write_text(str(stdout), encoding="utf-8")
        stderr_path.write_text(str(stderr), encoding="utf-8")
        metrics = parse_test_metrics(str(stdout)) if status == "succeeded" else {}
        required = {"final_score", "effective_kv_bits", "kv_compression_ratio"}
        if status == "succeeded" and not required.issubset(metrics):
            status = "failed"
            error = "official harness did not emit complete TEST_METRICS"
        return {
            "workload": workload,
            "label": WORKLOAD_LABELS[workload],
            "device": "cpu" if self.cpu else device,
            "status": status,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "error": error,
        }


def compute_official_score(task_dir: Path, metrics: dict[str, float]) -> float:
    source_root = Path(task_dir) / "tests" / "mlsbench_src"
    task_meta = Path(task_dir) / "tests" / "meta"
    if not source_root.is_dir():
        raise FileNotFoundError(f"official MLS-Bench scoring source is missing: {source_root}")
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(task_meta))
    try:
        evaluate = importlib.import_module("mlsbench.scoring.evaluate")
        anchors_module = importlib.import_module("mlsbench.scoring.anchors")
        anchors = anchors_module.BaselineAnchors(task_meta)
        spec = evaluate.load_expanded_spec(task_meta, anchors)
        score = evaluate.score_record(spec, metrics, anchors)
    finally:
        try:
            sys.path.remove(str(task_meta))
            sys.path.remove(str(source_root))
        except ValueError:
            pass
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("official MLS-Bench scoring returned a non-finite value")
    return min(max(value, 0.0), 1.0)


def parse_test_metrics(output: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        if "TEST_METRICS:" in line:
            return {name: float(value) for name, value in METRIC_PATTERN.findall(line)}
    return {}


def replace_quantizer_class(harness: str, candidate: str) -> str:
    tree = ast.parse(harness)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef)
            and item.name == "AdaptiveKVQuantizer"
        ),
        None,
    )
    if node is None or node.end_lineno is None:
        raise ValueError("upstream harness has no AdaptiveKVQuantizer class")
    lines = harness.splitlines(keepends=True)
    return (
        "".join(lines[: node.lineno - 1])
        + candidate.rstrip()
        + "\n"
        + "".join(lines[node.end_lineno :])
    )


def fixed_harness_sha256(harness: str) -> str:
    tree = ast.parse(harness)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "AdaptiveKVQuantizer"
    )
    if node.end_lineno is None:
        raise ValueError("upstream quantizer class has no source boundary")
    lines = harness.splitlines(keepends=True)
    fixed = "".join(lines[: node.lineno - 1]) + "".join(lines[node.end_lineno :])
    return hashlib.sha256(fixed.encode("utf-8")).hexdigest()


def _flatten_official_metrics(jobs: list[dict[str, Any]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for job in jobs:
        label = str(job["label"])
        for name, value in job["metrics"].items():
            flattened[f"{name}_{label}"] = float(value)
    return flattened


def _isolated_environment(*, device: str) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "")}
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return env


def _evaluation_environment(
    *,
    package_dir: Path,
    task_data: Path,
    output_dir: Path,
    device: str,
) -> dict[str, str]:
    allowed_prefixes = ("HF_", "HUGGINGFACE_", "TRANSFORMERS_", "CUDA_")
    allowed_names = {
        "PATH",
        "LD_LIBRARY_PATH",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "DATA_ROOT",
    }
    env = {
        name: value
        for name, value in os.environ.items()
        if name in allowed_names or name.startswith(allowed_prefixes)
    }
    env["PYTHONPATH"] = str(package_dir / "src")
    env["MLSBENCH_TASK_DIR"] = str(task_data)
    env["OUTPUT_DIR"] = str(output_dir)
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    return env


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0
