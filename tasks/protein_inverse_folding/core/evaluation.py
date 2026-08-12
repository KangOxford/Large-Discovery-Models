"""Mock, GPU-smoke, and MLS-Bench evaluators for candidate designs."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tasks.protein_inverse_folding.core.candidate import (
    CandidateProposal,
    assemble_candidate,
    design_code_without_overrides,
)


BENCHMARKS = ("CATH4.2", "CATH4.3", "TS50")
PARAMETER_BUDGET = 4_491_989
BASELINE_VALUES = {
    "CATH4.2": {
        "recovery": (0.4603, 0.4648, 0.4310),
        "perplexity": (5.4723, 5.2943, 5.8996),
    },
    "CATH4.3": {
        "recovery": (0.4612, 0.4772, 0.4337),
        "perplexity": (5.4639, 5.1294, 5.8209),
    },
    "TS50": {
        "recovery": (0.4829, 0.5228, 0.4602),
        "perplexity": (5.0827, 4.5513, 5.4831),
    },
}
TEST_METRIC_PATTERN = re.compile(
    r"^TEST_METRICS\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


class EvaluationError(RuntimeError):
    """Raised when a candidate cannot be evaluated faithfully."""


@dataclass(frozen=True)
class CommandResult:
    """Captured result for one isolated evaluator subprocess."""

    label: str
    returncode: int
    stdout: str
    stderr: str


def aggregate_mls_score(
    metrics: dict[str, Any], benchmarks: Sequence[str] = BENCHMARKS
) -> float:
    """Apply the public MLS-Bench recovery/perplexity composition."""

    setting_scores: list[float] = []
    for label in benchmarks:
        recovery = _finite_metric(metrics, f"recovery_{label}")
        perplexity = _finite_metric(metrics, f"perplexity_{label}")
        if perplexity <= 0:
            raise EvaluationError(f"perplexity_{label} must be positive")
        anchors = BASELINE_VALUES[label]
        recovery_term = _baseline_bounded_power(
            recovery,
            anchors["recovery"],
            direction="higher",
            bound=1.0,
        )
        perplexity_term = _baseline_bounded_power(
            perplexity,
            anchors["perplexity"],
            direction="lower",
            bound=1.0,
        )
        setting_scores.append((recovery_term + perplexity_term) / 2.0)
    if not setting_scores:
        raise EvaluationError("at least one benchmark is required")
    if all(value <= 0.0 for value in setting_scores):
        return 0.0
    product = math.prod(max(value, 0.01) for value in setting_scores)
    return product ** (1.0 / len(setting_scores))


def continuous_search_score(
    metrics: dict[str, Any], benchmarks: Sequence[str] = BENCHMARKS
) -> float:
    """Return an unclipped, MLS-aligned objective for surrogate fitting.

    The public bounded-power score is exactly zero below the worst pinned
    baseline.  This linear extension preserves the same directions and equal
    metric/benchmark weighting while retaining signal below that floor.
    """

    setting_scores: list[float] = []
    for label in benchmarks:
        recovery = _finite_metric(metrics, f"recovery_{label}")
        perplexity = _finite_metric(metrics, f"perplexity_{label}")
        recovery_floor = min(BASELINE_VALUES[label]["recovery"])
        perplexity_floor = max(BASELINE_VALUES[label]["perplexity"])
        recovery_term = (recovery - recovery_floor) / (1.0 - recovery_floor)
        perplexity_term = (perplexity_floor - perplexity) / (perplexity_floor - 1.0)
        setting_scores.append((recovery_term + perplexity_term) / 2.0)
    if not setting_scores:
        raise EvaluationError("at least one benchmark is required")
    return sum(setting_scores) / len(setting_scores)


def parse_test_metrics(output: str, label: str) -> dict[str, float]:
    """Parse the two fixed TEST_METRICS lines emitted by MLS-Bench."""

    metrics: dict[str, float] = {}
    for raw_line in output.splitlines():
        match = TEST_METRIC_PATTERN.fullmatch(raw_line.strip())
        if match:
            metrics[f"{match.group('name')}_{label}"] = float(match.group("value"))
    required = {f"recovery_{label}", f"perplexity_{label}"}
    missing = sorted(required - metrics.keys())
    if missing:
        raise EvaluationError(
            f"{label} output is missing metric(s): {', '.join(missing)}"
        )
    return metrics


def evaluate_mock(proposal: CandidateProposal) -> dict[str, float]:
    """Score code deterministically without importing torch or using a GPU."""

    tree = ast.parse(proposal.code)
    class_count = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    layer_norms = sum(
        isinstance(node, ast.Attribute) and node.attr == "LayerNorm"
        for node in ast.walk(tree)
    )
    residual_adds = sum(isinstance(node, ast.Add) for node in ast.walk(tree))
    dropout = float(proposal.config_overrides.get("dropout", 0.1))
    layers = int(proposal.config_overrides.get("num_encoder_layers", 3))
    architecture_signal = min(
        0.18,
        class_count * 0.01 + layer_norms * 0.012 + residual_adds * 0.004,
    )
    tuning_signal = max(0.0, 0.025 - abs(dropout - 0.08) * 0.1)
    depth_signal = max(0.0, 0.015 - abs(layers - 4) * 0.004)
    base_recovery = 0.39 + architecture_signal + tuning_signal + depth_signal
    metrics: dict[str, float] = {}
    for index, label in enumerate(BENCHMARKS):
        recovery = max(0.01, min(0.95, base_recovery - 0.012 * index))
        perplexity = max(1.01, 7.5 - recovery * 4.0 + 0.15 * index)
        metrics[f"recovery_{label}"] = round(recovery, 8)
        metrics[f"perplexity_{label}"] = round(perplexity, 8)
    metrics["search_score"] = continuous_search_score(metrics)
    metrics["aggregate_score"] = aggregate_mls_score(metrics)
    return metrics


def evaluate_gpu_smoke(
    proposal: CandidateProposal,
    *,
    work_dir: Path,
    prelude_path: Path,
    gpu_devices: Sequence[int],
    timeout_seconds: int,
    parameter_budget: int = PARAMETER_BUDGET,
) -> dict[str, Any]:
    """Run shape, normalization, and backward checks on every requested GPU."""

    if not gpu_devices:
        raise EvaluationError("GPU smoke requires at least one --gpu-device")
    work_dir.mkdir(parents=True, exist_ok=True)
    module_text = (
        prelude_path.read_text(encoding="utf-8")
        + "\n\n"
        + design_code_without_overrides(proposal.code)
        + "\nCONFIG_OVERRIDES = "
        + repr(proposal.config_overrides)
        + "\n"
        + "\n\n"
        + _GPU_SMOKE_MAIN
    )
    script_path = work_dir / "gpu_smoke_candidate.py"
    script_path.write_text(module_text, encoding="utf-8")

    results: dict[int, CommandResult] = {}
    with ThreadPoolExecutor(max_workers=len(gpu_devices)) as pool:
        futures = {
            pool.submit(
                _run_command,
                label=f"gpu_{device}",
                command=[sys.executable, str(script_path)],
                cwd=work_dir,
                env_overrides={
                    "CUDA_VISIBLE_DEVICES": str(device),
                    "LDM_PARAMETER_BUDGET": str(parameter_budget),
                },
                timeout_seconds=timeout_seconds,
            ): device
            for device in gpu_devices
        }
        for future in as_completed(futures):
            device = futures[future]
            results[device] = future.result()

    failures = [result for result in results.values() if result.returncode != 0]
    if failures:
        detail = "\n".join(
            f"{result.label}: {(result.stderr or result.stdout)[-1500:]}"
            for result in failures
        )
        raise EvaluationError(f"GPU smoke failed:\n{detail}")

    device_metrics: dict[str, Any] = {}
    for device, result in sorted(results.items()):
        payload = _last_json_metric(result.stdout)
        device_metrics[str(device)] = payload
        (work_dir / f"gpu_{device}.stdout.log").write_text(
            result.stdout, encoding="utf-8"
        )
        (work_dir / f"gpu_{device}.stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
    return {
        "aggregate_score": 1.0,
        "gpu_count": len(results),
        "parameter_budget": parameter_budget,
        "devices": device_metrics,
    }


def evaluate_benchmarks(
    proposal: CandidateProposal,
    *,
    work_dir: Path,
    scaffold_path: Path,
    data_root: Path,
    benchmarks: Sequence[str],
    gpu_devices: Sequence[int],
    epochs: int,
    batch_size: int,
    cath_max_train_hours: float,
    ts_max_train_hours: float,
    cath_job_timeout_seconds: int,
    ts_job_timeout_seconds: int,
    timeout_seconds: int,
    parallel: bool,
) -> dict[str, float]:
    """Assemble and run the public MLS-Bench CATH/TS50 evaluator."""

    unknown = sorted(set(benchmarks) - set(BENCHMARKS))
    if unknown:
        raise EvaluationError(f"unknown benchmark(s): {', '.join(unknown)}")
    if not scaffold_path.is_file():
        raise EvaluationError(f"MLS-Bench scaffold does not exist: {scaffold_path}")
    if not data_root.is_dir():
        raise EvaluationError(f"protein data root does not exist: {data_root}")
    if not gpu_devices:
        raise EvaluationError("benchmark evaluation requires at least one GPU")

    work_dir.mkdir(parents=True, exist_ok=True)
    assembled_path = work_dir / "custom_invfold.py"
    assembled_path.write_text(
        assemble_candidate(scaffold_path.read_text(encoding="utf-8"), proposal),
        encoding="utf-8",
    )
    budget_metrics = evaluate_gpu_smoke(
        proposal,
        work_dir=work_dir / "budget_smoke",
        prelude_path=Path(__file__).resolve().parents[1] / "resources" / "smoke_prelude.py",
        gpu_devices=[gpu_devices[0]],
        timeout_seconds=min(timeout_seconds, 300),
        parameter_budget=PARAMETER_BUDGET,
    )
    jobs: list[tuple[str, list[str], dict[str, str], Path, int]] = []
    for index, label in enumerate(benchmarks):
        dataset = "TS" if label == "TS50" else label
        output_dir = work_dir / f"output_{_safe_label(label)}"
        command = [
            sys.executable,
            str(assembled_path),
            "--dataset",
            dataset,
            "--data-root",
            str(data_root),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            "1e-3",
            "--hidden-dim",
            "128",
            "--num-encoder-layers",
            "3",
            "--k-neighbors",
            "30",
            "--dropout",
            "0.1",
            "--max-train-hours",
            str(ts_max_train_hours if label == "TS50" else cath_max_train_hours),
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ]
        jobs.append(
            (
                label,
                command,
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu_devices[index % len(gpu_devices)]),
                    "PYTHONUNBUFFERED": "1",
                },
                output_dir,
                min(
                    timeout_seconds,
                    ts_job_timeout_seconds if label == "TS50" else cath_job_timeout_seconds,
                ),
            )
        )

    results: list[CommandResult] = []
    if parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=min(len(jobs), len(gpu_devices))) as pool:
            futures = {
                pool.submit(
                    _run_command,
                    label=label,
                    command=command,
                    cwd=work_dir,
                    env_overrides=env,
                    timeout_seconds=job_timeout,
                ): (label, output_dir)
                for label, command, env, output_dir, job_timeout in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for label, command, env, _, job_timeout in jobs:
            results.append(
                _run_command(
                    label=label,
                    command=command,
                    cwd=work_dir,
                    env_overrides=env,
                    timeout_seconds=job_timeout,
                )
            )

    metrics: dict[str, float] = {}
    first_device = str(gpu_devices[0])
    metrics["parameter_count"] = float(
        budget_metrics["devices"][first_device]["parameter_count"]
    )
    for result in results:
        (work_dir / f"{_safe_label(result.label)}.stdout.log").write_text(
            result.stdout, encoding="utf-8"
        )
        (work_dir / f"{_safe_label(result.label)}.stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout)[-2000:]
            raise EvaluationError(
                f"{result.label} exited with {result.returncode}: {detail}"
            )
        metrics.update(parse_test_metrics(result.stdout, result.label))
    metrics["search_score"] = continuous_search_score(metrics, benchmarks)
    metrics["aggregate_score"] = aggregate_mls_score(metrics, benchmarks)
    return metrics


def _run_command(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    env_overrides: dict[str, str],
    timeout_seconds: int,
) -> CommandResult:
    env = {**os.environ, **env_overrides}
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        return CommandResult(label, 124, stdout, stderr + "\nevaluation timed out")
    return CommandResult(label, completed.returncode, completed.stdout, completed.stderr)


def _last_json_metric(output: str) -> dict[str, Any]:
    prefix = "LDM_METRICS "
    for line in reversed(output.splitlines()):
        if line.startswith(prefix):
            payload = json.loads(line[len(prefix) :])
            if isinstance(payload, dict):
                return payload
    raise EvaluationError("GPU smoke did not emit LDM_METRICS JSON")


def _finite_metric(metrics: dict[str, Any], key: str) -> float:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"missing or invalid metric: {key}") from exc
    if not math.isfinite(value):
        raise EvaluationError(f"metric is not finite: {key}={value}")
    return value


def _baseline_bounded_power(
    value: float,
    baselines: Sequence[float],
    *,
    direction: str,
    bound: float,
) -> float:
    """Match MLS-Bench bounded_power with worst/best baseline calibration."""

    if direction == "higher":
        floor_raw, ref_raw = min(baselines), max(baselines)
        transformed = value
        floor = floor_raw
        ref = ref_raw
        transformed_bound = bound
    elif direction == "lower":
        floor_raw, ref_raw = max(baselines), min(baselines)
        transformed = -value
        floor = -floor_raw
        ref = -ref_raw
        transformed_bound = -bound
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"unknown metric direction: {direction}")
    denominator = transformed_bound - floor
    if denominator <= 0:
        raise EvaluationError("invalid bounded-power baseline anchors")
    reference_ratio = min(1.0, max(0.0, (ref - floor) / denominator))
    gamma = (
        math.log(0.5) / math.log(reference_ratio)
        if 0.0 < reference_ratio < 1.0
        else 1.0
    )
    gamma = min(10.0, max(0.1, gamma))
    ratio = min(1.0, max(0.0, (transformed - floor) / denominator))
    return ratio**gamma


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label)


_GPU_SMOKE_MAIN = r'''
if __name__ == "__main__":
    import json
    import os
    import time

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    model = InverseFoldingModel(
        hidden_dim=128,
        num_encoder_layers=int(CONFIG_OVERRIDES.get("num_encoder_layers", 3)),
        k_neighbors=12,
        dropout=float(CONFIG_OVERRIDES.get("dropout", 0.1)),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_budget = int(os.environ.get("LDM_PARAMETER_BUDGET", "4491989"))
    if parameter_count > parameter_budget:
        raise RuntimeError(
            f"parameter budget exceeded: {parameter_count} > {parameter_budget}"
        )
    X = torch.randn(2, 24, 4, 3, device=device)
    mask = torch.ones(2, 24, device=device)
    mask[1, 19:] = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    log_probs = model(X, mask)
    if tuple(log_probs.shape) != (2, 24, 20):
        raise RuntimeError(f"unexpected output shape: {tuple(log_probs.shape)}")
    if not torch.isfinite(log_probs).all():
        raise RuntimeError("model output contains non-finite values")
    normalizer_error = (log_probs.exp().sum(-1) - 1.0).abs().max().item()
    if normalizer_error > 1e-4:
        raise RuntimeError(f"invalid log probabilities: {normalizer_error}")
    loss = -log_probs[mask.bool()].mean()
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("backward pass produced no parameter gradients")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print("LDM_METRICS " + json.dumps({
        "device_name": torch.cuda.get_device_name(0),
        "elapsed_seconds": elapsed,
        "normalizer_error": normalizer_error,
        "parameter_count": parameter_count,
        "shape": list(log_probs.shape),
    }, sort_keys=True))
'''
