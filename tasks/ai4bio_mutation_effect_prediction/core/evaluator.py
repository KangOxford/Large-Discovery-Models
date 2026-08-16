"""Qualification-only mock scoring and pinned MLS-Bench evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from collections.abc import Mapping

from ldm_tts.contracts import Candidate, EvaluationResult
from ldm_tts.engine.run_store import atomic_json_write

from tasks.ai4bio_mutation_effect_prediction.core.candidate import (
    EMBED_DIM,
    PARAMETER_LIMIT,
    predictor_parameter_count,
)
from tasks.ai4bio_mutation_effect_prediction.core.upstream import (
    OFFICIAL_COMMIT,
    TASK_PATH,
    UPSTREAM_ROOT_SHA256,
    UPSTREAM_SHA256,
)


OFFICIAL_ASSAYS = ("BLAT_ECOLX", "ESTA_BACSU", "RASH_HUMAN")
ASSAY_IDS = {
    "BLAT_ECOLX": "BLAT_ECOLX_Firnberg_2014",
    "ESTA_BACSU": "ESTA_BACSU_Nutschel_2020",
    "RASH_HUMAN": "RASH_HUMAN_Bandaru_2017",
}
ASSAY_COUNTS = {
    "BLAT_ECOLX": 4783,
    "ESTA_BACSU": 2172,
    "RASH_HUMAN": 3134,
}
METRIC_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)


class MockMutationEvaluator:
    """Deterministic architecture heuristic, explicitly not an official score."""

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        spec = candidate.payload["spec"]
        depth = len(spec["hidden_dims"])
        feature_bonus = {"embedding": 0.01, "delta": 0.04, "concat": 0.05}[
            spec["feature_mode"]
        ]
        activation_bonus = {"relu": 0.005, "gelu": 0.015, "silu": 0.012}[
            spec["activation"]
        ]
        depth_bonus = {0: 0.0, 1: 0.03, 2: 0.05, 3: 0.035}[depth]
        regularization_bonus = max(0.0, 0.02 - abs(float(spec["dropout"]) - 0.1) * 0.08)
        normalization_bonus = 0.01 if spec["layer_norm"] and depth else 0.0
        parameter_ratio = predictor_parameter_count(spec) / PARAMETER_LIMIT
        capacity_bonus = 0.02 * min(parameter_ratio / 0.2, 1.0)
        center = (
            0.42
            + feature_bonus
            + activation_bonus
            + depth_bonus
            + regularization_bonus
            + normalization_bonus
            + capacity_bonus
        )
        metrics = {
            "spearman_BLAT_ECOLX": min(center + 0.025, 0.95),
            "spearman_ESTA_BACSU": min(center - 0.015, 0.95),
            "spearman_RASH_HUMAN": min(center + 0.005, 0.95),
        }
        selection_score = geometric_mean_nonnegative(metrics)
        metrics["selection_score"] = selection_score
        metrics["parameter_count"] = float(predictor_parameter_count(spec))
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics=metrics,
            resource_usage={"benchmark_jobs": 1},
            metadata={
                "evaluator": "deterministic_mock",
                "official": False,
                "warning": "Synthetic qualification signal; not benchmark-comparable.",
            },
        )


class MLSBenchMutationEvaluator:
    """Run the three official ProteinGym jobs against an immutable template."""

    def __init__(
        self,
        *,
        upstream_root: Path,
        data_dir: Path,
        cv_dir: Path,
        run_dir: Path,
        timeout_seconds: float = 3540.0,
        evaluator_python: str | Path = sys.executable,
    ) -> None:
        self.upstream_root = Path(upstream_root).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.cv_dir = Path(cv_dir).resolve()
        self.run_dir = Path(run_dir)
        self.timeout_seconds = min(float(timeout_seconds), 3540.0)
        self.evaluator_python = str(evaluator_python)

    @property
    def task_dir(self) -> Path:
        return self.upstream_root / TASK_PATH

    @property
    def template_path(self) -> Path:
        return self.task_dir / "edits" / "custom_template.py"

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        evaluation_dir = self.run_dir / "evaluations" / candidate.candidate_id
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = evaluation_dir / "evaluation_manifest.json"
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "source_commit": OFFICIAL_COMMIT,
            "assays": list(OFFICIAL_ASSAYS),
            "jobs": [],
            "status": "running",
        }
        atomic_json_write(manifest_path, manifest)
        started = time.monotonic()
        try:
            validate_upstream_contract(self.task_dir)
            data_summary = validate_official_data(self.data_dir, self.cv_dir)
            template = self.template_path.read_text(encoding="utf-8")
            harness = materialize_harness(
                template,
                str(candidate.payload["code"]),
                candidate.payload["config_overrides"],
            )
            if fixed_template_sha256(harness) != fixed_template_sha256(template):
                raise ValueError("materialized harness changed fixed template regions")
            harness_path = evaluation_dir / "custom_mutation_pred.py"
            harness_path.write_text(harness, encoding="utf-8")
            contract = validate_predictor_contract(
                harness_path,
                expected_parameters=predictor_parameter_count(candidate.payload["spec"]),
            )
            jobs = self._run_jobs(harness_path, evaluation_dir)
            manifest["jobs"] = jobs
            failed = [job for job in jobs if job["status"] != "succeeded"]
            if failed:
                raise RuntimeError(
                    "; ".join(
                        f"{job['assay']}: {job.get('error') or job['status']}"
                        for job in failed
                    )
                )
            metrics = {
                key: value
                for job in jobs
                for key, value in job["metrics"].items()
            }
            scoring = score_with_pinned_mlsbench(
                self.task_dir,
                metrics,
                evaluator_python=self.evaluator_python,
            )
            metrics["official_score"] = scoring["official_score"]
            metrics["selection_score"] = metrics["official_score"]
            metrics.update(
                {
                    f"normalized_{name}": value
                    for name, value in scoring["setting_scores"].items()
                }
            )
            metrics["parameter_count"] = float(contract["parameter_count"])
            metrics["runtime_seconds"] = time.monotonic() - started
            manifest.update(
                status="succeeded",
                metrics=metrics,
                scoring=scoring,
                data=data_summary,
                contract=contract,
            )
            atomic_json_write(manifest_path, manifest)
            return EvaluationResult(
                candidate.candidate_id,
                "succeeded",
                metrics=metrics,
                artifacts={
                    "evaluation_manifest": self._artifact_path(manifest_path),
                    "harness": self._artifact_path(harness_path),
                },
                resource_usage={
                    "wall_seconds": time.monotonic() - started,
                    "benchmark_jobs": 3,
                },
                metadata={
                    "evaluator": "pinned_mlsbench_proteingym",
                    "official": True,
                    "source_commit": OFFICIAL_COMMIT,
                },
            )
        except Exception as exc:
            manifest.update(status="failed", error=str(exc))
            atomic_json_write(manifest_path, manifest)
            return EvaluationResult(
                candidate.candidate_id,
                "failed",
                error=str(exc),
                artifacts={"evaluation_manifest": self._artifact_path(manifest_path)},
                resource_usage={
                    "wall_seconds": time.monotonic() - started,
                    "benchmark_jobs": len(manifest["jobs"]),
                },
                metadata={"evaluator": "pinned_mlsbench_proteingym", "official": True},
            )

    def _artifact_path(self, path: Path) -> str:
        """Serialize evaluator artifacts relative to the portable run root."""

        try:
            return path.relative_to(self.run_dir).as_posix()
        except ValueError:
            return path.resolve().relative_to(self.run_dir.resolve()).as_posix()

    def _run_jobs(self, harness: Path, evaluation_dir: Path) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_job, harness, evaluation_dir, assay): index
                for index, assay in enumerate(OFFICIAL_ASSAYS)
            }
            indexed = {
                futures[future]: future.result()
                for future in as_completed(futures)
            }
        return [indexed[index] for index in sorted(indexed)]

    def _run_job(
        self, harness: Path, evaluation_dir: Path, assay: str
    ) -> dict[str, Any]:
        job_dir = evaluation_dir / "jobs" / assay
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.evaluator_python,
            str(harness),
            "--assay-id",
            ASSAY_IDS[assay],
            "--data-dir",
            str(self.data_dir),
            "--cv-dir",
            str(self.cv_dir),
            "--epochs",
            "200",
            "--batch-size",
            "64",
            "--lr",
            "0.001",
            "--weight-decay",
            "0.05",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=job_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            status = "succeeded" if completed.returncode == 0 else "failed"
            error = "" if status == "succeeded" else stderr[-2000:]
        except subprocess.TimeoutExpired as exc:
            stdout = _timeout_text(exc.stdout)
            stderr = _timeout_text(exc.stderr)
            status = "timed_out"
            error = f"assay exceeded {self.timeout_seconds:.1f}s"
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        metrics = parse_test_metrics(stdout, assay) if status == "succeeded" else {}
        results_path = output_dir / "results.pt"
        if status == "succeeded" and not metrics:
            status = "failed"
            error = "official harness did not emit TEST_METRICS spearman"
        if status == "succeeded" and not results_path.is_file():
            status = "failed"
            error = "official harness did not create results.pt"
        results_summary: dict[str, Any] = {}
        if status == "succeeded":
            import torch

            saved = torch.load(results_path, map_location="cpu", weights_only=False)
            saved_scores = [float(value) for value in saved.get("fold_spearmans", [])]
            saved_mean = float(saved.get("mean_spearman", float("nan")))
            parsed_mean = metrics[f"spearman_{assay}"]
            if saved.get("assay_id") != ASSAY_IDS[assay]:
                status = "failed"
                error = "results.pt assay ID does not match the requested assay"
            elif len(saved_scores) != 5 or not all(math.isfinite(value) for value in saved_scores):
                status = "failed"
                error = "results.pt does not contain five finite fold Spearman values"
            elif not math.isfinite(saved_mean) or abs(saved_mean - parsed_mean) > 1.0e-6:
                status = "failed"
                error = "results.pt mean does not match TEST_METRICS"
            else:
                results_summary = {
                    "mean_spearman": saved_mean,
                    "std_spearman": float(saved["std_spearman"]),
                    "fold_spearmans": saved_scores,
                }
        return {
            "assay": assay,
            "assay_id": ASSAY_IDS[assay],
            "device": env["CUDA_VISIBLE_DEVICES"],
            "status": status,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
            "results_summary": results_summary,
            "results": str(results_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "error": error,
        }


def validate_upstream_contract(task_dir: Path) -> dict[str, str]:
    task_dir = Path(task_dir)
    actual: dict[str, str] = {}
    for relative, expected in UPSTREAM_SHA256.items():
        path = task_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned upstream file is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"pinned upstream digest mismatch: {relative}")
        actual[relative] = digest
    upstream_root = task_dir.parents[1]
    for relative, expected in UPSTREAM_ROOT_SHA256.items():
        path = upstream_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned upstream scorer file is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"pinned upstream scorer digest mismatch: {relative}")
        actual[relative] = digest
    return actual


def validate_official_data(data_dir: Path, cv_dir: Path) -> dict[str, Any]:
    import pandas as pd
    import torch

    data_dir = Path(data_dir)
    cv_dir = Path(cv_dir)
    summary: dict[str, Any] = {}
    for assay in OFFICIAL_ASSAYS:
        assay_id = ASSAY_IDS[assay]
        embedding_path = data_dir / f"{assay_id}.pt"
        if not embedding_path.is_file():
            raise FileNotFoundError(f"official embedding is missing: {embedding_path}")
        payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
        required = {"embeddings", "scores", "mutant_ids", "wt_embedding"}
        if not required.issubset(payload):
            raise ValueError(f"{assay_id} embedding payload is incomplete")
        embeddings = payload["embeddings"]
        scores = payload["scores"]
        wt_embedding = payload["wt_embedding"]
        mutant_ids = [str(value) for value in payload["mutant_ids"]]
        expected_count = ASSAY_COUNTS[assay]
        if tuple(embeddings.shape) != (expected_count, EMBED_DIM):
            raise ValueError(f"{assay_id} embeddings have shape {tuple(embeddings.shape)}")
        if tuple(scores.shape) != (expected_count,):
            raise ValueError(f"{assay_id} scores have shape {tuple(scores.shape)}")
        if tuple(wt_embedding.shape) != (EMBED_DIM,):
            raise ValueError(f"{assay_id} WT embedding has shape {tuple(wt_embedding.shape)}")
        if len(mutant_ids) != expected_count:
            raise ValueError(f"{assay_id} mutant ID count does not match embeddings")
        tensors = (embeddings, scores, wt_embedding)
        if not all(tensor.dtype.is_floating_point for tensor in tensors):
            raise ValueError(f"{assay_id} tensors must use floating dtypes")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError(f"{assay_id} tensors contain non-finite values")
        matches = sorted(cv_dir.rglob(f"{assay_id}*.csv"))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one CV file for {assay_id}, found {len(matches)}")
        folds_frame = pd.read_csv(matches[0])
        if "mutant" not in folds_frame or "fold_random_5" not in folds_frame:
            raise ValueError(f"{assay_id} CV file lacks mutant or fold_random_5")
        singles = folds_frame[
            ~folds_frame["mutant"].astype(str).str.contains(":", regex=False)
        ].reset_index(drop=True)
        fold_mutants = singles["mutant"].astype(str).tolist()
        if fold_mutants != mutant_ids:
            raise ValueError(f"{assay_id} CV mutant order does not match embeddings")
        fold_ids = {int(value) for value in singles["fold_random_5"].tolist()}
        if fold_ids != {0, 1, 2, 3, 4}:
            raise ValueError(f"{assay_id} CV file does not contain all five random folds")
        summary[assay] = {
            "assay_id": assay_id,
            "samples": expected_count,
            "embedding_shape": [expected_count, EMBED_DIM],
            "dtype": str(embeddings.dtype),
            "fold_ids": sorted(fold_ids),
            "cv_file": str(matches[0]),
        }
    return summary


def materialize_harness(
    template: str, candidate_source: str, config_overrides: Mapping[str, Any]
) -> str:
    lines = template.splitlines()
    if len(lines) < 347:
        raise ValueError("upstream template is shorter than the pinned editable regions")
    allowed = {"learning_rate", "weight_decay"}
    if set(config_overrides) != allowed:
        raise ValueError("config overrides must contain learning_rate and weight_decay")
    override = "    CONFIG_OVERRIDES = " + repr(
        {key: float(config_overrides[key]) for key in sorted(allowed)}
    )
    lines[344:347] = [override]
    lines[107:137] = candidate_source.rstrip().splitlines()
    return "\n".join(lines) + "\n"


def fixed_template_sha256(source: str) -> str:
    lines = source.splitlines()
    if len(lines) < 140:
        raise ValueError("template lacks expected editable markers")
    start_markers = [
        index for index, line in enumerate(lines) if "EDITABLE SECTION START" in line
    ]
    end_markers = [
        index for index, line in enumerate(lines) if line.strip() == "# EDITABLE SECTION END"
    ]
    if len(start_markers) != 2 or len(end_markers) != 2:
        raise ValueError("template must contain exactly two editable regions")
    fixed: list[str] = []
    cursor = 0
    for start, end in zip(start_markers, end_markers):
        if start >= end:
            raise ValueError("template editable markers are malformed")
        fixed.extend(lines[cursor : start + 1])
        fixed.append("<EDITABLE_REGION>")
        cursor = end
    fixed.extend(lines[cursor:])
    return hashlib.sha256("\n".join(fixed).encode("utf-8")).hexdigest()


def validate_predictor_contract(
    harness_path: Path, *, expected_parameters: int
) -> dict[str, Any]:
    import torch

    spec = importlib.util.spec_from_file_location("_mutation_contract", harness_path)
    if spec is None or spec.loader is None:
        raise ValueError("could not import materialized predictor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MutationPredictor(embed_dim=EMBED_DIM)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected_parameters or actual_parameters > PARAMETER_LIMIT:
        raise ValueError(
            f"materialized parameter count {actual_parameters} does not match {expected_parameters}"
        )
    devices = ["cpu"]
    if not torch.cuda.is_available():
        raise ValueError("official evaluator requires CUDA")
    devices.append("cuda")
    for device in devices:
        checked = model.to(device).eval()
        embedding = torch.randn(3, EMBED_DIM, device=device, dtype=torch.float32)
        delta = torch.randn(3, EMBED_DIM, device=device, dtype=torch.float32)
        with torch.no_grad():
            output = checked(embedding, delta)
        if tuple(output.shape) != (3,):
            raise ValueError(f"predictor returned shape {tuple(output.shape)} on {device}")
        if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
            raise ValueError(f"predictor returned invalid output on {device}")
    return {
        "parameter_count": actual_parameters,
        "devices": devices,
        "output_shape": [3],
        "dtype": "torch.float32",
    }


def parse_test_metrics(output: str, assay_label: str) -> dict[str, float]:
    if assay_label not in OFFICIAL_ASSAYS:
        raise ValueError(f"unknown assay label: {assay_label}")
    for line in reversed(output.splitlines()):
        if line.strip().startswith("TEST_METRICS "):
            values = {name: float(value) for name, value in METRIC_PATTERN.findall(line)}
            if "spearman" in values:
                return {f"spearman_{assay_label}": values["spearman"]}
    return {}


def geometric_mean_nonnegative(metrics: Mapping[str, float]) -> float:
    names = tuple(f"spearman_{assay}" for assay in OFFICIAL_ASSAYS)
    missing = [name for name in names if name not in metrics]
    if missing:
        raise ValueError("missing official assay metric(s): " + ", ".join(missing))
    values = [float(metrics[name]) for name in names]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("official assay metrics must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("geometric aggregation requires non-negative assay scores")
    return float(math.prod(values) ** (1.0 / len(values)))


def score_with_pinned_mlsbench(
    task_dir: Path,
    metrics: Mapping[str, float],
    *,
    evaluator_python: str | Path = sys.executable,
) -> dict[str, Any]:
    names = tuple(f"spearman_{assay}" for assay in OFFICIAL_ASSAYS)
    record = {name: float(metrics[name]) for name in names if name in metrics}
    missing = [name for name in names if name not in record]
    if missing:
        raise ValueError("missing official assay metric(s): " + ", ".join(missing))
    if any(not math.isfinite(value) for value in record.values()):
        raise ValueError("official assay metrics must be finite")
    task_dir = Path(task_dir).resolve()
    validate_upstream_contract(task_dir)
    upstream_root = task_dir.parents[1]
    scorer = """
import json
from pathlib import Path
import sys

from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details

task_dir = Path(sys.argv[1])
record = json.load(sys.stdin)
anchors = BaselineAnchors(task_dir)
spec = load_expanded_spec(task_dir, anchors)
if spec is None:
    raise RuntimeError("pinned MLS-Bench score specification did not load")
score, settings, valid = score_record_details(spec, record, anchors)
print(json.dumps({
    "official_score": score,
    "record_valid": valid,
    "setting_scores": {item.name: item.score for item in settings},
}, sort_keys=True))
"""
    env = os.environ.copy()
    python_path = str(upstream_root / "src")
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path
    completed = subprocess.run(
        [str(evaluator_python), "-c", scorer, str(task_dir)],
        cwd=upstream_root,
        env=env,
        input=json.dumps(record),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "pinned MLS-Bench scorer failed: " + completed.stderr[-2000:]
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("pinned MLS-Bench scorer returned invalid JSON") from exc
    score = float(payload.get("official_score", float("nan")))
    setting_scores = payload.get("setting_scores")
    if not payload.get("record_valid") or not math.isfinite(score):
        raise ValueError("pinned MLS-Bench scorer rejected the evaluation record")
    if not isinstance(setting_scores, dict) or set(setting_scores) != set(OFFICIAL_ASSAYS):
        raise ValueError("pinned MLS-Bench scorer returned incomplete setting scores")
    return {
        "engine": "pinned_mlsbench.scoring.evaluate.score_record_details",
        "official_score": score,
        "record_valid": True,
        "setting_scores": {
            name: float(setting_scores[name]) for name in OFFICIAL_ASSAYS
        },
    }


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
