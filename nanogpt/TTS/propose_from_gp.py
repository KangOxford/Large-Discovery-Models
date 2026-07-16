#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TTS.run_model_based_search import (
    BufferEntry,
    GPSurrogate,
    OperationApplyResult,
    OperationSchema,
    Prediction,
    ValidatedOperation,
    apply_operations_to_train_text,
    as_float,
    choice_values_equal,
    expected_improvement,
    extract_top_level_assignment_values,
    fallback_operation_value,
    format_operation_value,
    load_buffer,
    load_operation_schema,
    normalize_operation_numeric,
    operation_feature_dim,
    operation_feature_version,
    operation_summary,
    sample_random_parameter_value,
    validate_operation_value,
)


@dataclass
class BaseTrain:
    text: str
    path: Path
    source: str
    best_entry: BufferEntry | None = None
    reconstruction: OperationApplyResult | None = None


@dataclass
class Proposal:
    operations: list[ValidatedOperation]
    apply_result: OperationApplyResult
    vector: list[float]
    params: dict[str, Any]
    source_hash: str
    prediction: Any
    source: str
    signature: str


@dataclass
class ScoredCandidate:
    operations: list[ValidatedOperation]
    vector: list[float]
    params: dict[str, Any]
    prediction: Any
    source: str
    signature: str


@dataclass
class Candidate:
    operations: list[ValidatedOperation]
    vector: list[float]
    params: dict[str, Any]
    source: str
    signature: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast GP-only proposer for fixed-operation TTS search. It loads a previous "
            "model-based search buffer, refits the same GP surrogate used by "
            "run_model_based_search.py, samples valid schema operations, and writes the "
            "best predicted candidate train.py without running training or calling an LLM."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Model-based run directory, model_based_summary.json, or JSONL GP buffer.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-file", type=Path, default=Path("TTS/real_train.py"))
    parser.add_argument(
        "--operation-schema",
        "--schema",
        dest="operation_schema",
        type=Path,
        default=None,
        help="Operation schema JSON. Default: <run>/operation_schema.json, then TTS/operation_schema_real_train.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <run>/gp_proposals/<timestamp> or TTS/runs/gp_proposals/<timestamp>.",
    )
    parser.add_argument("--samples", type=int, default=4096, help="Random operation candidates to sample and score.")
    parser.add_argument("--top-k", type=int, default=20, help="How many ranked candidates to write.")
    parser.add_argument("--max-operations", type=int, default=2, help="Maximum operations in the proposed next edit.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for proposal sampling.")
    parser.add_argument(
        "--base-source",
        choices=["best", "train-file"],
        default="best",
        help=(
            "Base script to mutate. best uses the best compatible buffer entry train_path when available, "
            "or reconstructs its schema values from --train-file; train-file mutates --train-file directly."
        ),
    )
    parser.add_argument(
        "--allow-seen",
        action="store_true",
        help="Allow proposing a full schema configuration already present in the GP buffer.",
    )
    parser.add_argument(
        "--surrogate-mode",
        choices=["lcb", "mean", "ei"],
        default="lcb",
        help="Acquisition used for ranking candidates. Lower selection_score is always better.",
    )
    parser.add_argument("--gp-beta", type=float, default=1.0, help="LCB/UCB exploration coefficient.")
    parser.add_argument("--gp-xi", type=float, default=0.001, help="Expected-improvement margin.")
    parser.add_argument("--gp-lengthscale", type=float, default=1.5)
    parser.add_argument("--gp-noise", type=float, default=1.0e-4)
    parser.add_argument("--prior-score", type=float, default=1.0)
    parser.add_argument("--prior-std", type=float, default=0.15)
    parser.add_argument("--score-key", default="val_bpb")
    parser.add_argument("--maximize", action="store_true")
    parser.add_argument("--failure-score", type=float, default=1.0e9)
    parser.add_argument(
        "--gp-reject-score-at-or-above",
        type=float,
        default=None,
        help="Skip buffer scores >= this threshold. Default: --failure-score for minimize runs.",
    )
    parser.add_argument(
        "--gp-reject-score-at-or-below",
        type=float,
        default=None,
        help="Skip buffer scores <= this threshold. Default: --failure-score for maximize runs.",
    )
    parser.add_argument(
        "--gp-allow-failure-status",
        action="store_true",
        help="Allow failed statuses such as crash/timeout/score_missing into the GP buffer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dir, buffer_path = resolve_input_path(args.path, project_root)
    schema = resolve_schema(args.operation_schema, run_dir, project_root)
    expected_dim = operation_feature_dim(schema)
    expected_version = operation_feature_version(schema)
    filter_args = argparse.Namespace(
        failure_score=args.failure_score,
        maximize=args.maximize,
        gp_reject_score_at_or_above=args.gp_reject_score_at_or_above,
        gp_reject_score_at_or_below=args.gp_reject_score_at_or_below,
        gp_allow_failure_status=args.gp_allow_failure_status,
    )
    entries = load_buffer(
        buffer_path,
        expected_dim,
        expected_feature_version=expected_version,
        args=filter_args,
    )
    if not entries:
        raise SystemExit(
            f"No compatible GP buffer entries found in {buffer_path}. "
            f"Expected feature_version={expected_version!r} and dim={expected_dim}."
        )

    gp = GPSurrogate(
        entries,
        lengthscale=args.gp_lengthscale,
        noise=args.gp_noise,
        prior_score=args.prior_score,
        prior_std=args.prior_std,
        minimize=not args.maximize,
    )
    train_file = resolve_path(args.train_file, project_root)
    base = choose_base_train(args, train_file, schema, entries, minimize=not args.maximize)
    out_dir = resolve_out_dir(args.out_dir, run_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_signatures = {
        params_signature(entry.params, schema)
        for entry in entries
        if isinstance(entry.params, dict) and entry.params
    }
    candidates, stats = generate_and_score_candidates(
        args,
        schema,
        base,
        gp,
        entries,
        seen_signatures=seen_signatures,
    )
    if not candidates:
        raise SystemExit(
            "Could not produce any valid unseen proposal. "
            "Try --allow-seen, increase --samples, or increase --max-operations."
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.prediction.selection_score),
            float(item.prediction.mean) if not args.maximize else -float(item.prediction.mean),
            item.signature,
        ),
    )
    top_k = materialize_top_proposals(base.text, ranked, schema, max_count=max(1, int(args.top_k)))
    stats["materialized"] = len(top_k)
    if not top_k:
        raise SystemExit("Top GP candidates could not be materialized as valid train.py edits.")
    best = top_k[0]
    write_outputs(args, out_dir, run_dir, buffer_path, schema, entries, gp, base, best, top_k, stats)
    payload = {
        "proposal_train": str((out_dir / "proposal_train.py").resolve()),
        "proposal_operations": str((out_dir / "proposal_operations.json").resolve()),
        "proposal_patch": str((out_dir / "proposal_patch.diff").resolve()),
        "ranked_proposals": str((out_dir / "ranked_proposals.tsv").resolve()),
        "summary": str((out_dir / "summary.json").resolve()),
        "buffer": str(buffer_path.resolve()),
        "entries": len(entries),
        "gp": gp.summary(),
        "base_source": base.source,
        "best_prediction": prediction_to_json(best.prediction),
        "best_operations": operation_summary(best.operations, best.apply_result.records),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def resolve_path(path: Path, project_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else project_root / path


def resolve_input_path(path: Path, project_root: Path) -> tuple[Path | None, Path]:
    path = resolve_path(path, project_root)
    if path.is_dir():
        for candidate in (path / "model_based_buffer.jsonl", path / "buffer.jsonl"):
            if candidate.exists():
                return path, candidate
        summary = load_json(path / "model_based_summary.json")
        if summary is not None:
            buffer_path = path_from_summary_buffer(summary, path, project_root)
            if buffer_path is not None:
                return path, buffer_path
        raise SystemExit(f"No model_based_buffer.jsonl or model_based_summary.json buffer found under {path}.")
    if path.name == "model_based_summary.json":
        summary = load_json(path)
        if summary is None:
            raise SystemExit(f"Could not read JSON summary {path}.")
        buffer_path = path_from_summary_buffer(summary, path.parent, project_root)
        if buffer_path is None:
            raise SystemExit(f"{path} does not contain a run_buffer or buffer path.")
        return path.parent, buffer_path
    return None, path


def path_from_summary_buffer(summary: dict[str, Any], base_dir: Path, project_root: Path) -> Path | None:
    value = summary.get("run_buffer") or summary.get("buffer")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    local = base_dir / path
    if local.exists():
        return local
    return project_root / path


def resolve_schema(schema_arg: Path | None, run_dir: Path | None, project_root: Path) -> OperationSchema:
    candidates: list[Path] = []
    if schema_arg is not None:
        candidates.append(resolve_path(schema_arg, project_root))
    if run_dir is not None:
        candidates.append(run_dir / "operation_schema.json")
    candidates.append(project_root / "TTS" / "operation_schema_real_train.json")
    candidates.append(project_root / "TTS" / "operation_schema_mock_train.json")
    for candidate in candidates:
        if candidate.exists():
            return load_operation_schema(candidate, project_root)
    raise SystemExit(
        "No operation schema found. Pass --operation-schema, or use a run directory that contains operation_schema.json."
    )


def resolve_out_dir(out_arg: Path | None, run_dir: Path | None, project_root: Path) -> Path:
    if out_arg is not None:
        return resolve_path(out_arg, project_root)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if run_dir is not None:
        return run_dir / "gp_proposals" / stamp
    return project_root / "TTS" / "runs" / "gp_proposals" / stamp


def choose_base_train(
    args: argparse.Namespace,
    train_file: Path,
    schema: OperationSchema,
    entries: list[BufferEntry],
    *,
    minimize: bool,
) -> BaseTrain:
    if not train_file.exists():
        raise SystemExit(f"--train-file does not exist: {train_file}")
    seed_text = train_file.read_text(encoding="utf-8")
    if args.base_source == "train-file":
        return BaseTrain(text=seed_text, path=train_file, source="train-file")

    best_entry = best_buffer_entry(entries, minimize=minimize)
    if best_entry is not None and best_entry.train_path:
        best_path = Path(best_entry.train_path).expanduser()
        if not best_path.is_absolute():
            best_path = train_file.parent / best_path
        if best_path.exists():
            return BaseTrain(
                text=best_path.read_text(encoding="utf-8"),
                path=best_path,
                source=f"best-buffer-train-path:{best_entry.state_id}",
                best_entry=best_entry,
            )

    if best_entry is not None and isinstance(best_entry.params, dict) and best_entry.params:
        reconstruction = reconstruct_from_entry_params(seed_text, schema, best_entry)
        if reconstruction is not None:
            return BaseTrain(
                text=reconstruction.text,
                path=train_file,
                source=f"best-buffer-params:{best_entry.state_id}",
                best_entry=best_entry,
                reconstruction=reconstruction,
            )
    return BaseTrain(text=seed_text, path=train_file, source="train-file-fallback", best_entry=best_entry)


def best_buffer_entry(entries: list[BufferEntry], *, minimize: bool) -> BufferEntry | None:
    finite = [entry for entry in entries if math.isfinite(float(entry.score))]
    if not finite:
        return None
    return sorted(finite, key=lambda entry: entry.score, reverse=not minimize)[0]


def reconstruct_from_entry_params(
    seed_text: str,
    schema: OperationSchema,
    entry: BufferEntry,
) -> OperationApplyResult | None:
    current = extract_top_level_assignment_values(seed_text)
    operations: list[ValidatedOperation] = []
    for parameter in schema.parameters.values():
        if parameter.name not in entry.params:
            continue
        try:
            value = validate_operation_value(entry.params[parameter.name], parameter, index=len(operations) + 1)
        except ValueError:
            continue
        if current.get(parameter.name) is not None and choice_values_equal(current.get(parameter.name), value):
            continue
        operations.append(
            ValidatedOperation(
                name=parameter.name,
                op="set_choice" if parameter.kind == "choice" else "set_numeric",
                value=value,
                rationale=f"Reconstruct best observed buffer entry {entry.state_id}.",
            )
        )
    if not operations:
        return None
    try:
        return apply_operations_to_train_text(seed_text, operations, schema)
    except Exception:
        return None


def generate_and_score_candidates(
    args: argparse.Namespace,
    schema: OperationSchema,
    base: BaseTrain,
    gp: GPSurrogate,
    entries: list[BufferEntry],
    *,
    seen_signatures: set[str],
) -> tuple[list[ScoredCandidate], dict[str, Any]]:
    rng = random.Random(int(args.seed))
    current = extract_top_level_assignment_values(base.text)
    base_params = {
        parameter.name: current[parameter.name]
        for parameter in schema.parameters.values()
        if parameter.name in current
    }
    observed_values = collect_observed_values(entries, schema)
    top_entries = sorted(entries, key=lambda entry: entry.score, reverse=bool(args.maximize))[:16]
    candidates: list[ScoredCandidate] = []
    candidate_signatures: set[str] = set()
    stats = {
        "attempted": 0,
        "valid": 0,
        "duplicate": 0,
        "seen": 0,
        "invalid": 0,
        "samples": max(0, int(args.samples)),
    }

    def add(operations: list[ValidatedOperation], source: str) -> None:
        stats["attempted"] += 1
        try:
            candidate = make_candidate(
                base_params,
                operations,
                schema,
                source=source,
            )
        except Exception:
            stats["invalid"] += 1
            return
        if candidate.signature in candidate_signatures:
            stats["duplicate"] += 1
            return
        if not args.allow_seen and candidate.signature in seen_signatures:
            stats["seen"] += 1
            return
        candidate_signatures.add(candidate.signature)
        candidates.append(
            ScoredCandidate(
                operations=candidate.operations,
                vector=candidate.vector,
                params=candidate.params,
                prediction=Prediction(mean=0.0, std=0.0, ei=0.0, lcb=0.0, selection_score=0.0),
                source=candidate.source,
                signature=candidate.signature,
            )
        )
        stats["valid"] += 1

    add_deterministic_candidates(add, schema, current, observed_values, top_entries, max_operations=args.max_operations)
    for sample_index in range(max(0, int(args.samples))):
        operations = sample_candidate_operations(
            schema,
            current,
            observed_values,
            max_operations=max(1, int(args.max_operations)),
            rng=rng,
        )
        if operations:
            add(operations, f"random_sample_{sample_index + 1}")
    stats["returned"] = len(candidates)
    batch_predictions = predict_many(
        gp,
        [candidate.vector for candidate in candidates],
        mode=args.surrogate_mode,
        beta=args.gp_beta,
        xi=args.gp_xi,
    )
    for candidate, prediction in zip(candidates, batch_predictions):
        candidate.prediction = prediction
    return candidates, stats


def collect_observed_values(entries: list[BufferEntry], schema: OperationSchema) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {name: [] for name in schema.parameters}
    seen: dict[str, set[str]] = {name: set() for name in schema.parameters}
    for entry in entries:
        if not isinstance(entry.params, dict):
            continue
        for parameter in schema.parameters.values():
            if parameter.name not in entry.params:
                continue
            try:
                value = validate_operation_value(entry.params[parameter.name], parameter, index=1)
            except ValueError:
                continue
            key = canonical_value(value)
            if key in seen[parameter.name]:
                continue
            seen[parameter.name].add(key)
            values[parameter.name].append(value)
    return values


def add_deterministic_candidates(
    add: Any,
    schema: OperationSchema,
    current: dict[str, Any],
    observed_values: dict[str, list[Any]],
    top_entries: list[BufferEntry],
    *,
    max_operations: int,
) -> None:
    for parameter in schema.parameters.values():
        if parameter.name not in current:
            continue
        for value in deterministic_values(parameter, current.get(parameter.name), observed_values.get(parameter.name, [])):
            operation = make_operation(parameter, value, rationale="Deterministic GP proposal sweep.")
            add([operation], f"deterministic:{parameter.name}")

    if max_operations <= 1:
        return
    for entry in top_entries:
        operations: list[ValidatedOperation] = []
        for parameter in schema.parameters.values():
            if parameter.name not in current:
                continue
            if parameter.name not in entry.params:
                continue
            try:
                value = validate_operation_value(entry.params[parameter.name], parameter, index=len(operations) + 1)
            except ValueError:
                continue
            if current.get(parameter.name) is not None and choice_values_equal(current.get(parameter.name), value):
                continue
            operations.append(make_operation(parameter, value, rationale=f"Reuse value from observed {entry.state_id}."))
            if len(operations) >= max_operations:
                break
        if operations:
            add(operations, f"top_observed:{entry.state_id}")


def deterministic_values(parameter: Any, current_value: Any, observed: list[Any]) -> list[Any]:
    candidates: list[Any] = []
    candidates.extend(observed[:12])
    if parameter.kind == "choice":
        candidates.extend(parameter.choices)
    elif parameter.kind == "int":
        lo = int(math.ceil(float(parameter.min_value)))
        hi = int(math.floor(float(parameter.max_value)))
        current = as_float(current_value)
        midpoint = int(round((lo + hi) / 2.0))
        candidates.extend([lo, hi, midpoint])
        if current is not None:
            candidates.extend([int(max(lo, round(current) - 1)), int(min(hi, round(current) + 1))])
    else:
        lo = float(parameter.min_value)
        hi = float(parameter.max_value)
        if parameter.scale == "log" and lo > 0 and hi > 0:
            midpoint = math.exp((math.log(lo) + math.log(hi)) / 2.0)
        else:
            midpoint = (lo + hi) / 2.0
        candidates.extend([lo, hi, midpoint])
        current = as_float(current_value)
        if current is not None:
            if current > 0:
                candidates.extend([max(lo, current * 0.5), min(hi, current * 1.5)])
            candidates.append(fallback_operation_value(current_value, parameter))

    output: list[Any] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            value = validate_operation_value(candidate, parameter, index=1)
        except ValueError:
            continue
        if current_value is not None and choice_values_equal(current_value, value):
            continue
        key = canonical_value(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def sample_candidate_operations(
    schema: OperationSchema,
    current: dict[str, Any],
    observed_values: dict[str, list[Any]],
    *,
    max_operations: int,
    rng: random.Random,
) -> list[ValidatedOperation]:
    parameters = [parameter for parameter in schema.parameters.values() if parameter.name in current]
    if not parameters:
        return []
    rng.shuffle(parameters)
    operation_count = rng.randint(1, max(1, min(max_operations, len(parameters))))
    operations: list[ValidatedOperation] = []
    for parameter in parameters:
        value: Any
        observed = observed_values.get(parameter.name) or []
        use_observed = bool(observed) and rng.random() < 0.35
        if use_observed:
            value = rng.choice(observed)
        else:
            value = sample_random_parameter_value(parameter, current.get(parameter.name), rng)
        try:
            value = validate_operation_value(value, parameter, index=len(operations) + 1)
        except ValueError:
            continue
        if current.get(parameter.name) is not None and choice_values_equal(current.get(parameter.name), value):
            continue
        operations.append(make_operation(parameter, value, rationale="Random GP acquisition candidate."))
        if len(operations) >= operation_count:
            break
    if operations:
        return operations
    for parameter in parameters:
        value = fallback_operation_value(current.get(parameter.name), parameter)
        if current.get(parameter.name) is None or not choice_values_equal(current.get(parameter.name), value):
            return [make_operation(parameter, value, rationale="Fallback GP acquisition candidate.")]
    return []


def make_operation(parameter: Any, value: Any, *, rationale: str) -> ValidatedOperation:
    return ValidatedOperation(
        name=parameter.name,
        op="set_choice" if parameter.kind == "choice" else "set_numeric",
        value=value,
        rationale=rationale,
    )


def make_candidate(
    base_params: dict[str, Any],
    operations: list[ValidatedOperation],
    schema: OperationSchema,
    *,
    source: str,
) -> Candidate:
    if not operations:
        raise ValueError("proposal has no operations")
    if len({operation.name for operation in operations}) != len(operations):
        raise ValueError("proposal repeats an operation name")
    params = dict(base_params)
    for operation in operations:
        current_value = params.get(operation.name)
        if current_value is not None and choice_values_equal(current_value, operation.value):
            raise ValueError("proposal operation is a no-op")
        params[operation.name] = operation.value
    vector = operation_vector_from_params(params, schema)
    return Candidate(
        operations=operations,
        vector=vector,
        params=params,
        source=source,
        signature=params_signature(params, schema),
    )


def predict_many(
    gp: GPSurrogate,
    vectors: list[list[float]],
    *,
    mode: str,
    beta: float,
    xi: float,
) -> list[Prediction]:
    if not vectors:
        return []
    if not gp.entries:
        means = np.full(len(vectors), float(gp.prior_score), dtype=float)
        stds = np.full(len(vectors), float(gp.prior_std), dtype=float)
    elif not gp.ready:
        scores = np.array([entry.score for entry in gp.entries], dtype=float)
        means = np.full(len(vectors), float(scores.mean()), dtype=float)
        stds = np.full(len(vectors), float(max(scores.std(), gp.prior_std)), dtype=float)
    else:
        X = np.array(vectors, dtype=float)
        Xz = (X - gp.x_mean) / gp.x_std
        Ks = rbf_kernel_many(Xz, gp.Xz, gp.lengthscale)
        mu_z = np.asarray(Ks @ gp.alpha, dtype=float).reshape(-1)
        V = np.linalg.solve(gp.L, Ks.T)
        var_z = np.maximum(1.0 - np.sum(V * V, axis=0), 1e-9)
        means = gp.y_mean + mu_z * gp.y_std
        stds = np.sqrt(var_z) * gp.y_std

    observed = [entry.score for entry in gp.entries]
    predictions: list[Prediction] = []
    for mean_raw, std_raw in zip(means, stds):
        mean = float(mean_raw)
        std = float(std_raw)
        best = min(observed) if gp.minimize and observed else max(observed) if observed else mean
        ei = expected_improvement(mean, std, best, minimize=gp.minimize, xi=xi)
        lcb = mean - beta * std if gp.minimize else mean + beta * std
        if mode == "mean":
            selection_score = mean if gp.minimize else -mean
        elif mode == "ei":
            selection_score = -ei
        else:
            selection_score = lcb if gp.minimize else -lcb
        predictions.append(
            Prediction(
                mean=float(mean),
                std=float(std),
                ei=float(ei),
                lcb=float(lcb),
                selection_score=float(selection_score),
            )
        )
    return predictions


def rbf_kernel_many(left: np.ndarray, right: np.ndarray, lengthscale: float) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=-1) / (float(lengthscale) * float(lengthscale)))


def materialize_top_proposals(
    base_text: str,
    candidates: list[ScoredCandidate],
    schema: OperationSchema,
    *,
    max_count: int,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.signature in seen:
            continue
        seen.add(candidate.signature)
        try:
            apply_result = apply_operations_to_train_text(base_text, candidate.operations, schema)
        except Exception:
            continue
        vector, params, source_hash = featurize_operation_text(apply_result.text, schema)
        proposals.append(
            Proposal(
                operations=candidate.operations,
                apply_result=apply_result,
                vector=vector,
                params=params,
                source_hash=source_hash,
                prediction=candidate.prediction,
                source=candidate.source,
                signature=candidate.signature,
            )
        )
        if len(proposals) >= max_count:
            break
    return proposals


def featurize_operation_text(text: str, schema: OperationSchema) -> tuple[list[float], dict[str, Any], str]:
    source_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    values = extract_top_level_assignment_values(text)
    vector = operation_vector_from_params(values, schema)
    params = {name: values[name] for name in schema.parameters if name in values}
    return vector, params, source_hash


def operation_vector_from_params(values: dict[str, Any], schema: OperationSchema) -> list[float]:
    vector: list[float] = []
    for parameter in schema.parameters.values():
        present = parameter.name in values
        raw_value = values.get(parameter.name)
        if parameter.kind == "choice":
            vector.extend(
                1.0 if present and choice_values_equal(raw_value, choice) else 0.0
                for choice in parameter.choices
            )
            vector.append(1.0 if present else 0.0)
            continue
        value = as_float(raw_value)
        if value is None:
            vector.append(0.0)
            vector.append(0.0)
            continue
        vector.append(normalize_operation_numeric(value, parameter))
        vector.append(1.0)
    return vector


def params_signature(params: dict[str, Any], schema: OperationSchema) -> str:
    payload = []
    for parameter in schema.parameters.values():
        value = params.get(parameter.name)
        if parameter.kind == "int":
            number = as_float(value)
            value = None if number is None else int(round(number))
        elif parameter.kind == "float":
            number = as_float(value)
            value = None if number is None else round(float(number), 12)
        payload.append([parameter.name, value])
    return json.dumps(payload, sort_keys=False, separators=(",", ":"))


def canonical_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return json.dumps(value, sort_keys=True)


def write_outputs(
    args: argparse.Namespace,
    out_dir: Path,
    run_dir: Path | None,
    buffer_path: Path,
    schema: OperationSchema,
    entries: list[BufferEntry],
    gp: GPSurrogate,
    base: BaseTrain,
    best: Proposal,
    top_k: list[Proposal],
    stats: dict[str, Any],
) -> None:
    (out_dir / "proposal_train.py").write_text(best.apply_result.text, encoding="utf-8")
    (out_dir / "proposal_patch.diff").write_text(best.apply_result.patch, encoding="utf-8")
    if base.reconstruction is not None:
        (out_dir / "base_train.py").write_text(base.text, encoding="utf-8")
        (out_dir / "base_reconstruction_patch.diff").write_text(base.reconstruction.patch, encoding="utf-8")
        (out_dir / "base_reconstruction_operations.json").write_text(
            json.dumps(
                {
                    "summary": operation_summary(
                        [
                            ValidatedOperation(
                                name=record["name"],
                                op=record["op"],
                                value=record["new_value"],
                                rationale=str(record.get("rationale") or ""),
                            )
                            for record in base.reconstruction.records
                        ],
                        base.reconstruction.records,
                    ),
                    "applied": base.reconstruction.records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    best_payload = proposal_payload(best, rank=1, schema=schema)
    best_payload.update(
        {
            "base_path": str(base.path),
            "base_source": base.source,
            "schema_version": schema.version,
            "buffer": str(buffer_path),
        }
    )
    (out_dir / "proposal_operations.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_ranked_tsv(out_dir / "ranked_proposals.tsv", top_k, args.score_key)
    write_top_candidate_files(out_dir / "candidates", top_k, schema)

    summary = {
        "input_run_dir": None if run_dir is None else str(run_dir),
        "buffer": str(buffer_path),
        "schema": None if schema.path is None else str(schema.path),
        "schema_version": schema.version,
        "feature_version": operation_feature_version(schema),
        "entries_loaded": len(entries),
        "score_key": args.score_key,
        "maximize": bool(args.maximize),
        "gp": gp.summary(),
        "acquisition": {
            "mode": args.surrogate_mode,
            "beta": args.gp_beta,
            "xi": args.gp_xi,
        },
        "sampling": stats,
        "base": {
            "path": str(base.path),
            "source": base.source,
            "best_entry_state_id": None if base.best_entry is None else base.best_entry.state_id,
            "best_entry_score": None if base.best_entry is None else base.best_entry.score,
            "reconstructed_from_params": base.reconstruction is not None,
        },
        "best": best_payload,
        "top_k": [proposal_payload(proposal, rank=index, schema=schema) for index, proposal in enumerate(top_k, start=1)],
        "outputs": {
            "proposal_train": str(out_dir / "proposal_train.py"),
            "proposal_patch": str(out_dir / "proposal_patch.diff"),
            "proposal_operations": str(out_dir / "proposal_operations.json"),
            "ranked_proposals": str(out_dir / "ranked_proposals.tsv"),
            "candidate_dir": str(out_dir / "candidates"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def proposal_payload(proposal: Proposal, *, rank: int, schema: OperationSchema) -> dict[str, Any]:
    return {
        "rank": rank,
        "source": proposal.source,
        "prediction": prediction_to_json(proposal.prediction),
        "summary": operation_summary(proposal.operations, proposal.apply_result.records),
        "signature": proposal.signature,
        "source_hash": proposal.source_hash,
        "params": proposal.params,
        "operations": [
            {
                "name": operation.name,
                "op": operation.op,
                "value": operation.value,
                "value_text": format_operation_value(operation.value, schema.parameters[operation.name]),
                "rationale": operation.rationale,
            }
            for operation in proposal.operations
        ],
        "applied": proposal.apply_result.records,
    }


def prediction_to_json(prediction: Any) -> dict[str, float]:
    return {
        "mean": float(prediction.mean),
        "std": float(prediction.std),
        "ei": float(prediction.ei),
        "lcb": float(prediction.lcb),
        "selection_score": float(prediction.selection_score),
    }


def write_ranked_tsv(path: Path, proposals: list[Proposal], score_key: str) -> None:
    columns = [
        "rank",
        "selection_score",
        f"predicted_{score_key}",
        "std",
        "ei",
        "lcb",
        "source",
        "operations",
    ]
    lines = ["\t".join(columns)]
    for rank, proposal in enumerate(proposals, start=1):
        prediction = proposal.prediction
        lines.append(
            "\t".join(
                [
                    str(rank),
                    f"{float(prediction.selection_score):.10g}",
                    f"{float(prediction.mean):.10g}",
                    f"{float(prediction.std):.10g}",
                    f"{float(prediction.ei):.10g}",
                    f"{float(prediction.lcb):.10g}",
                    clean_tsv(proposal.source),
                    clean_tsv(operation_summary(proposal.operations, proposal.apply_result.records)),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_top_candidate_files(path: Path, proposals: list[Proposal], schema: OperationSchema) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for rank, proposal in enumerate(proposals, start=1):
        stem = f"candidate_{rank:03d}"
        (path / f"{stem}_train.py").write_text(proposal.apply_result.text, encoding="utf-8")
        (path / f"{stem}_patch.diff").write_text(proposal.apply_result.patch, encoding="utf-8")
        (path / f"{stem}_operations.json").write_text(
            json.dumps(proposal_payload(proposal, rank=rank, schema=schema), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def clean_tsv(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
