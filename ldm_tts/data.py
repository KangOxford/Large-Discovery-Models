"""Unified data collection, augmentation, and rendering for LDM-TTS.

Runtime task code records accepted teacher actions through ``DataCollectionSink``.
Offline data preparation can then add expert justifications through
``ExpertJustificationPipeline``. Both workflows operate on the same ldm-2.0
intermediate representation so augmented reasoning remains attached to the
action it explains and rendered SFT data can always be regenerated from IR.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ldm_tts.data_collection import (
    DataCollectionPaths,
    DataCollectionSink,
    LDMDataCollectionError,
    append_jsonl,
    dataset_info_payload,
    jdump,
    make_complete_design_ir,
    make_parameter_edit_ir,
    normalize_task_id,
    read_jsonl,
    render_prose,
    render_record,
    smallmol_ir_from_prompt_response,
    smallmol_irs_from_round_record,
    validate_ir_record,
)

EXPERT_JUSTIFICATION_SYSTEM_PROMPT = (
    "You are an expert scientific search assistant. Given the exact context and "
    "the accepted action, write a concise, faithful, first-person justification "
    "that explains how the visible evidence supports that action. Use concrete "
    "analysis, decomposition, or trade-offs. Do not claim access to hidden scores "
    "or outcomes, do not mention that an answer was supplied, and do not repeat "
    "the final action. Return only the justification, without tags or headings."
)

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OUTER_THINK_TAGS = re.compile(
    r"^\s*<think>\s*|\s*</think>\s*$", re.IGNORECASE
)


@dataclass(frozen=True)
class JustificationRequest:
    """The model-visible context and accepted target an expert must justify."""

    instruction: str
    reference_answer: str
    source: str | None = None

    @property
    def user_prompt(self) -> str:
        source_line = f"\nSource task: {self.source}" if self.source else ""
        return (
            "## Model-visible context\n"
            f"{self.instruction}\n\n"
            "## Accepted action or answer\n"
            f"{self.reference_answer}"
            f"{source_line}\n\n"
            "Explain why this action or answer follows from the visible context."
        )


class ExpertJustifier(Protocol):
    """Port implemented by production model clients and test adapters."""

    def justify(self, request: JustificationRequest) -> str:
        """Return one justification for ``request``."""


class OpenAICompatibleExpert:
    """Expert adapter for OpenAI-compatible chat-completions endpoints.

    ``openai`` is imported lazily so collection and local data processing remain
    dependency-light when expert augmentation is not being used.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.7,
        system_prompt: str = EXPERT_JUSTIFICATION_SYSTEM_PROMPT,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "expert augmentation requires openai>=1.0; install it in the "
                "task environment before running the augmentation CLI"
            ) from exc

        self.model = model
        self.temperature = float(temperature)
        self.system_prompt = system_prompt
        self.cache_identity = hashlib.sha256(
            jdump(
                {
                    "adapter": type(self).__name__,
                    "base_url": base_url,
                    "model": model,
                    "system_prompt": system_prompt,
                    "temperature": self.temperature,
                }
            ).encode("utf-8")
        ).hexdigest()
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def justify(self, request: JustificationRequest) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()


@dataclass(frozen=True)
class AugmentationReport:
    """Observable result of one expert-justification run."""

    total: int
    written: int
    generated: int
    resumed: int
    skipped_existing: int
    skipped_unavailable: int
    failed_indices: tuple[int, ...]
    output_path: Path
    checkpoint_path: Path
    sft_output_path: Path | None = None

    @property
    def failed(self) -> int:
        return len(self.failed_indices)


@dataclass(frozen=True)
class _PreparedRecord:
    index: int
    record_id: str
    request: JustificationRequest
    target_kind: str


class ExpertJustificationPipeline:
    """Add expert reasoning to ldm-2.0 IR or Alpaca rows with safe resume.

    The preferred input is ldm-2.0 IR. For compatibility with existing datasets,
    Alpaca rows are also accepted: JSON outputs with a ``reasoning`` field are
    updated structurally, while plain-text outputs receive a leading ``<think>``
    block. Output is always JSONL, matching the repository's training pipeline.
    """

    def __init__(
        self,
        expert: ExpertJustifier,
        *,
        workers: int = 4,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        overwrite_reasoning: bool = False,
        include_reasoning_unavailable: bool = False,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        self.expert = expert
        self.workers = workers
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.overwrite_reasoning = overwrite_reasoning
        self.include_reasoning_unavailable = include_reasoning_unavailable
        self.expert_name = str(
            getattr(expert, "model", type(expert).__qualname__)
        )
        self.expert_identity = str(
            getattr(expert, "cache_identity", type(expert).__qualname__)
        )

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        limit: int | None = None,
        checkpoint_path: str | Path | None = None,
        sft_output_path: str | Path | None = None,
        render_mode: str = "prose",
        include_parent_artifact: bool = True,
    ) -> AugmentationReport:
        """Augment a dataset, write JSONL atomically, and return a run report."""

        source = Path(input_path)
        output = Path(output_path)
        if _same_path(source, output):
            raise LDMDataCollectionError("input and output paths must be different")
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1 when provided")

        loaded = read_jsonl(source)
        rows = loaded[:limit] if limit is not None else loaded
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise LDMDataCollectionError(f"record {index} must be a JSON object")

        checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else Path(str(output) + ".checkpoint.jsonl")
        )
        sft_output = Path(sft_output_path) if sft_output_path is not None else None
        protected_paths = [source, output]
        if sft_output is not None:
            protected_paths.append(sft_output)
        for protected in protected_paths:
            if _same_path(checkpoint, protected):
                raise LDMDataCollectionError(
                    "checkpoint path must differ from input and output paths"
                )
        if sft_output is not None:
            if _same_path(source, sft_output) or _same_path(output, sft_output):
                raise LDMDataCollectionError(
                    "SFT output path must differ from input and augmented IR output"
                )
            if any(not _is_ir(row) for row in rows):
                raise LDMDataCollectionError(
                    "--sft-output requires every input record to be ldm-2.0 IR"
                )
            if render_mode not in {"prose", "json"}:
                raise LDMDataCollectionError(
                    f"unsupported render mode {render_mode!r}"
                )
        done = _load_checkpoint(checkpoint)
        result = [copy.deepcopy(dict(row)) for row in rows]
        prepared: list[_PreparedRecord] = []
        skipped_existing = 0
        skipped_unavailable = 0

        for index, row in enumerate(result):
            item, skip_reason = self._prepare(index, row)
            if skip_reason == "existing":
                skipped_existing += 1
            elif skip_reason == "unavailable":
                skipped_unavailable += 1
            elif item is not None:
                prepared.append(item)

        resumed = 0
        todo: list[_PreparedRecord] = []
        for item in prepared:
            reasoning = done.get((self.expert_identity, item.record_id))
            if reasoning:
                _apply_reasoning(
                    result[item.index],
                    item.target_kind,
                    reasoning,
                    expert_name=self.expert_name,
                )
                resumed += 1
            else:
                todo.append(item)

        generated = 0
        failed_indices: list[int] = []
        if todo:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(self._generate, item.request): item for item in todo
                }
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        reasoning = future.result()
                    except Exception:  # noqa: BLE001 - failure is recorded per row
                        failed_indices.append(item.index)
                        continue
                    _apply_reasoning(
                        result[item.index],
                        item.target_kind,
                        reasoning,
                        expert_name=self.expert_name,
                    )
                    done[(self.expert_identity, item.record_id)] = reasoning
                    append_jsonl(
                        checkpoint,
                        {
                            "expert_id": self.expert_identity,
                            "record_id": item.record_id,
                            "reasoning": reasoning,
                        },
                    )
                    generated += 1

        rendered: list[dict[str, Any]] | None = None
        if sft_output is not None:
            rendered = [
                render_record(
                    row,
                    mode=render_mode,
                    include_parent_artifact=include_parent_artifact,
                )
                for row in result
            ]

        _write_jsonl_atomic(output, result)
        if sft_output is not None and rendered is not None:
            _write_jsonl_atomic(sft_output, rendered)

        return AugmentationReport(
            total=len(loaded),
            written=len(result),
            generated=generated,
            resumed=resumed,
            skipped_existing=skipped_existing,
            skipped_unavailable=skipped_unavailable,
            failed_indices=tuple(sorted(failed_indices)),
            output_path=output,
            checkpoint_path=checkpoint,
            sft_output_path=sft_output,
        )

    def _prepare(
        self, index: int, row: Mapping[str, Any]
    ) -> tuple[_PreparedRecord | None, str | None]:
        record_id = _record_id(index, row)
        if _is_ir(row):
            validate_ir_record(row)
            task = row["task"]
            action = row["action"]
            if (
                task.get("reasoning_available") is False
                and not self.include_reasoning_unavailable
            ):
                return None, "unavailable"
            if _has_text(action.get("reasoning")) and not self.overwrite_reasoning:
                return None, "existing"
            reference = copy.deepcopy(dict(action))
            reference["reasoning"] = None
            request = JustificationRequest(
                instruction=render_prose(row),
                reference_answer=jdump(reference, indent=1),
                source=normalize_task_id(str(task.get("id", ""))) or None,
            )
            return _PreparedRecord(index, record_id, request, "ir"), None

        instruction = row.get("instruction")
        output = row.get("output")
        if not isinstance(instruction, str) or not isinstance(output, str):
            raise LDMDataCollectionError(
                f"record {index} must be ldm-2.0 IR or an Alpaca row with "
                "string instruction and output fields"
            )
        source = str(row.get("source") or "").strip() or None
        if (
            normalize_task_id(source or "") == "protein"
            and not self.include_reasoning_unavailable
        ):
            return None, "unavailable"

        parsed_output = _json_object_or_none(output)
        if parsed_output is not None and "reasoning" in parsed_output:
            if (
                _has_text(parsed_output.get("reasoning"))
                and not self.overwrite_reasoning
            ):
                return None, "existing"
            reference = dict(parsed_output)
            reference["reasoning"] = None
            target_kind = "json_output"
            reference_answer = jdump(reference, indent=1)
        else:
            if _THINK_BLOCK.match(output) and not self.overwrite_reasoning:
                return None, "existing"
            target_kind = "think"
            reference_answer = _THINK_BLOCK.sub("", output, count=1)

        input_text = row.get("input")
        context = instruction
        if isinstance(input_text, str) and input_text.strip():
            context += f"\n\n## Additional input\n{input_text}"
        request = JustificationRequest(
            instruction=context,
            reference_answer=reference_answer,
            source=source,
        )
        return _PreparedRecord(index, record_id, request, target_kind), None

    def _generate(self, request: JustificationRequest) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                reasoning = _clean_reasoning(self.expert.justify(request))
                if not reasoning:
                    raise ValueError("expert model returned an empty justification")
                return reasoning
            except Exception as exc:  # noqa: BLE001 - retry external model errors
                last_error = exc
                if attempt < self.max_retries and self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * attempt)
        assert last_error is not None
        raise last_error


def _is_ir(row: Mapping[str, Any]) -> bool:
    return row.get("schema_version") == "ldm-2.0"


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _json_object_or_none(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _clean_reasoning(reasoning: str) -> str:
    if not isinstance(reasoning, str):
        raise TypeError("expert justification must be a string")
    return _OUTER_THINK_TAGS.sub("", reasoning.strip()).strip()


def _apply_reasoning(
    row: dict[str, Any],
    target_kind: str,
    reasoning: str,
    *,
    expert_name: str,
) -> None:
    if target_kind == "ir":
        row["action"]["reasoning"] = reasoning
        collection = row.setdefault("collection", {})
        if isinstance(collection, dict):
            collection["augmentation"] = {
                "kind": "expert_justification",
                "expert": expert_name,
            }
        return
    if target_kind == "json_output":
        output = json.loads(row["output"])
        output["reasoning"] = reasoning
        row["output"] = jdump(output)
        return
    if target_kind == "think":
        answer = _THINK_BLOCK.sub("", row["output"], count=1)
        row["output"] = f"<think>\n{reasoning}\n</think>\n\n{answer}"
        return
    raise AssertionError(f"unknown augmentation target {target_kind!r}")


def _record_id(index: int, row: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{index}:{hashlib.sha256(canonical).hexdigest()}"


def _load_checkpoint(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    done: dict[tuple[str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(row, Mapping)
            and _has_text(row.get("expert_id"))
            and _has_text(row.get("record_id"))
            and _has_text(row.get("reasoning"))
        ):
            key = (str(row["expert_id"]), str(row["record_id"]))
            done[key] = str(row["reasoning"])
    return done


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(jdump(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


__all__ = [
    "AugmentationReport",
    "DataCollectionPaths",
    "DataCollectionSink",
    "EXPERT_JUSTIFICATION_SYSTEM_PROMPT",
    "ExpertJustificationPipeline",
    "ExpertJustifier",
    "JustificationRequest",
    "LDMDataCollectionError",
    "OpenAICompatibleExpert",
    "dataset_info_payload",
    "make_complete_design_ir",
    "make_parameter_edit_ir",
    "normalize_task_id",
    "read_jsonl",
    "render_prose",
    "render_record",
    "smallmol_ir_from_prompt_response",
    "smallmol_irs_from_round_record",
    "validate_ir_record",
]
