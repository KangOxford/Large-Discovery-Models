from __future__ import annotations

import asyncio
import difflib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from ldm_tts.contracts.evaluation import is_finite_number
from ldm_tts.engine.run_store import JsonlTrajectoryRecorder


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


DEFAULT_TASK_CONTEXT = """Autoresearch task context:
- `train.py` is the single editable file. It contains the full GPT model, optimizer, and training loop.
- Everything inside `train.py` is fair game: architecture, hyperparameters, optimizer details, batch size, schedule, and training-loop choices.
- The benchmark runs training for a fixed 30 second wall-clock budget, excluding startup/compilation. Edits should improve performance under that same fixed budget, not by assuming longer training.
- The primary metric is `val_bpb` validation bits per byte. Lower is better, and this metric is intended to be vocab-size independent.
- Keep the script self-contained. Do not require new data files, distributed training, complex configs, or extra dependencies beyond the existing PyTorch-oriented setup.
- Because only one GPU, one file, one metric, and one fixed time budget are assumed, prefer robust changes that are likely to execute successfully on the local platform."""


@dataclass
class SearchConfig:
    project_root: Path
    seed_train_path: Path
    out_dir: Path
    eval_command: str = "uv run python {train_path}"
    eval_shell: bool = False
    score_key: str = "val_bpb"
    minimize: bool = True
    failure_score: float = 1.0e9
    timeout_seconds: int = 900
    run_evaluation: bool = True
    generator: str = "api"
    llm_url: str = ""
    llm_model_name: str = ""
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    stop: list[str] | None = None
    disable_thinking: bool = False
    concurrency: int = 1
    prompt_max_chars: int = 100_000
    task_context: str = DEFAULT_TASK_CONTEXT
    extra_instruction: str = ""
    feedback_context: str = ""
    show_progress: bool = True
    progress_width: int = 28
    response_log_chars: int = 20_000
    num_edits_per_step: int = 1
    eval_each_num_steps: int = 1
    request_logprobs: bool = False
    top_logprobs: int | None = None
    debug: bool = False
    debug_log_path: Path | None = None

    def normalized(self) -> "SearchConfig":
        self.project_root = self.project_root.resolve()
        self.seed_train_path = self.seed_train_path.resolve()
        self.out_dir = self.out_dir.resolve()
        if self.debug_log_path is not None:
            self.debug_log_path = self.debug_log_path.resolve()
        self.num_edits_per_step = max(1, int(self.num_edits_per_step))
        self.eval_each_num_steps = max(1, int(self.eval_each_num_steps))
        if self.top_logprobs is not None:
            self.top_logprobs = max(0, int(self.top_logprobs))
        return self


@dataclass
class SearchState:
    state_id: str
    parent_id: str | None
    depth: int
    workdir: Path
    train_path: Path
    status: str = "created"
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    token_usage: int = 0
    token_usage_detail: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str | None = None
    description: str = ""
    prompt_path: Path | None = None
    response_path: Path | None = None
    patch_path: Path | None = None
    llm_response: str = ""
    llm_response_truncated: bool = False
    edits: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["workdir"] = str(self.workdir)
        data["train_path"] = str(self.train_path)
        data["prompt_path"] = None if self.prompt_path is None else str(self.prompt_path)
        data["response_path"] = None if self.response_path is None else str(self.response_path)
        data["patch_path"] = None if self.patch_path is None else str(self.patch_path)
        return data


class SearchEngine:
    def __init__(self, config: SearchConfig):
        self.config = config.normalized()
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        (self.config.out_dir / "states").mkdir(exist_ok=True)
        self.states: list[SearchState] = []
        self._counter = 0
        # asyncio primitives are bound to the active loop on Python 3.9.
        # SearchEngine is constructed synchronously, so create the semaphore
        # lazily when generation first runs inside an event loop.
        self._generation_sem: asyncio.Semaphore | None = None
        self._generation_loop: asyncio.AbstractEventLoop | None = None
        self.evaluation_count = 0
        self.manifest_path = self.config.out_dir / "manifest.jsonl"
        self._manifest_recorder = JsonlTrajectoryRecorder(
            self.config.out_dir,
            rounds_filename="manifest.jsonl",
            sort_keys=True,
        )
        self._progress: ProgressBar | None = None
        self._progress_count = 0
        self._debug_started_at = time.time()
        if self.config.debug:
            self.debug_event(
                "engine_init",
                out_dir=self.config.out_dir,
                project_root=self.config.project_root,
                generator=self.config.generator,
            )

    @property
    def evaluation_interval(self) -> int:
        """Depth interval used by task-neutral search traversal."""

        return max(1, int(self.config.eval_each_num_steps))

    def start_progress(self, total: int, *, label: str = "search") -> None:
        if not self.config.show_progress or total <= 0:
            return
        self._progress_count = 0
        self._progress = ProgressBar(total=total, label=label, width=self.config.progress_width)
        self._progress.update(0, status="starting")

    def finish_progress(self) -> None:
        if self._progress is not None:
            best = self.best_state()
            self._progress.finish(
                self._progress_count,
                best_score=None if best is None else best.score,
                status="done",
            )
            self._progress = None

    def create_seed_state(self) -> SearchState:
        state = self._new_state(parent=None, depth=0)
        shutil.copy2(self.config.seed_train_path, state.train_path)
        state.status = "seed"
        self._write_state_meta(state)
        self._record_manifest(state)
        return state

    async def expand_state(
        self,
        parent: SearchState,
        count: int,
        *,
        search_note: str = "",
    ) -> list[SearchState]:
        tasks = [
            self._generate_one(parent, child_index=i + 1, sibling_count=count, search_note=search_note)
            for i in range(count)
        ]
        return [state for state in await asyncio.gather(*tasks) if state is not None]

    def _get_generation_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._generation_sem is None or self._generation_loop is not loop:
            self._generation_sem = asyncio.Semaphore(max(1, self.config.concurrency))
            self._generation_loop = loop
        return self._generation_sem

    async def evaluate_many(self, states: Iterable[SearchState]) -> None:
        for state in states:
            self.evaluate_state(state)

    async def defer_evaluation_many(self, states: Iterable[SearchState], *, reason: str) -> None:
        for state in states:
            self.defer_evaluation(state, reason=reason)

    def should_evaluate_depth(self, depth: int, max_depth: int | None = None) -> bool:
        depth = int(depth)
        if depth <= 0:
            return True
        interval = self.evaluation_interval
        if depth % interval == 0:
            return True
        return max_depth is not None and depth >= max_depth

    def evaluation_depths(self, max_depth: int) -> list[int]:
        return [
            depth
            for depth in range(1, max(1, int(max_depth)) + 1)
            if self.should_evaluate_depth(depth, max(1, int(max_depth)))
        ]

    def defer_evaluation(self, state: SearchState, *, reason: str) -> None:
        if state.status == "generation_error":
            self._write_state_meta(state)
            self._record_manifest(state)
            return
        state.status = "evaluation_deferred"
        state.score = None
        state.error = reason
        self._write_state_meta(state)
        self._record_manifest(state)

    def evaluate_state(self, state: SearchState) -> None:
        if state.status == "generation_error":
            self.debug_event("evaluation_skip_generation_error", state_id=state.state_id, error=state.error)
            self._write_state_meta(state)
            self._record_manifest(state)
            self._advance_progress(state)
            return

        if not self.config.run_evaluation:
            self.debug_event("evaluation_skip_disabled", state_id=state.state_id)
            state.status = "evaluation_skipped"
            state.score = None
            state.error = "Evaluation skipped by --skip-eval."
            self._write_state_meta(state)
            self._record_manifest(state)
            self._advance_progress(state)
            return

        self.evaluation_count += 1
        diagnostics_path = state.workdir / "diagnostics.json"
        stdout_path = state.workdir / "stdout.log"
        stderr_path = state.workdir / "stderr.log"
        env = dict(os.environ)
        env["AUTORESEARCH_DIAGNOSTICS_JSON"] = str(diagnostics_path)
        project_python_paths = [self.config.project_root]
        scripts_dir = self.config.project_root / "scripts"
        if scripts_dir.is_dir():
            project_python_paths.insert(0, scripts_dir)
        inherited_python_paths = [
            Path(entry)
            for entry in env.get("PYTHONPATH", "").split(os.pathsep)
            if entry and Path(entry) not in project_python_paths
        ]
        env["PYTHONPATH"] = os.pathsep.join(
            str(path) for path in [*project_python_paths, *inherited_python_paths]
        )
        task_cache_dir = self.config.project_root / "cache" / "autoresearch"
        if task_cache_dir.is_dir() and not env.get("AUTORESEARCH_CACHE_DIR"):
            env["AUTORESEARCH_CACHE_DIR"] = str(task_cache_dir)

        command = self._format_eval_command(state, diagnostics_path)
        start = time.time()
        self.debug_event(
            "evaluation_start",
            state_id=state.state_id,
            command=command,
            cwd=self.config.project_root,
            timeout_seconds=self.config.timeout_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            diagnostics_path=diagnostics_path,
        )
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                completed = subprocess.run(
                    command,
                    cwd=self.config.project_root,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    shell=self.config.eval_shell,
                    check=False,
                )
            state.elapsed_seconds = time.time() - start
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
            metrics = parse_metrics(stdout_text + "\n" + stderr_text, diagnostics_path)
            state.metrics = metrics
            score = metrics.get(self.config.score_key)
            if score is not None and _is_finite_number(score):
                state.score = float(score)
                state.status = "evaluated" if completed.returncode == 0 else "evaluated_nonzero"
            else:
                state.score = self.config.failure_score
                state.status = "score_missing" if completed.returncode == 0 else "crash"
                state.error = f"Missing numeric score key {self.config.score_key!r}."
            if completed.returncode != 0 and state.error is None:
                state.error = f"Evaluation command exited with code {completed.returncode}."
            self.debug_event(
                "evaluation_complete",
                state_id=state.state_id,
                returncode=completed.returncode,
                elapsed_seconds=state.elapsed_seconds,
                status=state.status,
                score=state.score,
                error=state.error,
                metric_keys=sorted(metrics),
            )
        except subprocess.TimeoutExpired:
            state.elapsed_seconds = time.time() - start
            state.score = self.config.failure_score
            state.status = "timeout"
            state.error = f"Evaluation exceeded {self.config.timeout_seconds}s."
            self.debug_event(
                "evaluation_timeout",
                state_id=state.state_id,
                elapsed_seconds=state.elapsed_seconds,
                timeout_seconds=self.config.timeout_seconds,
            )
        except Exception as exc:
            state.elapsed_seconds = time.time() - start
            state.score = self.config.failure_score
            state.status = "evaluation_error"
            state.error = repr(exc)
            self.debug_event(
                "evaluation_error",
                state_id=state.state_id,
                elapsed_seconds=state.elapsed_seconds,
                error=repr(exc),
            )

        self._write_state_meta(state)
        self._record_manifest(state)
        self._advance_progress(state)

    def best_state(self, states: Iterable[SearchState] | None = None) -> SearchState | None:
        candidates = [
            state
            for state in (self.states if states is None else states)
            if state.score is not None and _is_finite_number(state.score)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda state: state.score, reverse=not self.config.minimize)[0]

    def ranked_states(self, states: Iterable[SearchState] | None = None) -> list[SearchState]:
        candidates = [
            state
            for state in (self.states if states is None else states)
            if state.score is not None and _is_finite_number(state.score)
        ]
        return sorted(candidates, key=lambda state: state.score, reverse=not self.config.minimize)

    def reward(self, state: SearchState) -> float:
        """Map a valid task score to a larger-is-better reward for MCTS."""

        failure_statuses = {
            "crash",
            "evaluation_error",
            "generation_error",
            "score_missing",
            "timeout",
        }
        if state.status in failure_statuses or not _is_finite_number(state.score):
            return 0.0
        scored = [
            float(candidate.score)
            for candidate in self.states
            if candidate.status not in failure_statuses and _is_finite_number(candidate.score)
        ]
        if not scored:
            return 0.0
        low = min(scored)
        high = max(scored)
        if high <= low:
            return 1.0
        score = float(state.score)
        reward = (high - score) / (high - low) if self.config.minimize else (score - low) / (high - low)
        return min(1.0, max(0.0, reward))

    def write_summary(self, *, method: str, args: dict[str, Any], best: SearchState | None) -> Path:
        summary = {
            "method": method,
            "args": jsonable(args),
            "best_state_id": None if best is None else best.state_id,
            "best_score": None if best is None else best.score,
            "score_key": self.config.score_key,
            "minimize": self.config.minimize,
            "evaluation_count": self.evaluation_count,
            "states": [state.to_json() for state in self.states],
        }
        path = self.config.out_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if best is not None:
            shutil.copy2(best.train_path, self.config.out_dir / "best_train.py")
        return path

    async def _generate_one(
        self,
        parent: SearchState,
        *,
        child_index: int,
        sibling_count: int,
        search_note: str,
    ) -> SearchState | None:
        state = self._new_state(parent=parent, depth=parent.depth + 1)
        num_edits = max(1, self.config.num_edits_per_step)
        current_text = parent.train_path.read_text(encoding="utf-8")
        prior_edits: list[dict[str, Any]] = []
        try:
            async with self._get_generation_semaphore():
                for edit_index in range(1, num_edits + 1):
                    self._progress_status(f"generating {state.state_id} edit {edit_index}/{num_edits}")
                    prompt = self._build_prompt(
                        parent,
                        child_index=child_index,
                        sibling_count=sibling_count,
                        search_note=search_note,
                        current_train_text=current_text,
                        edit_index=edit_index,
                        total_edits=num_edits,
                        prior_edits=prior_edits,
                    )
                    state.prompt_path = self._edit_artifact_path(state, "prompt", edit_index, num_edits, "md")
                    state.prompt_path.write_text(prompt, encoding="utf-8")

                    response, token_usage = await self._call_generator(prompt, state, current_text)
                    if response is None:
                        response = ""
                    if not isinstance(response, str):
                        response = str(response)
                    completion_tokens, token_usage_detail = normalize_token_usage(token_usage)
                    logprobs_payload = extract_generation_logprobs(token_usage)
                    logprob_summary = summarize_generation_logprobs(logprobs_payload)
                    logprobs_path = None
                    if logprobs_payload is not None:
                        logprobs_path = self._edit_artifact_path(state, "logprobs", edit_index, num_edits, "json")
                        logprobs_path.write_text(
                            json.dumps(logprobs_payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    state.token_usage += completion_tokens
                    state.token_usage_detail = add_token_usage(state.token_usage_detail, token_usage_detail)

                    state.response_path = self._edit_artifact_path(state, "response", edit_index, num_edits, "md")
                    state.response_path.write_text(response, encoding="utf-8")
                    if not response.strip():
                        raise ValueError(
                            f"Generator edit {edit_index}/{num_edits} returned an empty message. "
                            "The provider may have hit max_tokens, returned only hidden reasoning, or filtered content."
                        )
                    state.llm_response, state.llm_response_truncated = self._inline_llm_response(response)
                    tool_blocks = extract_tool_call_edit_blocks(response)
                    search_replace_blocks = tool_blocks or extract_search_replace_blocks(response)
                    edit_source = "tool_call" if tool_blocks else ("search_replace" if search_replace_blocks else "")
                    description = extract_description(response)
                    edit_record: dict[str, Any] | None = None
                    if search_replace_blocks:
                        edit_record = {
                            "edit_index": edit_index,
                            "total_edits": num_edits,
                            "description": description,
                            "prompt_path": str(state.prompt_path),
                            "response_path": str(state.response_path),
                            "patch_path": None,
                            "llm_response": state.llm_response,
                            "llm_response_truncated": state.llm_response_truncated,
                            "token_usage": completion_tokens,
                            "token_usage_detail": token_usage_detail,
                            "logprobs_path": None if logprobs_path is None else str(logprobs_path),
                            "logprob_summary": logprob_summary,
                            "edit_source": edit_source,
                            "detected_edit_count": len(search_replace_blocks),
                            "detected_edits": summarize_edit_blocks(search_replace_blocks),
                            "status": "detected",
                        }
                        state.edits.append(edit_record)
                    patch = None
                    replacement = None
                    fallback_edit_source = ""
                    edit_applied = False
                    if search_replace_blocks:
                        new_text = apply_search_replace_blocks(current_text, search_replace_blocks)
                        if new_text == current_text:
                            raise ValueError("SEARCH/REPLACE edits produced no changes.")
                        patch = make_unified_diff(
                            current_text,
                            new_text,
                            fromfile=f"{parent.state_id}/train.py",
                            tofile=f"{state.state_id}/train.py",
                        )
                        state.patch_path = self._edit_artifact_path(state, "patch", edit_index, num_edits, "diff")
                        state.patch_path.write_text(patch, encoding="utf-8")
                        edit_applied = True
                    else:
                        patch = extract_unified_diff(response)
                        fallback_edit_source = "unified_diff" if patch else ""
                        replacement = None if patch else extract_replacement_train_file(response)
                    if patch and not search_replace_blocks:
                        state.patch_path = self._edit_artifact_path(state, "patch", edit_index, num_edits, "diff")
                        state.patch_path.write_text(patch, encoding="utf-8")
                        new_text = apply_unified_diff_to_text(current_text, patch)
                        edit_applied = True
                    elif replacement and not search_replace_blocks:
                        new_text = replacement
                        fallback_edit_source = "complete_file"
                        patch = make_unified_diff(
                            current_text,
                            new_text,
                            fromfile=f"{parent.state_id}/train.py",
                            tofile=f"{state.state_id}/train.py",
                        )
                        state.patch_path = self._edit_artifact_path(state, "patch", edit_index, num_edits, "diff")
                        state.patch_path.write_text(patch, encoding="utf-8")
                        edit_applied = True
                    if not edit_applied:
                        raise ValueError(
                            f"Generator edit {edit_index}/{num_edits} did not return SEARCH/REPLACE blocks, "
                            "a unified diff, or a complete train.py."
                        )

                    if edit_record is None:
                        edit_record = {
                            "edit_index": edit_index,
                            "total_edits": num_edits,
                            "description": description,
                            "prompt_path": str(state.prompt_path),
                            "response_path": str(state.response_path),
                            "patch_path": str(state.patch_path),
                            "llm_response": state.llm_response,
                            "llm_response_truncated": state.llm_response_truncated,
                            "token_usage": completion_tokens,
                            "token_usage_detail": token_usage_detail,
                            "logprobs_path": None if logprobs_path is None else str(logprobs_path),
                            "logprob_summary": logprob_summary,
                            "edit_source": fallback_edit_source,
                            "detected_edit_count": 1,
                            "status": "applied",
                        }
                        state.edits.append(edit_record)
                    else:
                        edit_record["patch_path"] = str(state.patch_path)
                        edit_record["status"] = "applied"
                    self._progress_status(f"generated {state.state_id} {format_state_logprob_summary(state)}")
                    prior_edits.append(
                        {
                            "edit_index": edit_index,
                            "description": description,
                            "patch": truncate_text(patch, 4000),
                        }
                    )
                    current_text = new_text

                if num_edits > 1:
                    self._write_latest_artifact_aliases(state)
                state.train_path.write_text(current_text, encoding="utf-8")
                state.description = " | ".join(
                    edit["description"] for edit in state.edits if edit.get("description")
                )
                state.description = state.description[:300]
            state.status = "generated"
        except Exception as exc:
            state.status = "generation_error"
            state.score = self.config.failure_score
            state.error = repr(exc)
            for edit in reversed(state.edits):
                if edit.get("status") == "detected":
                    edit["status"] = "application_error"
                    edit["error"] = repr(exc)
                    break
            state.description = " | ".join(
                edit["description"] for edit in state.edits if edit.get("description")
            )
            state.description = state.description[:300]
            if not state.train_path.exists():
                state.train_path.write_text(current_text, encoding="utf-8")

        self._write_state_meta(state)
        self._record_manifest(state)
        self._progress_status(f"generated {state.state_id}")
        return state

    async def _call_generator(self, prompt: str, state: SearchState, current_train_text: str) -> tuple[str, Any]:
        if self.config.generator == "mock":
            return make_mock_patch(current_train_text, state.state_id), 0
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful code-research agent. Return only a unified diff "
                    "for train.py unless explicitly asked for a full file."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if self.config.generator == "closed_loop":
            from tasks.nanogpt.core.api_generate import openai_compatible_generate

            return await openai_compatible_generate(
                messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=self.config.stop,
                llm_url=self.config.llm_url,
                llm_model_name=self.config.llm_model_name,
                disable_thinking=self.config.disable_thinking,
                api_key=self.config.api_key,
                chat_template_extra=False,
                logprobs=self.config.request_logprobs,
                top_logprobs=self.config.top_logprobs,
            )
        if self.config.generator == "tool_call":
            from tasks.nanogpt.core.api_generate import tool_call_generate

            return await tool_call_generate(
                messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=self.config.stop,
                llm_url=self.config.llm_url,
                llm_model_name=self.config.llm_model_name,
                disable_thinking=self.config.disable_thinking,
                api_key=self.config.api_key,
                logprobs=self.config.request_logprobs,
                top_logprobs=self.config.top_logprobs,
            )
        if self.config.generator == "harness":
            from tasks.nanogpt.core.api_generate import harness_generate

            return await harness_generate(
                prompt,
                current_train_text,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=self.config.stop,
                llm_url=self.config.llm_url,
                llm_model_name=self.config.llm_model_name,
                disable_thinking=self.config.disable_thinking,
                api_key=self.config.api_key,
                logprobs=self.config.request_logprobs,
                top_logprobs=self.config.top_logprobs,
            )
        if self.config.generator != "api":
            raise ValueError(f"Unknown generator {self.config.generator!r}.")

        from tasks.nanogpt.core.api_generate import vllm_generate

        return await vllm_generate(
            messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=self.config.stop,
            llm_url=self.config.llm_url,
            llm_model_name=self.config.llm_model_name,
            disable_thinking=self.config.disable_thinking,
            api_key=self.config.api_key,
            logprobs=self.config.request_logprobs,
            top_logprobs=self.config.top_logprobs,
        )

    def _build_prompt(
        self,
        parent: SearchState,
        *,
        child_index: int,
        sibling_count: int,
        search_note: str,
        current_train_text: str | None = None,
        edit_index: int = 1,
        total_edits: int = 1,
        prior_edits: list[dict[str, Any]] | None = None,
    ) -> str:
        train_text = current_train_text if current_train_text is not None else parent.train_path.read_text(encoding="utf-8")
        if len(train_text) > self.config.prompt_max_chars:
            keep = self.config.prompt_max_chars // 2
            train_text = (
                train_text[:keep]
                + "\n\n# ... train.py truncated in the prompt middle ...\n\n"
                + train_text[-keep:]
            )

        history_lines = []
        for state in self.ranked_states()[:8]:
            direction = "lower is better" if self.config.minimize else "higher is better"
            history_lines.append(
                f"- {state.state_id}: {self.config.score_key}={state.score} ({direction}); "
                f"status={state.status}; note={state.description or 'n/a'}"
            )
        history = "\n".join(history_lines) if history_lines else "- No evaluated candidates yet."
        parent_score = "unknown" if parent.score is None else f"{parent.score:.8g}"
        task_context = self.config.task_context.strip()
        task_context_block = (
            "\nProject and benchmark context:\n" + task_context + "\n"
            if task_context
            else ""
        )
        extra_instruction = (
            "\nAdditional instruction from the caller:\n"
            + self.config.extra_instruction.strip()
            + "\n"
            if self.config.extra_instruction.strip()
            else ""
        )
        feedback_context = (
            "\nFeedback from previous iterations:\n"
            + self.config.feedback_context.strip()
            + "\n"
            if self.config.feedback_context.strip()
            else ""
        )
        prior_edits_text = format_prior_edits(prior_edits or [])
        edit_loop_text = (
            f"- Edit pass in this transition: {edit_index}/{total_edits}\n"
            f"- Previous edits already applied within this same child state:\n{prior_edits_text}\n"
            if total_edits > 1
            else ""
        )

        return f"""We are doing test-time inference scaling for autoresearch on `train.py`.

Search state:
- Parent state: {parent.state_id}
- Parent depth: {parent.depth}
- Parent {self.config.score_key}: {parent_score}
- Candidate among siblings: {child_index}/{sibling_count}
- Search note: {search_note or "propose one useful child edit"}
{edit_loop_text}
{task_context_block}

Objective:
- Improve `{self.config.score_key}` after executing the script.
- The default metric is validation BPB, where lower is better.
- Preserve the printed final summary and diagnostics JSON behavior so evaluation can parse results.
- Edit only `train.py`; do not require changes to `prepare.py`, data files, or dependencies.
- Prefer changes that are coherent and likely to run under the same fixed training-time budget.
- If this is one of multiple edit passes, make one coherent incremental edit that composes with prior edits.

Recent evaluated states:
{history}
{feedback_context}
{extra_instruction}
Return format:
- If tool calling is available, call the `edit_train_py` tool with a JSON object:
  {{"summary": "...", "edits": [{{"search": "exact code", "replace": "replacement code"}}]}}.
- Prefer SEARCH/REPLACE edit blocks. Return one or more blocks in this exact format:
  train.py
  <<<<<<< SEARCH
  exact code from the current train.py
  =======
  replacement code
  >>>>>>> REPLACE
- The SEARCH text must match the current train.py exactly, including indentation.
- Keep every edit neat and minimal:
  - Use the smallest unique SEARCH block that contains the changed lines plus only the context needed for uniqueness.
  - For a one-line hyperparameter change, SEARCH should usually be that one exact line, not a whole section.
  - Do not include large unchanged regions such as setup, model construction, optimizer creation, dataloader creation, or the training loop around a small change.
  - Do not include unrelated neighboring lines just because they are nearby.
  - Keep SEARCH and replacement blocks similar in scope; the replacement should only change what the summary says it changes.
- Do not return overlapping or redundant edits. If two edits touch the same lines, merge them into one larger SEARCH/REPLACE block.
- The edits are applied sequentially, so each later SEARCH must match the file after all earlier edits have been applied.
- If you cannot use SEARCH/REPLACE blocks, return a valid unified diff for train.py.
- Do not include prose outside a short Summary line and the edit blocks.

Current parent `train.py`:
```python
{train_text}
```
"""

    def _new_state(self, *, parent: SearchState | None, depth: int) -> SearchState:
        state_id = f"state_{self._counter:04d}"
        self._counter += 1
        workdir = self.config.out_dir / "states" / state_id
        workdir.mkdir(parents=True, exist_ok=True)
        state = SearchState(
            state_id=state_id,
            parent_id=None if parent is None else parent.state_id,
            depth=depth,
            workdir=workdir,
            train_path=workdir / "train.py",
        )
        self.states.append(state)
        return state

    def _format_eval_command(self, state: SearchState, diagnostics_path: Path) -> str | list[str]:
        mapping = {
            "train_path": shlex.quote(str(state.train_path)),
            "workdir": shlex.quote(str(state.workdir)),
            "diagnostics_path": shlex.quote(str(diagnostics_path)),
            "project_root": shlex.quote(str(self.config.project_root)),
        }
        command = self.config.eval_command.format(**mapping)
        return command if self.config.eval_shell else shlex.split(command)

    def _write_state_meta(self, state: SearchState) -> None:
        (state.workdir / "meta.json").write_text(
            json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _record_manifest(self, state: SearchState) -> None:
        self._manifest_recorder.append_round(state.to_json())

    def _edit_artifact_path(self, state: SearchState, stem: str, edit_index: int, total_edits: int, suffix: str) -> Path:
        if total_edits <= 1:
            return state.workdir / f"{stem}.{suffix}"
        return state.workdir / f"{stem}_edit_{edit_index:02d}.{suffix}"

    def _write_latest_artifact_aliases(self, state: SearchState) -> None:
        aliases = [
            (state.prompt_path, state.workdir / "prompt.md"),
            (state.response_path, state.workdir / "response.md"),
            (state.patch_path, state.workdir / "patch.diff"),
        ]
        for source, alias in aliases:
            if source is not None and source.exists() and source != alias:
                shutil.copy2(source, alias)

    def _inline_llm_response(self, response: str) -> tuple[str, bool]:
        limit = self.config.response_log_chars
        if limit < 0 or len(response) <= limit:
            return response, False
        if limit == 0:
            return "", True
        return response[:limit], True

    def _advance_progress(self, state: SearchState) -> None:
        if self._progress is None:
            return
        self._progress_count += 1
        best = self.best_state()
        status = state.state_id
        if state.score is not None and _is_finite_number(state.score):
            status += f" {self.config.score_key}={float(state.score):.6g}"
        else:
            status += f" {state.status}"
        logprob_text = format_state_logprob_summary(state)
        if logprob_text:
            status += f" {logprob_text}"
        self._progress.update(
            self._progress_count,
            best_score=None if best is None else best.score,
            status=status,
        )

    def _progress_status(self, status: str) -> None:
        self.debug_event("progress_status", status=status)
        if self._progress is not None:
            best = self.best_state()
            self._progress.update(
                self._progress_count,
                best_score=None if best is None else best.score,
                status=status,
            )

    def debug_event(self, event: str, **payload: Any) -> None:
        if not self.config.debug:
            return
        path = self.config.debug_log_path or (self.config.out_dir / "debug.jsonl")
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": round(time.time() - self._debug_started_at, 6),
            "event": event,
            **{key: jsonable(value) for key, value in payload.items()},
        }
        line = json.dumps(record, sort_keys=True) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(line)
        except OSError:
            pass
        print(f"[nanogpt-debug] {line.rstrip()}", file=sys.stderr, flush=True)


class ProgressBar:
    def __init__(self, *, total: int, label: str, width: int):
        self.total = max(1, int(total))
        self.label = label
        self.width = max(10, int(width))
        self.started_at = time.time()
        self.last_len = 0

    def update(self, current: int, *, best_score: float | None = None, status: str = "") -> None:
        current = max(0, int(current))
        display_total = max(self.total, current, 1)
        frac = min(current / display_total, 1.0)
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.time() - self.started_at, 1e-9)
        rate = current / elapsed if current else 0.0
        eta = (display_total - current) / rate if rate > 0 and current < display_total else 0.0
        best_text = "" if best_score is None else f" best={float(best_score):.6g}"
        status_text = "" if not status else f" | {status}"
        line = (
            f"\r{self.label} [{bar}] {current}/{display_total} "
            f"{100 * frac:5.1f}%{best_text} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}{status_text}"
        )
        padding = " " * max(0, self.last_len - len(line))
        sys.stderr.write(line + padding)
        sys.stderr.flush()
        self.last_len = len(line)

    def finish(self, current: int, *, best_score: float | None = None, status: str = "done") -> None:
        self.update(max(current, self.total), best_score=best_score, status=status)
        sys.stderr.write("\n")
        sys.stderr.flush()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def parse_metrics(output_text: str, diagnostics_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if diagnostics_path.exists():
        try:
            loaded = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metrics.update(loaded)
        except json.JSONDecodeError:
            pass

    inline_prefix = "diagnostics_json_inline:"
    for line in output_text.splitlines():
        if line.startswith(inline_prefix):
            try:
                loaded = json.loads(line[len(inline_prefix) :].strip())
                if isinstance(loaded, dict):
                    metrics.update(loaded)
            except json.JSONDecodeError:
                pass

    for match in re.finditer(rf"^([A-Za-z_][\w_]*):\s*({FLOAT_RE})\s*$", output_text, re.MULTILINE):
        key, value = match.groups()
        metrics.setdefault(key, float(value))
    return metrics


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def normalize_token_usage(usage: Any) -> tuple[int, dict[str, int]]:
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    else:
        prompt_tokens = 0
        completion_tokens = int(usage or 0)
        total_tokens = prompt_tokens + completion_tokens
    return completion_tokens, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def extract_generation_logprobs(usage: Any) -> Any:
    if isinstance(usage, dict):
        return usage.get("logprobs")
    return None


def summarize_generation_logprobs(logprobs_payload: Any) -> dict[str, Any]:
    token_logprobs = collect_token_logprobs(logprobs_payload)
    if not token_logprobs:
        return {}
    total = float(sum(token_logprobs))
    count = len(token_logprobs)
    mean = total / count
    return {
        "token_count": count,
        "sum_logprob": total,
        "mean_logprob": mean,
        "min_logprob": float(min(token_logprobs)),
        "max_logprob": float(max(token_logprobs)),
        "mean_probability": float(math.exp(mean)),
        "perplexity": float(math.exp(-mean)),
    }


def collect_token_logprobs(payload: Any) -> list[float]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        if isinstance(payload.get("content"), list):
            values = []
            for item in payload["content"]:
                if isinstance(item, dict) and _is_finite_number(item.get("logprob")):
                    values.append(float(item["logprob"]))
            return values
        if isinstance(payload.get("token_logprobs"), list):
            return [float(value) for value in payload["token_logprobs"] if _is_finite_number(value)]
    if isinstance(payload, list):
        values = []
        for item in payload:
            if isinstance(item, dict) and _is_finite_number(item.get("logprob")):
                values.append(float(item["logprob"]))
            elif _is_finite_number(item):
                values.append(float(item))
        return values
    return []


def format_state_logprob_summary(state: SearchState) -> str:
    if not state.edits:
        return ""
    summary = state.edits[-1].get("logprob_summary")
    if not isinstance(summary, dict) or not summary:
        return ""
    parts = []
    mean_logprob = summary.get("mean_logprob")
    perplexity = summary.get("perplexity")
    token_count = summary.get("token_count")
    if _is_finite_number(mean_logprob):
        parts.append(f"lp={float(mean_logprob):.4g}")
    if _is_finite_number(perplexity):
        parts.append(f"ppl={float(perplexity):.4g}")
    if _is_finite_number(token_count):
        parts.append(f"lptok={int(token_count)}")
    return " ".join(parts)


def add_token_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": int(left.get("prompt_tokens", 0)) + int(right.get("prompt_tokens", 0)),
        "completion_tokens": int(left.get("completion_tokens", 0)) + int(right.get("completion_tokens", 0)),
        "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
    }


def format_prior_edits(prior_edits: list[dict[str, Any]]) -> str:
    if not prior_edits:
        return "- None yet."
    lines = []
    for edit in prior_edits:
        lines.append(
            f"- Edit {edit.get('edit_index')}: {edit.get('description') or 'no description'}\n"
            f"  Patch excerpt:\n{indent_text(str(edit.get('patch') or ''), '  ')}"
        )
    return "\n".join(lines)


def indent_text(text: str, prefix: str) -> str:
    if not text:
        return prefix + "(empty)"
    return "\n".join(prefix + line for line in text.splitlines())


def truncate_text(text: str, limit: int) -> str:
    if limit < 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def extract_search_replace_blocks(response: str) -> list[dict[str, str]]:
    blocks = []
    pattern = re.compile(
        r"(?:^|\n)(?:(?P<filename>[^\n`<>]+?)\n)?"
        r"<<<<<<< SEARCH\n"
        r"(?P<search>.*?)\n"
        r"=======\n"
        r"(?P<replace>.*?)\n"
        r">>>>>>> REPLACE",
        re.DOTALL,
    )
    for match in pattern.finditer(response):
        filename = (match.group("filename") or "train.py").strip()
        if filename and filename != "train.py" and not filename.endswith("/train.py"):
            continue
        blocks.append(
            {
                "search": strip_code_fence_edges(match.group("search")),
                "replace": strip_code_fence_edges(match.group("replace")),
            }
        )
    return blocks


def extract_tool_call_edit_blocks(response: str) -> list[dict[str, str]]:
    payload_match = re.search(r"<tool_calls>\s*(.*?)\s*</tool_calls>", response, re.DOTALL)
    if not payload_match:
        return []
    try:
        tool_calls = json.loads(payload_match.group(1))
    except json.JSONDecodeError:
        return []
    blocks = []
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("name") != "edit_train_py":
            continue
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        for edit in arguments.get("edits", []):
            if isinstance(edit, dict) and "search" in edit and "replace" in edit:
                blocks.append({"search": str(edit["search"]), "replace": str(edit["replace"])})
    return blocks


def extract_tool_call_summary(response: str) -> str:
    payload_match = re.search(r"<tool_calls>\s*(.*?)\s*</tool_calls>", response, re.DOTALL)
    if not payload_match:
        return ""
    try:
        tool_calls = json.loads(payload_match.group(1))
    except json.JSONDecodeError:
        return ""
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("name") != "edit_train_py":
            continue
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        summary = arguments.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()[:300]
    return ""


def summarize_edit_blocks(blocks: list[dict[str, str]], *, limit: int = 600) -> list[dict[str, str]]:
    summary = []
    for block in blocks:
        summary.append(
            {
                "search": truncate_text(str(block.get("search") or ""), limit),
                "replace": truncate_text(str(block.get("replace") or ""), limit),
            }
        )
    return summary


def apply_search_replace_blocks(text: str, blocks: list[dict[str, str]]) -> str:
    current = text
    for index, block in enumerate(blocks, start=1):
        search = block["search"]
        replace = block["replace"]
        if not search:
            raise ValueError(f"SEARCH/REPLACE block {index} has empty SEARCH text.")
        count = current.count(search)
        if count == 0:
            replace_count = current.count(replace) if replace else 0
            if replace_count == 1:
                continue
            blank_line_tolerant = apply_blank_line_tolerant_replace(current, search, replace)
            if blank_line_tolerant is not None:
                current = blank_line_tolerant
                continue
            raise ValueError(
                f"SEARCH/REPLACE block {index} SEARCH text was not found exactly. "
                "The response was parsed, but the SEARCH text did not match the current train.py."
            )
        if count > 1:
            raise ValueError(f"SEARCH/REPLACE block {index} SEARCH text matched {count} times; make it unique.")
        current = current.replace(search, replace, 1)
    return current


def apply_blank_line_tolerant_replace(text: str, search: str, replace: str) -> str | None:
    spans = find_blank_line_tolerant_spans(text, search)
    if not spans:
        return None
    if len(spans) > 1:
        raise ValueError(
            f"SEARCH text was not found exactly, and blank-line-tolerant matching found {len(spans)} matches."
        )
    start, end = spans[0]
    return text[:start] + replace + text[end:]


def find_blank_line_tolerant_spans(text: str, search: str) -> list[tuple[int, int]]:
    current_lines = text.splitlines(keepends=True)
    pattern_lines = search.splitlines(keepends=True)
    if not current_lines or not pattern_lines:
        return []

    offsets: list[int] = []
    offset = 0
    for line in current_lines:
        offsets.append(offset)
        offset += len(line)

    spans: list[tuple[int, int]] = []
    for start_line in range(len(current_lines)):
        match = match_blank_line_tolerant_at(current_lines, pattern_lines, start_line)
        if match is None:
            continue
        last_line_index, last_pattern_line = match
        start = offsets[start_line]
        end = offsets[last_line_index] + matched_line_end_len(current_lines[last_line_index], last_pattern_line)
        spans.append((start, end))
    return spans


def match_blank_line_tolerant_at(
    current_lines: list[str],
    pattern_lines: list[str],
    start_line: int,
) -> tuple[int, str] | None:
    current_index = start_line
    pattern_index = 0
    last_line_index: int | None = None
    last_pattern_line = ""

    while pattern_index < len(pattern_lines):
        pattern_line = pattern_lines[pattern_index]
        if is_blank_line(pattern_line):
            while pattern_index < len(pattern_lines) and is_blank_line(pattern_lines[pattern_index]):
                last_pattern_line = pattern_lines[pattern_index]
                pattern_index += 1
            if current_index >= len(current_lines) or not is_blank_line(current_lines[current_index]):
                return None
            while current_index < len(current_lines) and is_blank_line(current_lines[current_index]):
                last_line_index = current_index
                current_index += 1
            continue

        if current_index >= len(current_lines) or is_blank_line(current_lines[current_index]):
            return None
        if not lines_equal_ignoring_trailing_space(current_lines[current_index], pattern_line):
            return None
        last_line_index = current_index
        last_pattern_line = pattern_line
        current_index += 1
        pattern_index += 1

    if last_line_index is None:
        return None
    return last_line_index, last_pattern_line


def is_blank_line(line: str) -> bool:
    return strip_line_ending(line).strip(" \t") == ""


def lines_equal_ignoring_trailing_space(left: str, right: str) -> bool:
    return strip_line_ending(left).rstrip(" \t") == strip_line_ending(right).rstrip(" \t")


def matched_line_end_len(current_line: str, pattern_line: str) -> int:
    if pattern_line.endswith(("\n", "\r")):
        return len(current_line)
    return len(strip_line_ending(current_line))


def strip_line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith(("\n", "\r")):
        return line[:-1]
    return line


def strip_code_fence_edges(text: str) -> str:
    text = text.strip("\n")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def extract_unified_diff(response: str) -> str | None:
    fence_re = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    for block in fence_re.findall(response):
        block = block.strip()
        if looks_like_unified_diff(block):
            return _trim_diff(block)

    raw = response.strip()
    if looks_like_unified_diff(raw):
        return _trim_diff(raw)

    for marker in ("diff --git", "--- a/train.py", "--- train.py"):
        idx = response.find(marker)
        if idx >= 0:
            candidate = response[idx:].strip()
            if looks_like_unified_diff(candidate):
                return _trim_diff(candidate)
    return None


def extract_replacement_train_file(response: str) -> str | None:
    tagged = re.search(r"<train\.py>\s*(.*?)\s*</train\.py>", response, re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip() + "\n"

    fence_re = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    for block in fence_re.findall(response):
        if _looks_like_train_file(block):
            return block.strip() + "\n"

    if _looks_like_train_file(response):
        return response.strip() + "\n"
    return None


def looks_like_unified_diff(text: str) -> bool:
    return ("@@ " in text or "\n@@" in text) and (
        "diff --git" in text or "--- a/train.py" in text or "--- train.py" in text
    )


def apply_unified_diff(parent_train_path: Path, patch: str) -> str:
    return apply_unified_diff_to_text(parent_train_path.read_text(encoding="utf-8"), patch)


def apply_unified_diff_to_text(parent_text: str, patch: str) -> str:
    with tempfile.TemporaryDirectory(prefix="tts_patch_") as tmp:
        tmpdir = Path(tmp)
        train_path = tmpdir / "train.py"
        train_path.write_text(parent_text, encoding="utf-8")
        errors = []
        for args in (["git", "apply", "--whitespace=nowarn", "-"], ["git", "apply", "-p0", "--whitespace=nowarn", "-"]):
            proc = subprocess.run(
                args,
                cwd=tmpdir,
                input=patch,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0:
                return train_path.read_text(encoding="utf-8")
            errors.append(proc.stderr.strip())
            train_path.write_text(parent_text, encoding="utf-8")
    raise ValueError("Could not apply generated patch: " + " | ".join(error for error in errors if error))


def make_unified_diff(old: str, new: str, *, fromfile: str, tofile: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def extract_description(response: str) -> str:
    tool_summary = extract_tool_call_summary(response)
    if tool_summary:
        return tool_summary
    for pattern in (r"(?im)^description:\s*(.+)$", r"(?im)^summary:\s*(.+)$"):
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()[:300]
    blocks = extract_search_replace_blocks(response)
    if blocks:
        return f"{len(blocks)} search/replace block(s)"
    patch = extract_unified_diff(response)
    if patch:
        added = [line[1:].strip() for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")]
        return " / ".join(added[:2])[:300]
    return ""


def make_mock_patch(parent_text: str, state_id: str) -> str:
    new_text, description = make_mock_candidate_text(parent_text, state_id)
    response = make_search_replace_response(parent_text, new_text, description)
    if not response:
        new_text = parent_text.rstrip() + f"\n# TTS mock candidate: {state_id}\n"
        response = make_search_replace_response(parent_text, new_text, f"tie-breaker mock edit for {state_id}")
    return response


def make_mock_candidate_text(text: str, state_id: str) -> tuple[str, str]:
    if "Mock autoresearch training script" in text and "compute_mock_metrics" in text:
        return tune_mock_train_text(text, state_id)

    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    candidate = "".join(
        lines[:insert_at] + [f"# TTS mock candidate: {state_id}\n"] + lines[insert_at:]
    )
    return candidate, f"comment-only mock edit for {state_id}"


def tune_mock_train_text(text: str, state_id: str) -> tuple[str, str]:
    targets = [
        ("DEPTH", "10"),
        ("WIDTH", "704"),
        ("MATRIX_LR", "0.022"),
        ("EMBEDDING_LR", "0.55"),
        ("WEIGHT_DECAY", "0.16"),
        ("VALUE_GATE_CHANNELS", "64"),
        ("WARMDOWN_RATIO", "0.45"),
        ("NGRAM_SCALE", "0.18"),
    ]
    match = re.search(r"(\d+)$", state_id)
    start = int(match.group(1)) if match else 0
    for offset in range(len(targets)):
        name, target = targets[(start + offset) % len(targets)]
        pattern = rf"^{name}\s*=\s*[^\n#]+"
        line_match = re.search(pattern, text, re.MULTILINE)
        if line_match is None:
            continue
        replacement = f"{name} = {target}"
        if line_match.group(0).strip() == replacement:
            continue
        return (
            text[: line_match.start()] + replacement + text[line_match.end() :],
            f"move {name} toward the mock optimum",
        )

    return text.rstrip() + f"\n# TTS mock candidate: {state_id}\n", f"tie-breaker mock edit for {state_id}"


def make_search_replace_response(old_text: str, new_text: str, description: str) -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    blocks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        context_before = max(0, i1 - 2)
        context_after = min(len(old_lines), i2 + 2)
        search = "".join(old_lines[context_before:context_after]).rstrip("\n")
        replace = (
            "".join(old_lines[context_before:i1])
            + "".join(new_lines[j1:j2])
            + "".join(old_lines[i2:context_after])
        ).rstrip("\n")
        blocks.append(
            "train.py\n"
            "<<<<<<< SEARCH\n"
            f"{search}\n"
            "=======\n"
            f"{replace}\n"
            ">>>>>>> REPLACE"
        )
    if not blocks:
        return ""
    return f"Summary: {description}\n\n" + "\n\n".join(blocks) + "\n"


def _trim_diff(text: str) -> str:
    text = text.strip()
    if not text.endswith("\n"):
        text += "\n"
    return text


def _looks_like_train_file(text: str) -> bool:
    looks_like_real_train = (
        "class GPT" in text
        and "def forward" in text
        and "evaluate_bpb" in text
        and "import torch" in text
    )
    looks_like_mock_train = (
        "val_bpb" in text
        and "diagnostics_json_inline" in text
        and "AUTORESEARCH_DIAGNOSTICS_JSON" in text
    )
    return looks_like_real_train or looks_like_mock_train


def _is_finite_number(value: Any) -> bool:
    return is_finite_number(value)
