"""Trajectory recording for the LLM advisor.

The :class:`TrajectoryRecorder` writes a single final JSON at the end
of a run. Intermediate state lives in memory in :class:`RoundRecord`
and is committed to disk only on completion (or on fatal error via
:meth:`dump_emergency_json`).

Layout of the final JSON:

.. code-block:: jsonc

    {
      "status": "completed" | "fatal_error",
      "run_metadata": {
        "method": "...",
        "seed": 0,
        "llm_model": "...",
        "started_at": "ISO-8601",
        "finished_at": "ISO-8601",
        "duration_seconds": 0.0
      },
      "config": {...},
      "history": [
        {"index": 0, "smiles": "CCO", "score": -7.2}, ...
      ],
      "rounds": [
        {
          "round_idx": 0,
          "phase": "bo",
          "timestamp": "ISO-8601",
          "pre_state_snapshot": {...},
          "llm_interactions": {
            "phase_a": {
              "executed": true,
              "skipped_reason": null,
              "attempts": [LLMAttemptRecord, ...],
              "fallback_used": false,
              "final_blocks": [...],
              "applied_actions": {...}
            },
            "phase_b": {
              "executed": true,
              "attempts": [...],
              "fallback_used": false,
              "final_blocks": [...],
              "review_bo_block": {...},
              "final_candidates": [...],
              "overrides": {...}
            }
          },
          "pool_after_phase_a": [...],
          "bo_suggestions": [...],
          "scores": {...},
          "pool_after": [...],
          "pending_analogs_after": [...],
          "warnings": [...],
          "errors": []
        }
      ],
      "fatal_error": null | {...}
    }

Output path conventions
----------------------

The :func:`resolve_trajectory_path` helper applies the same dir-or-
file disambiguation as :func:`run_search.resolve_output_path`:

* ``--trajectory-path output/bo_llm`` (directory) →
  ``output/bo_llm/{method}_seed={seed}_trajectory.json``
* ``--trajectory-path output/bo_llm/foo.json`` (file) → verbatim.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from strbo_v1.llm_advisor.advisor import LLMAttemptRecord
from strbo_v1.llm_advisor.blocks import LLMBlock
from strbo_v1.llm_advisor.state import AnalogueRecord

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_trajectory_path(
    raw_path: str | Path,
    *,
    method: str,
    seed: int,
) -> Path:
    """Apply the run_search-style dir-or-file disambiguation.

    If ``raw_path`` ends in ``.json`` use it verbatim; otherwise treat
    it as a directory and append
    ``{method}_seed={seed}_trajectory.json``.
    """
    raw = Path(raw_path)
    if raw.suffix.lower() == ".json":
        return raw
    return raw / f"{method}_seed={seed}_trajectory.json"


# ---------------------------------------------------------------------------
# Per-round record (in-memory; serialized on commit)
# ---------------------------------------------------------------------------


@dataclass
class RoundRecord:
    """All events from one BO round, in-memory.

    The orchestrator constructs one of these per round via
    :meth:`TrajectoryRecorder.round_context`, fills its fields, and
    lets the context manager commit it to the recorder's internal
    list. No disk I/O happens until :meth:`write_final` (or
    :meth:`dump_emergency_json`).
    """

    round_idx: int = 0
    phase: str = "bo"
    timestamp: str = ""

    pre_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    pool_after_phase_a: List[str] = field(default_factory=list)
    bo_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    llm_interactions: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, Any] = field(default_factory=dict)
    pool_after: List[str] = field(default_factory=list)
    pending_analogs_after: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "pre_state_snapshot": self.pre_state_snapshot,
            "pool_after_phase_a": list(self.pool_after_phase_a),
            "bo_suggestions": list(self.bo_suggestions),
            "llm_interactions": dict(self.llm_interactions),
            "scores": dict(self.scores),
            "pool_after": list(self.pool_after),
            "pending_analogs_after": list(self.pending_analogs_after),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryRecorder:
    """In-memory collector that writes one final JSON on completion.

    Construction with ``path`` is lazy — the file is only created on
    :meth:`write_final` (success path) or :meth:`dump_emergency_json`
    (fatal path).
    """

    path: Path
    method: str = "llm-bo"
    seed: int = 0

    _rounds: List[RoundRecord] = field(default_factory=list, init=False, repr=False)
    _current: Optional[RoundRecord] = field(default=None, init=False, repr=False)
    _started_at: float = field(default=0.0, init=False, repr=False)
    _status: str = field(default="completed", init=False, repr=False)
    _fatal_error: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _config_echo: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _llm_model: str = field(default="", init=False, repr=False)
    _final_history: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _written: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_run(
        self,
        config: Dict[str, Any],
        llm_model: str,
    ) -> None:
        """Initialize per-run metadata. Call once at process start."""
        self._started_at = time.monotonic()
        self._config_echo = dict(config)
        self._llm_model = llm_model
        self._rounds = []
        self._current = None
        self._status = "completed"
        self._fatal_error = None
        self._written = False

    def set_status(self, status: str) -> None:
        if status not in ("completed", "fatal_error"):
            raise ValueError(f"unknown status: {status!r}")
        self._status = status

    def set_final_history(
        self, history: List[tuple[str, Any]],
    ) -> None:
        """Store the final history (post-run) for inclusion in the JSON.

        For n_obj==1 each ``sc`` is a bare float; for n_obj>=2 each
        ``sc`` is a ``list[float]`` of length n_obj. The on-disk
        representation preserves that distinction (the LDM does not
        collapse multi-obj scores to a single float).
        """
        out: List[Dict[str, Any]] = []
        for i, (smi, sc) in enumerate(history):
            if isinstance(sc, (list, tuple)):
                cleaned = [_scrub_score(x) for x in sc]
                out.append({"index": i, "smiles": smi, "scores": cleaned})
            else:
                out.append({"index": i, "smiles": smi, "score": _scrub_score(sc)})
        self._final_history = out

    # ------------------------------------------------------------------
    # Round context
    # ------------------------------------------------------------------

    @contextmanager
    def round_context(self, round_idx: int) -> Iterator[RoundRecord]:
        """Context manager: yields a fresh :class:`RoundRecord`.

        The orchestrator fills the record's fields inside the
        ``with`` block. The recorder auto-populates ``round_idx``,
        ``phase``, and ``timestamp`` on entry.
        """
        rec = RoundRecord(
            round_idx=round_idx,
            phase="bo",
            timestamp=_now_iso(),
        )
        self._current = rec
        try:
            yield rec
        finally:
            self._rounds.append(rec)
            self._current = None

    @property
    def current_round(self) -> int:
        return self._current.round_idx if self._current is not None else -1

    # ------------------------------------------------------------------
    # Commit / write
    # ------------------------------------------------------------------

    def commit_in_flight(self) -> None:
        """No-op stub: in-memory only. Reserved for future streaming."""
        return

    def record_fatal_error(
        self,
        *,
        round_idx: int,
        exc: BaseException,
    ) -> None:
        """Stamp the current (or specified) round as the failure point.

        Stores a structured ``fatal_error`` blob for the final JSON
        and sets ``status = "fatal_error"``.
        """
        self._status = "fatal_error"
        self._fatal_error = {
            "round_idx": round_idx,
            "exc_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }

    def write_final(self) -> None:
        """Write the final trajectory JSON. Idempotent."""
        if self._written:
            return
        self._written = True
        payload = self._build_payload()
        # Coerce any non-JSON-serializable values (e.g. dataclass instances
        # like LLMClientConfig leaked via stored snapshots) into a safe
        # repr-like string. Trajectory is for audit, not pickling.
        _coerce_to_jsonable_inplace(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        LOGGER.info("trajectory written: %s (%d rounds)", self.path, len(self._rounds))

    def dump_emergency_json(self) -> Optional[Path]:
        """Write a sidecar ``*.error.json`` containing the same content
        as ``write_final`` would, plus the structured ``fatal_error``
        field. Returns the sidecar path, or ``None`` if no error was
        recorded.
        """
        if self._fatal_error is None and self._status != "fatal_error":
            return None
        sidecar = self.path.with_suffix(self.path.suffix + ".error.json")
        payload = self._build_payload()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        LOGGER.error("emergency trajectory written: %s", sidecar)
        return sidecar

    # ------------------------------------------------------------------
    # Payload assembly
    # ------------------------------------------------------------------

    def _build_payload(self) -> Dict[str, Any]:
        finished = time.monotonic()
        return {
            "status": self._status,
            "run_metadata": {
                "method": self.method,
                "seed": self.seed,
                "llm_model": self._llm_model,
                "started_at": _iso_from_ts(self._started_at),
                "finished_at": _iso_from_ts(finished),
                "duration_seconds": round(finished - self._started_at, 3),
            },
            "config": self._config_echo,
            "history": self._final_history,
            "rounds": [r.to_dict() for r in self._rounds],
            "fatal_error": self._fatal_error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _scrub_score(sc: Any) -> Any:
    """Make scores JSON-serializable (lists stay as lists; pass through None / floats).

    The LDM orchestrator's history uses the ``float | list[float]``
    convention (no tuple-wrapping): a single-obj score is a bare
    float, a multi-obj score is a list of length ``n_obj``. This
    function preserves that distinction in the trajectory JSON.
    """
    if sc is None:
        return None
    if isinstance(sc, (int, float)):
        return float(sc)
    if isinstance(sc, (tuple, list)):
        return [_scrub_score(x) for x in sc]
    return sc


def _scrub_value(v: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable form.

    Used as a last-resort scrubber for the trajectory's payload. The
    orchestrator's per-round ``pre_state_snapshot`` may contain values
    that were not originally intended for JSON (e.g. when an upstream
    caller stored a dataclass instance). We recurse through dicts /
    lists and stringify any non-trivial value.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _scrub_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scrub_value(x) for x in v]
    if dataclasses.is_dataclass(v):
        try:
            return _scrub_value(dataclasses.asdict(v))
        except Exception:
            return f"<{type(v).__name__}>"
    return f"<{type(v).__name__}>"


def _coerce_to_jsonable_inplace(obj: Any) -> None:
    """Walk an arbitrary nested structure in place, replacing any
    non-JSON-serializable leaf with its ``__class__.__name__`` repr.
    """
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            try:
                import json as _json
                _json.dumps(v)
                continue
            except TypeError:
                obj[k] = _scrub_value(v)
    elif isinstance(obj, (list, tuple)):
        for i in range(len(obj)):
            try:
                import json as _json
                _json.dumps(obj[i])
            except TypeError:
                obj[i] = _scrub_value(obj[i])


def serialize_blocks(blocks: List[LLMBlock]) -> List[Dict[str, Any]]:
    return [b.to_dict() for b in blocks]


def serialize_analogues(records: List[AnalogueRecord]) -> List[Dict[str, Any]]:
    return [a.to_dict() for a in records]


def serialize_attempts(attempts: List[LLMAttemptRecord]) -> List[Dict[str, Any]]:
    return [a.to_dict() for a in attempts]


__all__ = [
    "TrajectoryRecorder",
    "RoundRecord",
    "resolve_trajectory_path",
    "serialize_blocks",
    "serialize_analogues",
    "serialize_attempts",
]
