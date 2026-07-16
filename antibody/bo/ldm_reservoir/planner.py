"""LLM planner for the standalone reservoir LDM prototype.

Three planner modes are supported:
- ``propose``: one LLM output contains a list of K strategies.
- ``propose_n_choices``: one LLM request asks the backend for K independent
  choices, each returning one strategy. This is the efficient independent
  sampling mode when the backend supports an ``n``/``num_generations`` style
  parameter.
- ``propose_independent``: K independent LLM calls are issued concurrently,
  each returning one strategy. This remains as a fallback/comparison mode.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from typing import Any, Optional

from bo.ldm.dsl.bias import BiasAtom
from bo.ldm.dsl.sandbox import safe_exec_dsl
from bo.ldm.dsl.search_space import Or, SearchSpaceAtom
from bo.ldm.dsl.validator import validate_bias_atom, validate_search_atom
from bo.ldm.llm.client import LLMClient
from .config import ReservoirLDMConfig


@dataclass
class ReservoirPlan:
    strategies: list[SearchSpaceAtom]
    bias_dsl: Optional[BiasAtom]
    rationale: Optional[str]
    raw: str


class ReservoirPlanner:
    """Ask an LLM for strategy atoms z conditioned on BO history C_t."""

    def __init__(self, llm_client: LLMClient, config: Optional[ReservoirLDMConfig] = None) -> None:
        self.llm_client = llm_client
        self.config = config or ReservoirLDMConfig()
        self.last_error: Optional[str] = None

    def propose(self, context: dict[str, Any]) -> ReservoirPlan:
        """Single-output mode: one LLM response contains K strategies."""
        last_error = None
        for _ in range(max(1, self.config.max_retries)):
            prompt = self._build_prompt(context, last_error)
            raw = self.llm_client.call(
                prompt,
                temperature=self.config.llm_temperature,
                timeout_s=self.config.llm_call_timeout_s,
            )
            try:
                plan = self.parse(raw)
                self.last_error = None
                return plan
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.last_error = last_error
        raise RuntimeError(f"LLM failed to produce a valid reservoir plan: {last_error}")

    def propose_independent(self, context: dict[str, Any], n: int | None = None) -> ReservoirPlan:
        """Independent-parallel mode: launch n concurrent LLM calls.

        Each call samples one strategy z_i from p_theta(z | C_t). The calls do
        not condition on each other's outputs; they only share the same BO
        context. Returned strategies are concatenated into one ReservoirPlan.
        """
        n = int(n or self.config.n_strategies)
        if n <= 0:
            raise ValueError("n must be positive")

        def call_one(sample_idx: int) -> tuple[int, SearchSpaceAtom, Optional[BiasAtom], str, str]:
            last_error = None
            for _ in range(max(1, self.config.max_retries)):
                prompt = self._build_single_strategy_prompt(context, sample_idx, n, last_error)
                raw = self.llm_client.call(
                    prompt,
                    temperature=self.config.llm_temperature,
                    timeout_s=self.config.llm_call_timeout_s,
                )
                try:
                    atom, bias, rationale = self.parse_one(raw)
                    self.last_error = None
                    return sample_idx, atom, bias, rationale or "", raw
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"sample {sample_idx} failed: {last_error}")

        results: list[tuple[int, SearchSpaceAtom, Optional[BiasAtom], str, str]] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(call_one, i): i for i in range(n)}
            for future in as_completed(futures):
                sample_idx = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(f"sample {sample_idx}: {type(exc).__name__}: {exc}")

        if not results:
            self.last_error = "; ".join(errors)
            raise RuntimeError(f"all independent strategy samples failed: {self.last_error}")

        results.sort(key=lambda item: item[0])
        strategies = [item[1] for item in results][:n]
        bias = next((item[2] for item in results if item[2] is not None), None)
        rationales = [f"z{item[0]}: {item[3]}" for item in results if item[3]]
        raw_payload = {
            "mode": "independent_parallel",
            "n_requested": n,
            "n_success": len(results),
            "errors": errors,
            "raw_responses": [{"sample_idx": i, "raw": raw} for i, _, _, _, raw in results],
        }
        if errors:
            self.last_error = "; ".join(errors)
        return ReservoirPlan(
            strategies=strategies,
            bias_dsl=bias,
            rationale="\n".join(rationales) if rationales else None,
            raw=json.dumps(raw_payload, ensure_ascii=False),
        )

    def propose_n_choices(self, context: dict[str, Any], n: int | None = None) -> ReservoirPlan:
        """Independent-choice mode: one transport call returns n completions.

        This is the closest implementation of sampling z_1..z_n independently
        from p_theta(z | C_t) without issuing n separate network requests. It
        relies on an LLM client exposing ``call_many(..., n=n)``. If the client
        lacks that API, we fall back to the older independent call loop so the
        planner remains usable in tests and with simple clients.
        """
        n = int(n or self.config.n_strategies)
        if n <= 0:
            raise ValueError("n must be positive")

        last_error = None
        for _ in range(max(1, self.config.max_retries)):
            prompt = self._build_n_choices_prompt(context, n, last_error)
            used_call_many = hasattr(self.llm_client, "call_many")
            if used_call_many:
                raw_outputs = self.llm_client.call_many(
                    prompt,
                    temperature=self.config.llm_temperature,
                    timeout_s=self.config.llm_call_timeout_s,
                    n=n,
                )
            else:
                raw_outputs = [
                    self.llm_client.call(
                        prompt,
                        temperature=self.config.llm_temperature,
                        timeout_s=self.config.llm_call_timeout_s,
                    )
                    for _ in range(n)
                ]

            if not isinstance(raw_outputs, list) or not raw_outputs:
                last_error = "call_many returned no choices"
                self.last_error = last_error
                continue

            results: list[tuple[int, SearchSpaceAtom, Optional[BiasAtom], str, str]] = []
            errors: list[str] = []
            for sample_idx, raw in enumerate(raw_outputs[:n]):
                try:
                    atom, bias, rationale = self.parse_one(raw)
                    results.append((sample_idx, atom, bias, rationale or "", raw))
                except Exception as exc:
                    errors.append(f"choice {sample_idx}: {type(exc).__name__}: {exc}")

            if results:
                self.last_error = "; ".join(errors) if errors else None
                results.sort(key=lambda item: item[0])
                strategies = [item[1] for item in results][:n]
                bias = next((item[2] for item in results if item[2] is not None), None)
                rationales = [f"z{item[0]}: {item[3]}" for item in results if item[3]]
                raw_payload = {
                    "mode": "n_choices",
                    "transport": "call_many" if used_call_many else "call_loop_fallback",
                    "n_requested": n,
                    "n_returned": len(raw_outputs),
                    "n_success": len(results),
                    "errors": errors,
                    "raw_responses": [{"choice_idx": i, "raw": raw} for i, _, _, _, raw in results],
                }
                return ReservoirPlan(
                    strategies=strategies,
                    bias_dsl=bias,
                    rationale="\n".join(rationales) if rationales else None,
                    raw=json.dumps(raw_payload, ensure_ascii=False),
                )

            last_error = "; ".join(errors) or "no valid choices"
            self.last_error = last_error
        raise RuntimeError(f"all n-choice strategy samples failed: {last_error}")

    def parse(self, raw: str) -> ReservoirPlan:
        obj = self._load_json(raw)
        allowed = {"rationale", "strategies", "update_bias", "bias"}
        unknown = set(obj) - allowed
        if unknown:
            raise ValueError(f"unknown top-level keys: {sorted(unknown)}")

        strategies_src = obj.get("strategies")
        if not isinstance(strategies_src, list) or not strategies_src:
            raise ValueError("response must contain a non-empty 'strategies' list")

        strategies: list[SearchSpaceAtom] = []
        for item in strategies_src:
            source = item.get("trust_region") if isinstance(item, dict) else item
            strategies.extend(self._parse_search_source(source))

        if len(strategies) > self.config.n_strategies:
            strategies = strategies[: self.config.n_strategies]

        bias = self._parse_bias(obj.get("update_bias", obj.get("bias")))
        return ReservoirPlan(
            strategies=strategies,
            bias_dsl=bias,
            rationale=obj.get("rationale"),
            raw=raw,
        )

    def parse_one(self, raw: str) -> tuple[SearchSpaceAtom, Optional[BiasAtom], Optional[str]]:
        """Parse one independent-strategy response."""
        obj = self._load_json(raw)
        allowed = {"rationale", "strategy", "trust_region", "update_trust_region", "update_bias", "bias", "name", "strategies"}
        unknown = set(obj) - allowed
        if unknown:
            raise ValueError(f"unknown top-level keys: {sorted(unknown)}")

        if isinstance(obj.get("strategies"), list):
            items = obj["strategies"]
            if not items:
                raise ValueError("strategies list is empty")
            source = items[0].get("trust_region") if isinstance(items[0], dict) else items[0]
        else:
            source = obj.get("trust_region", obj.get("update_trust_region", obj.get("strategy")))
            if isinstance(source, dict):
                source = source.get("trust_region") or source.get("dsl")

        atoms = self._parse_search_source(source)
        if not atoms:
            raise ValueError("no strategy atom parsed")
        bias = self._parse_bias(obj.get("update_bias", obj.get("bias")))
        return atoms[0], bias, obj.get("rationale")

    def _parse_search_source(self, source: Any) -> list[SearchSpaceAtom]:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("strategy must be a DSL string or {'trust_region': DSL}")
        atom = safe_exec_dsl(
            source,
            whitelist=self.config.atoms_whitelist,
            expect_kind=SearchSpaceAtom,
        )
        atoms = atom.children if isinstance(atom, Or) else [atom]
        out: list[SearchSpaceAtom] = []
        for child in atoms:
            errors = validate_search_atom(child, sample_timeout_s=self.config.sample_timeout_s)
            if errors:
                raise ValueError(f"invalid strategy {source!r}: {errors}")
            out.append(child)
        return out

    def _parse_bias(self, source: Any) -> Optional[BiasAtom]:
        if not source:
            return None
        if not isinstance(source, str):
            raise ValueError("update_bias must be a DSL string")
        bias = safe_exec_dsl(
            source,
            whitelist=self.config.atoms_whitelist,
            expect_kind=BiasAtom,
        )
        errors = validate_bias_atom(bias)
        if errors:
            raise ValueError(f"invalid bias {source!r}: {errors}")
        return bias

    def _load_json(self, raw: str) -> dict[str, Any]:
        if not raw or not raw.strip():
            raise ValueError("empty LLM response")
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"expected JSON object, got {type(obj).__name__}")
        return obj

    def _context_text(self, context: dict[str, Any]) -> tuple[str, str, str, str]:
        history = context.get("history", [])
        if len(history) > self.config.history_max_in_prompt:
            half = self.config.history_max_in_prompt // 2
            history = history[:half] + history[-(self.config.history_max_in_prompt - half):]
        history_text = "\n".join(str(row) for row in history) or "(empty)"
        antigen = str(context.get("antigen_context", {}))[:1200]
        best_sequence = context.get("best_sequence", "(unknown)")
        best_value = context.get("best_value", "(unknown)")
        return history_text, antigen, best_sequence, best_value

    def _build_prompt(self, context: dict[str, Any], last_error: Optional[str]) -> str:
        history_text, antigen, best_sequence, best_value = self._context_text(context)
        feedback = last_error or "(none)"
        n = self.config.n_strategies
        budget = self.config.per_strategy_budget
        return f"""
You are planning one AntBO Bayesian-optimization step for antibody CDRH3.
Return exactly {n} search strategies z conditioned on history C_t.

Each strategy must be one AntBO DSL expression, e.g.
  LocalSearch('ARDYGNYWYFD', radius=3, restart=2, steps=99)
  NeighborSampling('ARDYGNYWYFD', radius=4, mut_pr=0.45, budget={budget})

The downstream algorithm executes each strategy pool in parallel/batched mode,
keeps one best candidate from each pool, then selects the final protein
candidate by acquisition softmax/argmax.

Constraints:
- Sequence length is 11.
- Amino-acid alphabet is ACDEFGHIKLMNPQRSTVWY.
- Keep each strategy budget <= {budget}.
- Output ONLY JSON with keys: rationale, strategies, update_bias.
- strategies must be a list of objects: {{"name": "...", "trust_region": "..."}}.

Recommended bias DSL:
  MaxCysteine(1) + MaxHydrophobicRun(4) + MaxAromatic(2) + NetChargeRange(-1, 2) + NoNGlycosylation()

Antigen context:
{antigen}

Best so far: {best_sequence}, score={best_value}

History C_t:
{history_text}

Previous parser/validation error:
{feedback}
""".strip()

    def _build_n_choices_prompt(
        self,
        context: dict[str, Any],
        n: int,
        last_error: Optional[str],
    ) -> str:
        history_text, antigen, best_sequence, best_value = self._context_text(context)
        feedback = last_error or "(none)"
        budget = self.config.per_strategy_budget
        return f"""
You are independently sampling ONE search strategy z from p_theta(z | C_t) for
AntBO Bayesian optimization of antibody CDRH3.

The API will request {n} independent completions from this same prompt. Each
completion must return exactly one strategy. Do not return a list and do not
assume the contents of other completions.

A strategy is one AntBO DSL expression, e.g.
  LocalSearch('ARDYGNYWYFD', radius=3, restart=1, steps={max(1, budget - 1)})
  NeighborSampling('ARDYGNYWYFD', radius=4, mut_pr=0.45, budget={budget})
  LatinHyperCubeSampling(num={budget})

Constraints:
- Sequence length is 11.
- Amino-acid alphabet is ACDEFGHIKLMNPQRSTVWY.
- Keep this single strategy budget <= {budget}.
- Output ONLY JSON with keys: rationale, trust_region, update_bias.
- trust_region must be one DSL string, not a list.

Recommended optional update_bias:
  MaxCysteine(1) + MaxHydrophobicRun(4) + MaxAromatic(2) + NetChargeRange(-1, 2) + NoNGlycosylation()

Antigen context:
{antigen}

Best so far: {best_sequence}, score={best_value}

History C_t:
{history_text}

Previous parser/validation error:
{feedback}
""".strip()

    def _build_single_strategy_prompt(
        self,
        context: dict[str, Any],
        sample_idx: int,
        n: int,
        last_error: Optional[str],
    ) -> str:
        history_text, antigen, best_sequence, best_value = self._context_text(context)
        feedback = last_error or "(none)"
        budget = self.config.per_strategy_budget
        return f"""
You are independently sampling ONE search strategy z from p_theta(z | C_t) for
AntBO Bayesian optimization of antibody CDRH3.

This is independent parallel sample {sample_idx + 1} of {n}. Other LLM calls are
sampling their own strategies at the same time. Do not assume their outputs and
do not return a list. Return exactly one strategy.

A strategy is one AntBO DSL expression, e.g.
  LocalSearch('ARDYGNYWYFD', radius=3, restart=1, steps={max(1, budget - 1)})
  NeighborSampling('ARDYGNYWYFD', radius=4, mut_pr=0.45, budget={budget})
  LatinHyperCubeSampling(num={budget})

Constraints:
- Sequence length is 11.
- Amino-acid alphabet is ACDEFGHIKLMNPQRSTVWY.
- Keep this single strategy budget <= {budget}.
- Output ONLY JSON with keys: rationale, trust_region, update_bias.
- trust_region must be one DSL string, not a list.

Recommended optional update_bias:
  MaxCysteine(1) + MaxHydrophobicRun(4) + MaxAromatic(2) + NetChargeRange(-1, 2) + NoNGlycosylation()

Antigen context:
{antigen}

Best so far: {best_sequence}, score={best_value}

History C_t:
{history_text}

Previous parser/validation error:
{feedback}
""".strip()
