"""Core utilities for direct LLM antibody generation baselines.

The methods in this module intentionally do not expose LDM search-function
atoms such as LocalSearch or NeighborSampling. The LLM emits antibody
sequences directly as JSON lists of strings.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Iterable

import numpy as np
import torch

from bo.ldm.llm.client import LLMClient
from bo.ldm_light.ldm_acq import (
    AA,
    fit_gp_and_make_acquisition,
    indices_to_seqs,
    passes_developability,
    random_candidates,
    seqs_to_indices,
    valid_seq,
)


DEFAULT_METHOD_NAMES = {
    "bo/ldm": "LDM_fn_seq_argmax",
    "bo/ldm_reservoir:softmax": "LDM_fn_par_softmax",
    "bo/ldm_reservoir:argmax": "LDM_fn_par_argmax",
    "bo/ldm_light/ldm_acq.py": "LDM_fn_one_argmax",
    "bo/ldm/llm/LLM_baseline.py": "LLM_rerank",
    "bo/llm_direct:LLM_gen": "LLM_gen",
    "bo/llm_direct:LDM_gen_softmax": "LDM_gen_softmax",
    "bo/llm_direct:LDM_gen_argmax": "LDM_gen_argmax",
}


def _history_payload(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    best = [
        {"sequence": row["LastProtein"], "score": float(row["LastValue"])}
        for row in sorted(rows, key=lambda r: float(r["LastValue"]))[:top_k]
    ]
    recent = [
        {
            "sequence": row["LastProtein"],
            "score": float(row["LastValue"]),
            "best_so_far": float(row["BestValue"]),
            "iter": int(row["Index"]),
        }
        for row in rows[-top_k:]
    ]
    return {
        "num_observed": len(rows),
        "best": best,
        "recent": recent,
    }


def build_direct_generation_prompt(
    *,
    antigen: str,
    seq_len: int,
    num_sequences: int,
    observed: Iterable[str],
    rows: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    top_k: int,
    mode: str,
) -> str:
    """Build a JSON prompt for direct sequence generation.

    The required model output is a top-level JSON list of amino-acid strings,
    for example ``["ADGHTKQNPRA"]``.
    """

    payload = {
        "task": "Direct CDRH3 antibody sequence generation for AntBO.",
        "method": mode,
        "objective": "Minimize Absolut binding energy. Lower true score is better.",
        "important_difference_from_LDM": (
            "Do not output search functions, code, LocalSearch, NeighborSampling, "
            "LatinHyperCubeSampling, or explanations. Generate antibody strings directly."
        ),
        "antigen": antigen,
        "constraints": {
            "length": seq_len,
            "alphabet": AA,
            "num_sequences": int(num_sequences),
            "do_not_repeat": sorted(set(observed))[-300:],
            "developability_filter_used_by_code": {
                "max_cysteine": 1,
                "max_hydrophobic_run": 4,
                "max_aromatic_FWY": 2,
                "net_charge_range": [-1.0, 2.0],
                "forbid_n_glycosylation_NXS_or_NXT": True,
            },
        },
        "history": _history_payload(rows, top_k),
        "antigen_context": antigen_context or {},
        "required_output": [
            "ADGHTKQNPRA",
        ],
        "output_rules": [
            "Return JSON only.",
            "The top-level JSON value must be a list of strings.",
            "Each string must be one antibody CDRH3 sequence.",
            "Do not include ids, scores, rationales, markdown, or comments.",
            "Sequences must be novel relative to do_not_repeat.",
        ],
    }
    return json.dumps(payload, indent=2)


def extract_json_value(raw: str) -> Any:
    """Extract a JSON list or object from an LLM response."""

    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()

    if text.startswith("[") or text.startswith("{"):
        return json.loads(text)

    starts = [idx for idx in (text.find("["), text.find("{")) if idx >= 0]
    if not starts:
        raise ValueError("No JSON list or object found in LLM response")
    start = min(starts)
    open_ch = text[start]
    close_ch = "]" if open_ch == "[" else "}"
    end = text.rfind(close_ch)
    if end <= start:
        raise ValueError("Could not find matching JSON terminator")
    return json.loads(text[start:end + 1])


def _sequence_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip().upper()
    if isinstance(item, dict):
        for key in ("sequence", "antibody", "cdrh3", "x"):
            if key in item:
                return str(item[key]).strip().upper()
    return ""


def parse_generated_sequences(
    raw: str,
    *,
    seq_len: int,
    observed: Iterable[str],
    max_sequences: int | None = None,
) -> list[str]:
    """Parse and validate directly generated antibody sequences."""

    obj = extract_json_value(raw)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = None
        for key in ("antibodies", "sequences", "selected", "candidates"):
            if isinstance(obj.get(key), list):
                items = obj[key]
                break
        if items is None:
            raise ValueError("JSON object must contain antibodies/sequences/selected/candidates list")
    else:
        raise ValueError("LLM output must be a JSON list of strings")

    observed_set = {str(seq).upper() for seq in observed}
    out: list[str] = []
    used: set[str] = set()
    for item in items:
        seq = _sequence_from_item(item)
        if (
            valid_seq(seq, seq_len)
            and passes_developability(seq)
            and seq not in observed_set
            and seq not in used
        ):
            out.append(seq)
            used.add(seq)
            if max_sequences is not None and len(out) >= max_sequences:
                break
    return out


class CountingLLMClient:
    """Count API requests and completions while preserving the LLMClient API."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.total_calls = 0
        self.total_completions = 0

    def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
        self.total_calls += 1
        self.total_completions += 1
        return self.inner.call(prompt, temperature=temperature, timeout_s=timeout_s)

    def call_many(self, prompt: str, temperature: float, timeout_s: int, n: int) -> list[str]:
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        if hasattr(self.inner, "call_many"):
            self.total_calls += 1
            outputs = self.inner.call_many(prompt, temperature=temperature, timeout_s=timeout_s, n=n)
            self.total_completions += len(outputs)
            return list(outputs)
        if hasattr(self.inner, "_client") and hasattr(self.inner, "model"):
            self.total_calls += 1
            if hasattr(self.inner, "make_chat_completion_kwargs"):
                kwargs = self.inner.make_chat_completion_kwargs(
                    prompt,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    n=n,
                )
            else:
                kwargs = {
                    "model": self.inner.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "timeout": timeout_s,
                    "n": n,
                }
            response = self.inner._client.chat.completions.create(**kwargs)
            outputs = [choice.message.content or "" for choice in response.choices]
            self.total_completions += len(outputs)
            return outputs
        return [self.call(prompt, temperature=temperature, timeout_s=timeout_s) for _ in range(n)]

    def close(self) -> None:
        if hasattr(self.inner, "close"):
            self.inner.close()


class MockDirectLLMClient(LLMClient):
    """Deterministic no-network LLM used by tests and smoke runs."""

    def __init__(self, seed: int = 0, seq_len: int = 11) -> None:
        self.rng = random.Random(seed)
        self.seq_len = seq_len
        self.used: set[str] = set()

    def _next_seq(self) -> str:
        attempts = 0
        safe_alphabet = "ADEGHKNPQRST"
        while attempts < 10000:
            attempts += 1
            seq = "".join(self.rng.choice(safe_alphabet) for _ in range(self.seq_len))
            if seq not in self.used and valid_seq(seq, self.seq_len) and passes_developability(seq):
                self.used.add(seq)
                return seq
        raise RuntimeError("MockDirectLLMClient could not generate a valid sequence")

    def call(self, prompt: str, temperature: float, timeout_s: int) -> str:
        n = 1
        try:
            payload = extract_json_value(prompt)
            n = int(payload.get("constraints", {}).get("num_sequences", 1))
        except Exception:
            n = 1
        return json.dumps([self._next_seq() for _ in range(max(1, n))])

    def call_many(self, prompt: str, temperature: float, timeout_s: int, n: int) -> list[str]:
        return [json.dumps([self._next_seq()]) for _ in range(int(n))]


def _fallback_random_sequences(
    rng: random.Random,
    *,
    n: int,
    seq_len: int,
    observed: Iterable[str],
) -> list[str]:
    return [
        item["sequence"]
        for item in random_candidates(rng, n, seq_len, set(str(s).upper() for s in observed))
    ]


def propose_generated_batch(
    *,
    llm: CountingLLMClient,
    rng: random.Random,
    antigen: str,
    seq_len: int,
    num_sequences: int,
    observed: set[str],
    rows: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    args: Any,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ask the LLM for a direct JSON list of antibody strings."""

    errors: list[dict[str, Any]] = []
    prompt = ""
    for attempt in range(1, int(args.max_retries) + 1):
        prompt = build_direct_generation_prompt(
            antigen=antigen,
            seq_len=seq_len,
            num_sequences=num_sequences,
            observed=observed,
            rows=rows,
            antigen_context=antigen_context,
            top_k=int(args.history_top_k),
            mode=mode,
        )
        raw = llm.call(prompt, temperature=float(args.temperature), timeout_s=int(args.timeout_s))
        try:
            seqs = parse_generated_sequences(
                raw,
                seq_len=seq_len,
                observed=observed,
                max_sequences=num_sequences,
            )
            if len(seqs) >= num_sequences:
                return [{"sequence": seq, "score": None} for seq in seqs[:num_sequences]], {
                    "source": mode,
                    "attempt": attempt,
                    "prompt": prompt,
                    "raw_response": raw,
                    "parsed_sequences": seqs[:num_sequences],
                }
            raise ValueError(f"Only {len(seqs)} valid novel sequences returned")
        except Exception as exc:
            errors.append({"attempt": attempt, "error": str(exc), "raw_response": raw})

    if bool(args.fallback_random):
        seqs = _fallback_random_sequences(rng, n=num_sequences, seq_len=seq_len, observed=observed)
        return [{"sequence": seq, "score": None} for seq in seqs], {
            "source": f"{mode}_fallback_random",
            "prompt": prompt,
            "errors": errors,
        }
    raise RuntimeError(json.dumps(errors, indent=2))


def propose_generated_many(
    *,
    llm: CountingLLMClient,
    rng: random.Random,
    antigen: str,
    seq_len: int,
    n: int,
    observed: set[str],
    rows: list[dict[str, Any]],
    antigen_context: dict[str, Any] | None,
    args: Any,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample n direct antibody proposals, preferably with one call_many request."""

    errors: list[dict[str, Any]] = []
    prompt = ""
    observed_plus = set(observed)
    for attempt in range(1, int(args.max_retries) + 1):
        prompt = build_direct_generation_prompt(
            antigen=antigen,
            seq_len=seq_len,
            num_sequences=1,
            observed=observed_plus,
            rows=rows,
            antigen_context=antigen_context,
            top_k=int(args.history_top_k),
            mode=mode,
        )
        raw_outputs = llm.call_many(
            prompt,
            temperature=float(args.temperature),
            timeout_s=int(args.timeout_s),
            n=int(n),
        )
        seqs: list[str] = []
        for idx, raw in enumerate(raw_outputs):
            try:
                parsed = parse_generated_sequences(
                    raw,
                    seq_len=seq_len,
                    observed=observed_plus.union(seqs),
                    max_sequences=1,
                )
                seqs.extend(parsed)
            except Exception as exc:
                errors.append({"attempt": attempt, "choice": idx, "error": str(exc), "raw_response": raw})
        if len(seqs) >= n:
            seqs = seqs[:n]
            return [{"sequence": seq, "score": None} for seq in seqs], {
                "source": mode,
                "attempt": attempt,
                "prompt": prompt,
                "raw_outputs": raw_outputs,
                "parsed_sequences": seqs,
                "n_requested": int(n),
                "n_valid": len(seqs),
            }
        observed_plus.update(seqs)

    if bool(args.fallback_random):
        seqs = _fallback_random_sequences(rng, n=int(n), seq_len=seq_len, observed=observed)
        return [{"sequence": seq, "score": None} for seq in seqs], {
            "source": f"{mode}_fallback_random",
            "prompt": prompt,
            "errors": errors,
            "n_requested": int(n),
        }
    raise RuntimeError(json.dumps(errors, indent=2))


def score_candidates_by_acquisition(
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    gp_train_steps: int,
    device: str | torch.device,
) -> list[dict[str, Any]]:
    """Fit the GP on history and score direct candidates with EI."""

    if not candidates:
        return []
    gp, f_acq = fit_gp_and_make_acquisition(rows, gp_train_steps=gp_train_steps, device=device)
    device = device if isinstance(device, torch.device) else torch.device(device)
    seqs = [candidate["sequence"] for candidate in candidates]
    x = torch.tensor(seqs_to_indices(seqs), dtype=torch.float32, device=device)
    with torch.no_grad():
        posterior = gp.likelihood(gp(x))
        acq = f_acq(x).detach().cpu().numpy().reshape(-1)
        mu = posterior.mean.detach().cpu().numpy().reshape(-1)
        sigma = posterior.stddev.detach().cpu().numpy().reshape(-1)

    scored: list[dict[str, Any]] = []
    for candidate, acq_value, mu_value, sigma_value in zip(candidates, acq, mu, sigma):
        item = dict(candidate)
        item["acquisition_score"] = float(acq_value)
        item["acquisition_raw"] = float(acq_value)
        item["mu"] = float(mu_value)
        item["sigma"] = float(sigma_value)
        scored.append(item)
    return scored


def select_scored_candidates(
    scored: list[dict[str, Any]],
    *,
    batch_size: int,
    selection: str,
    eta: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[float]]:
    """Select scored candidates by acquisition argmax or softmax."""

    if not scored:
        raise ValueError("No scored candidates to select from")
    scores = np.array([float(item.get("acquisition_score", -np.inf)) for item in scored], dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        probs = np.ones(len(scored), dtype=float) / len(scored)
    elif selection == "argmax":
        probs = np.zeros(len(scored), dtype=float)
        probs[int(np.nanargmax(scores))] = 1.0
    elif selection == "softmax":
        safe_scores = np.where(finite, scores, scores[finite].min() - 1.0)
        eta = float(eta)
        if eta <= 0:
            probs = np.ones(len(scored), dtype=float) / len(scored)
        else:
            shifted = eta * (safe_scores - safe_scores.max())
            exp_scores = np.exp(shifted)
            probs = exp_scores / exp_scores.sum()
    else:
        raise ValueError("selection must be 'softmax' or 'argmax'")

    k = min(int(batch_size), len(scored))
    if selection == "argmax":
        ids = list(np.argsort(scores)[::-1][:k])
    else:
        ids = list(rng.choice(len(scored), size=k, replace=False, p=probs))
    return [int(i) for i in ids], [float(p) for p in probs]


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    if "sequence" not in out and "seq" in out:
        out["sequence"] = indices_to_seqs([out["seq"]])[0]
    return out
