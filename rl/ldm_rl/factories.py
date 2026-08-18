"""Build an ``LDMEnv`` from a registered LDM task.

Factories reuse the task's own ``describe_ldm_task`` and core adapters instead
of reimplementing domain logic, so environment semantics stay identical to
campaign semantics. Each factory imports its task lazily so building one task
never pulls in another task's dependencies.

Supported tasks and modes:

- ``ai4bio_mutation_effect_prediction`` (mock; real requires upstream paths)
- ``causal_discovery_discrete`` (mock; real requires upstream paths)
- ``small_molecule`` (mock only; real requires the vina binary and the
  torch/gpytorch/gauche/rdkit runtime)

Real mode factories take the same keyword arguments the task's own CLI does
(e.g. ``upstream_root``, ``data_dir``, ``cv_dir``, ``run_dir``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldm_tts.contracts import CandidateDomainAdapter, CandidateEvaluator, LDMTaskSpec

from ldm_rl.env import EnvConfig, LDMEnv


@dataclass(frozen=True)
class EnvComponents:
    """Task adapter bundle assembled by one factory."""

    task_spec: LDMTaskSpec
    domain: CandidateDomainAdapter
    evaluator: CandidateEvaluator
    parse_action: Any = None  # callable(text) -> list[payload] | None (spec-declared)
    context: dict[str, Any] | None = None
    selector: Any = None  # AcquisitionSelector | None
    surrogate_encoder: Any = None  # SurrogateEncoder | None


def build_env(
    task_id: str,
    *,
    mode: str = "mock",
    config: EnvConfig | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> LDMEnv:
    """Assemble an environment for a registered task.

    ``kwargs`` flow to the per-task factory (reservoir size, real-evaluator
    paths, ...). Use ``config`` to fix the episode lifecycle/reward policy;
    otherwise a default 8-round episode is created. ``context`` overlays the
    factory's default episode context (e.g. assay or case lists).
    """

    if mode not in {"mock", "real"}:
        raise ValueError("mode must be 'mock' or 'real'")
    factory = _TASK_FACTORIES.get(task_id)
    if factory is None:
        raise KeyError(
            f"no RL environment factory for task {task_id!r}; registered: "
            + ", ".join(sorted(_TASK_FACTORIES))
        )
    components = factory(mode=mode, **kwargs)
    if config is None:
        config = EnvConfig(
            iterations=8,
            reservoir_size=kwargs.get("reservoir_size", 2),
            evaluations_per_round=kwargs.get("evaluations_per_round", 1),
        )
    env_context = dict(components.context or {})
    if context:
        env_context.update(context)
    return LDMEnv(
        task_spec=components.task_spec,
        domain=components.domain,
        evaluator=components.evaluator,
        config=config,
        parse_action=components.parse_action,
        context=env_context or None,
        selector=components.selector,
        surrogate_encoder=components.surrogate_encoder,
    )


# ----------------------------------------------------------------- ai4bio


def _ai4bio_factory(mode: str = "mock", **kwargs: Any) -> EnvComponents:
    from ldm_tts.optimization.gp import RBFGPUCBSelector
    from tasks.ai4bio_mutation_effect_prediction.core import workflow as _wf
    from tasks.ai4bio_mutation_effect_prediction.core.candidate import (
        MutationPredictorCandidateDomain,
    )
    from tasks.ai4bio_mutation_effect_prediction.core.evaluator import (
        MLSBenchMutationEvaluator,
        MockMutationEvaluator,
    )
    from tasks.ai4bio_mutation_effect_prediction.core.surrogate import (
        FEATURE_VERSION,
        PredictorSpecEncoder,
    )

    reservoir_size = int(kwargs.get("reservoir_size", 2))
    args = _wf.parse_args(["--mock"] if mode == "mock" else [])
    args.reservoir_size = reservoir_size
    spec = _wf.describe_ldm_task(args)

    domain = MutationPredictorCandidateDomain()
    if mode == "mock":
        evaluator = MockMutationEvaluator()
    else:
        missing = [
            name
            for name in ("upstream_root", "data_dir", "cv_dir")
            if kwargs.get(name) is None
        ]
        if missing:
            raise ValueError(
                "real ai4bio mode requires kwargs: " + ", ".join(missing)
            )
        evaluator = MLSBenchMutationEvaluator(
            upstream_root=kwargs["upstream_root"],
            data_dir=kwargs["data_dir"],
            cv_dir=kwargs["cv_dir"],
            run_dir=kwargs.get("run_dir") or Path("rl_runs/ai4bio"),
            timeout_seconds=float(kwargs.get("evaluation_timeout", 3540.0)),
            evaluator_python=str(kwargs.get("evaluator_python") or os.sys.executable),
        )
    context = {"assays": list(_wf.OFFICIAL_ASSAYS)} if mode == "real" else None
    encoder = PredictorSpecEncoder()
    selector = RBFGPUCBSelector(
        objective_name=spec.objectives[0].name,
        beta=float(kwargs.get("acquisition_beta", 1.0)),
        feature_version=FEATURE_VERSION,
    )
    return EnvComponents(
        task_spec=spec,
        domain=domain,
        evaluator=evaluator,
        context=context,
        selector=selector,
        surrogate_encoder=encoder,
    )


# ------------------------------------------------------- causal discovery


def _cd_factory(mode: str = "mock", **kwargs: Any) -> EnvComponents:
    from ldm_tts.optimization.gp import RBFGPUCBSelector
    from tasks.causal_discovery_discrete.core import workflow as _wf
    from tasks.causal_discovery_discrete.core.candidate import (
        CausalAlgorithmCandidateDomain,
        normalize_algorithm_spec,
    )
    from tasks.causal_discovery_discrete.core.evaluator import (
        MLSBenchCausalEvaluator,
        MockCausalEvaluator,
    )
    from tasks.causal_discovery_discrete.core.surrogate import (
        FEATURE_VERSION,
        CausalSpecEncoder,
    )
    from ldm_rl.parsing import parse_candidate_list

    reservoir_size = int(kwargs.get("reservoir_size", 2))
    args = _wf.parse_args(["--mock"] if mode == "mock" else [])
    args.reservoir_size = reservoir_size
    spec = _wf.describe_ldm_task(args)

    domain = CausalAlgorithmCandidateDomain()

    def parse_action(text: str) -> list[Any]:
        # The declared parser normalizes one payload, not raw text, so wrap it.
        payloads = parse_candidate_list(text, expected_count=reservoir_size)
        return [normalize_algorithm_spec(item) for item in payloads]

    if mode == "mock":
        evaluator = MockCausalEvaluator()
    else:
        if kwargs.get("upstream_root") is None:
            raise ValueError("real causal_discovery mode requires upstream_root")
        evaluator = MLSBenchCausalEvaluator(
            upstream_root=kwargs["upstream_root"],
            run_dir=kwargs.get("run_dir") or Path("rl_runs/causal_discovery"),
            timeout_seconds=float(kwargs.get("evaluation_timeout", 3540.0)),
            evaluator_python=str(kwargs.get("evaluator_python") or os.sys.executable),
        )
    context = {"cases": list(_wf.OFFICIAL_CASES)} if mode == "real" else None
    encoder = CausalSpecEncoder()
    selector = RBFGPUCBSelector(
        objective_name=spec.objectives[0].name,
        beta=float(kwargs.get("acquisition_beta", 1.0)),
        feature_version=FEATURE_VERSION,
    )
    return EnvComponents(
        task_spec=spec,
        domain=domain,
        evaluator=evaluator,
        parse_action=parse_action,
        context=context,
        selector=selector,
        surrogate_encoder=encoder,
    )


# --------------------------------------------------------- small molecule


def _small_molecule_factory(mode: str = "mock", **kwargs: Any) -> EnvComponents:
    """Mock-only adapter over the task's canonicalize + mock scorers.

    Uses the task's own ``describe_ldm_task`` with the ``llm_order`` method
    (``surrogate kind=none``), so no torch/gpytorch/gauche is imported. The
    environment evaluates proposals in reservoir order; the task's own EHVI-GP
    selection path lives in ``ldm_tilted_case2.loop`` and is intentionally not
    pulled in here.

    ``canonicalize_smiles`` and ``parse_m1_direct_smiles`` are reimplemented
    inline because their modules import ``ldm_tilted_case2.config`` (which
    imports ``core.gp`` -> torch) at module scope; the inline versions keep the
    same semantics on the no-RDKit path so the env stays dependency-light.
    """

    if mode != "mock":
        raise ValueError(
            "small_molecule RL factory supports mock mode only; real mode needs "
            "the vina binary plus the torch/gpytorch/gauche/rdkit runtime."
        )

    import hashlib

    from ldm_tts.contracts import (
        Candidate,
        CandidateRejection,
        EvaluationResult,
        RawProposal,
    )
    from ldm_tts.transport.parsing import (
        load_json_object,
        reject_keys,
        require_list,
        require_str,
    )
    from tasks.small_molecule.core import workflow as _wf

    args = _wf.parse_args(["--mock", "--method", "m1_stratified_direct_llm_only"])
    spec = _wf.describe_ldm_task(args)
    max_len = int(spec.candidate_domain.constraints.get("max_smiles_len", 80))
    vina_fn, activity_fn = _wf.build_mock_scorers()
    banned_keys = set(
        spec.response_spaces[0].metadata.get(
            "banned_score_keys",
            [
                "score",
                "objective_score",
                "constraint_score",
                "acquisition_score",
                "uncertainty",
                "proxy_value",
            ],
        )
    )

    def canonicalize_smiles(smiles: Any) -> str | None:
        # Same contract as tasks.small_molecule.core.ldm_tilted_case2
        # .canonicalize.canonicalize_smiles on the no-RDKit fallback path.
        text = str(smiles or "").strip()
        if not text or "." in text:
            return None
        try:
            from rdkit import Chem, RDLogger

            RDLogger.DisableLog("rdApp.*")
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                return None
            canonical = Chem.MolToSmiles(mol, canonical=True)
            return None if "." in canonical else canonical
        except ImportError:
            return text

    class SmallMoleculeDomain:
        def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
            payload = proposal.payload
            if isinstance(payload, str):
                smiles, rationale = payload, ""
            elif isinstance(payload, dict):
                smiles = payload.get("smiles")
                rationale = str(payload.get("rationale", ""))
            else:
                return CandidateRejection(
                    "invalid_payload",
                    "payload must be a SMILES string or an object with 'smiles'",
                    proposal.source,
                )
            canonical = canonicalize_smiles(smiles)
            if canonical is None:
                return CandidateRejection(
                    "invalid_smiles",
                    "SMILES could not be canonicalized",
                    proposal.source,
                )
            if len(canonical) > max_len:
                return CandidateRejection(
                    "smiles_too_long",
                    f"SMILES exceeds {max_len} characters",
                    proposal.source,
                )
            return Candidate(
                candidate_id="mol-" + hashlib.sha256(canonical.encode()).hexdigest()[:12],
                payload={"smiles": canonical, "rationale": rationale},
                canonical_key=canonical,
                source=proposal.source,
            )

    class SmallMoleculeMockEvaluator:
        def evaluate(self, candidate: Candidate) -> EvaluationResult:
            smiles = candidate.payload["smiles"]
            return EvaluationResult(
                candidate.candidate_id,
                "succeeded",
                metrics={
                    "vina": float(vina_fn([smiles])[0]),
                    "activity": float(activity_fn([smiles])[0]),
                },
                resource_usage={"benchmark_jobs": 1},
            )

    def parse_action(text: str) -> list[Any]:
        # Same contract as tasks.small_molecule.core.ldm_tilted_case2.schemas
        # .parse_m1_direct_smiles.
        data = load_json_object(text)
        reject_keys(data, banned_keys)
        rows = require_list(data, "direct_smiles")
        return [
            {"smiles": require_str(row, "smiles"), "rationale": str(row.get("rationale", ""))}
            for row in rows
        ]

    return EnvComponents(
        task_spec=spec,
        domain=SmallMoleculeDomain(),
        evaluator=SmallMoleculeMockEvaluator(),
        parse_action=parse_action,
    )


_TASK_FACTORIES = {
    "ai4bio_mutation_effect_prediction": _ai4bio_factory,
    "causal_discovery_discrete": _cd_factory,
    "small_molecule": _small_molecule_factory,
}


__all__ = ["EnvComponents", "build_env"]
