"""RL environment adapter for the small-molecule task.

Mock-only adapter over the task's canonicalize + mock scorers. Uses the task's
own ``describe_ldm_task`` with the ``llm_order`` method (``surrogate kind=none``)
so no torch/gpytorch/gauche is imported. The environment evaluates proposals in
reservoir order; the task's own EHVI-GP selection path lives in
``ldm_tilted_case2.loop`` and is intentionally not pulled in here.

``canonicalize_smiles`` and ``parse_m1_direct_smiles`` are reimplemented inline
because their modules import ``ldm_tilted_case2.config`` (which imports
``core.gp`` -> torch) at module scope; the inline versions keep the same
semantics on the no-RDKit path so the env stays dependency-light.

The ``ldm_rl`` import is deferred to call time so this module stays importable
without the ``rl/`` directory on ``sys.path``.
"""

from __future__ import annotations

from typing import Any


def build_rl_components(mode: str = "mock", **kwargs: Any) -> Any:
    if mode == "real":
        from tasks.small_molecule.core.rl_real import build_real_components

        return build_real_components(**kwargs)
    if mode != "mock":
        raise ValueError("small_molecule RL factory supports mock and real modes only.")

    import hashlib

    from ldm_rl.components import EnvComponents
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
    if kwargs.get("seed") is not None:
        args.seed = int(kwargs["seed"])
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
