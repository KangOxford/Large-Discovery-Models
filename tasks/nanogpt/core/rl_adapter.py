"""RL environment adapter for the nanoGPT knob-search task.

Two modes:

``real``
    One environment step = one real ``train.py`` run for its full wall-clock
    budget on one GPU, scored by the measured ``val_bpb``. See
    :mod:`tasks.nanogpt.core.rl_real`.

``mock``
    Same plumbing, a deliberately **structure-free** score. Use it to exercise
    parsing, admission, de-duplication, selection, reward and the Slime bridge
    without a GPU. It is not a training environment; see
    :class:`MockPlumbingEvaluator`.

Note on scope: this adapter drives the *stateless* proposal model -- policy text
becomes a complete knob dictionary, which is admitted and evaluated on its own.
It deliberately does **not** reuse ``engine_adapters.NanogptEvaluator``. That
one evaluates ``state_id`` references which a task-local expander
(``NanogptIterationExpander``) must first materialise inside an
``OperationSearchEngine``, and that indirection is what previously made this
task look incompatible with the shared environment. A full config needs no such
indirection, so nothing here depends on the search engine.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ldm_tts.contracts import Candidate, EvaluationResult

from tasks.nanogpt.core import rl_encoder, rl_knobs as knobs_mod, rl_task


class MockPlumbingEvaluator:
    """A deterministic, **information-free** stand-in for the real trainer.

    ``val_bpb`` is a hash of the candidate's canonical key mapped into a
    plausible band. This is intentional: it is uncorrelated with the config, so
    no policy can learn anything from it, and therefore nobody can mistake a
    mock run for a result.

    The predecessor of this module shipped a *plausible-looking* analytic
    val_bpb formula instead. It was never fitted, people trained on it, and its
    ranking turned out to be inverted against the real trainer (Spearman -0.60;
    6 of 7 policy proposals were 17-50 sigma regressions, and the one knob the
    formula treated as exactly inert cost 0.023 bpb in reality). A mock that
    cannot be mistaken for a surrogate is the fix.
    """

    #: Band that brackets real measurements of this task (baseline ~0.996).
    BAND = (0.95, 1.35)

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        digest = hashlib.sha256(candidate.canonical_key.encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        low, high = self.BAND
        config = candidate.payload["config"]
        geometry = knobs_mod.geometry(config)
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={
                "val_bpb": low + unit * (high - low),
                # Real, cheap-to-derive diagnostics so the feedback transcript
                # has the shape the policy will see in real mode.
                "num_params_M": geometry["total_params"] / 1e6,
                "model_dim": geometry["model_dim"],
                "grad_accum_steps": geometry["grad_accum_steps"],
            },
            resource_usage={"benchmark_jobs": 1.0, "gpu_seconds": 0.0},
            metadata={"mock": True, "canonical_key": candidate.canonical_key},
        )


def build_rl_components(mode: str = "mock", **kwargs: Any) -> Any:
    """Assemble the task's RL adapter bundle.

    Recognised ``kwargs`` (all optional, normally supplied by an episode's
    ``real_kwargs``):

    ``free_knobs``, ``pinned``, ``time_budget``
        the instance -- which knobs the policy sets, what the rest are held at,
        and the trainer's wall-clock budget.
    ``reservoir_size``
        proposals the policy must emit per round.
    ``use_surrogate``
        whether to fit a GP to order candidates within a round (default True).
    ``output_dir``, ``eval_cache_file``, ``eval_gpus``, ``repo_root``,
    ``autoresearch_cache_dir``, ``run_command``, ``timeout_slack``,
    ``gpu_claim_timeout``, ``keep_run_dirs``
        real-mode evaluation plumbing; see
        :func:`tasks.nanogpt.core.rl_real.build_runner`.
    ``reference_val_bpb``
        the instance's measured default-config score, shown in the prompt.
    """

    if mode == "real":
        from tasks.nanogpt.core.rl_real import build_real_components

        return build_real_components(**kwargs)
    if mode != "mock":
        raise ValueError("nanogpt RL factory supports 'mock' and 'real' modes only")

    from ldm_rl.components import EnvComponents

    instance = rl_task.KnobInstance.from_kwargs(**kwargs)
    reservoir_size = int(kwargs.get("reservoir_size", 1) or 1)
    use_gp = bool(kwargs.get("use_surrogate", True))

    selector = encoder = None
    if use_gp:
        from tasks.nanogpt.core.rl_real import build_selector

        selector = build_selector(**kwargs)
        encoder = rl_encoder.NanogptSurrogateEncoder()

    def parse_action(text: str) -> list[Any]:
        return rl_task.parse_knob_proposals(text, expected_count=reservoir_size)

    return EnvComponents(
        task_spec=rl_task.describe_rl_task(
            instance, reservoir_size=reservoir_size, surrogate=use_gp
        ),
        domain=rl_task.NanogptKnobDomain(instance),
        evaluator=MockPlumbingEvaluator(),
        parse_action=parse_action,
        context={**instance.context(), "mock": True},
        selector=selector,
        surrogate_encoder=encoder,
    )


__all__ = ["MockPlumbingEvaluator", "build_rl_components"]
