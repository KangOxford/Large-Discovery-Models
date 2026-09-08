"""Real-mode RL adapter for the nanoGPT knob-search task.

Assembles the components ``LDMEnv`` needs so that one environment step is one
*measured* nanoGPT training run:

* :class:`NanogptCandidateEvaluator` -- runs ``train.py`` for its wall-clock
  budget and reports the measured ``val_bpb``;
* :class:`NanogptKnobDomain` -- admission (from ``rl_task``);
* :class:`~tasks.nanogpt.core.rl_encoder.NanogptSurrogateEncoder` +
  :class:`MinimisingGPUCBSelector` -- which of the admitted proposals to spend
  the budget on first.

The GP here is **only** used to order candidates inside a round. It never
produces the reward: the reward is the measured objective. This is the opposite
of the small-molecule task's default (``reward="acquisition"``), and it is
deliberate for nanoGPT -- an analytic/fitted stand-in for this objective was
measured to invert the true ranking, so nothing but a real run is trusted to
score a proposal.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ldm_tts.contracts import Candidate, EvaluationResult
from ldm_tts.optimization.gp import RBFGPUCBSelector
from ldm_tts.optimization.records import BOObservation

from tasks.nanogpt.core import rl_encoder, rl_eval, rl_knobs as knobs_mod, rl_task

#: How a :class:`~tasks.nanogpt.core.rl_eval.RunOutcome` failure maps onto the
#: evaluation-status vocabulary of ``ldm_tts.contracts``.
#:
#: ``rejected_by_trainer`` is ``invalid`` rather than ``failed`` because the
#: config was never runnable -- keeping it distinct is what lets a training run
#: tell "the policy proposed something illegal" apart from "the machine broke".
_STATUS_BY_FAILURE = {
    "rejected_by_trainer": "invalid",
    "timed_out": "timed_out",
    "oom": "failed",
    "loss_diverged": "failed",
    "no_metrics": "failed",
    "crashed": "failed",
}


class NanogptCandidateEvaluator:
    """Measure one candidate config with the real trainer."""

    def __init__(self, runner: rl_eval.RealNanogptEvaluator) -> None:
        self.runner = runner

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        config = candidate.payload["config"]
        budget = float(candidate.payload.get("time_budget", self.runner.time_budget))
        outcome = self.runner.evaluate_config(config, budget)

        metadata = {
            "canonical_key": outcome.canonical_key,
            "failure_kind": outcome.failure_kind,
            "run_dir": outcome.run_dir,
            "gpu": outcome.gpu,
            "cached": outcome.cached,
            "summary": candidate.payload.get("summary", ""),
        }
        if not outcome.ok:
            return EvaluationResult(
                candidate.candidate_id,
                _STATUS_BY_FAILURE.get(outcome.failure_kind, "failed"),
                metrics={k: float(v) for k, v in outcome.metrics.items()},
                error=outcome.error or outcome.failure_kind,
                resource_usage={"gpu_seconds": float(outcome.wall_seconds)},
                metadata=metadata,
            )
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={k: float(v) for k, v in outcome.metrics.items()},
            resource_usage={
                "benchmark_jobs": 1.0,
                # A cache hit really did cost nothing, and saying so keeps any
                # cost accounting downstream honest.
                "gpu_seconds": 0.0 if outcome.cached else float(outcome.wall_seconds),
            },
            artifacts={"run_dir": outcome.run_dir} if outcome.run_dir else {},
            metadata=metadata,
        )


class MinimisingGPUCBSelector:
    """``RBFGPUCBSelector`` corrected for a minimised objective.

    ``RBFGPUCBSelector`` builds its acquisition with ``minimize=(False,)``
    hardcoded (``ldm_tts/optimization/gp.py``), and
    ``BOObservation.from_observation`` stores the **raw** metric value with no
    orientation applied. Composing the two on a minimised objective therefore
    ranks the candidate with the *highest* predicted ``val_bpb`` first, i.e. it
    spends the budget on the proposal the GP thinks is worst.

    No existing task hits this (small-molecule uses its own tilted selector;
    nanoGPT's campaign selector sorts ascending explicitly), so rather than
    change shared behaviour this wrapper negates the objective on the way in.
    The inner GP then maximises ``-val_bpb``, which is minimising ``val_bpb``,
    and ``predictions`` keep the inner (negated) scale -- they are only ever
    used for ordering, never as a reward.
    """

    def __init__(self, inner: RBFGPUCBSelector) -> None:
        self.inner = inner

    def describe(self):
        return self.inner.describe()

    def fit(self, history: Sequence[BOObservation]) -> None:
        self.inner.fit(
            [
                BOObservation(
                    candidate_id=item.candidate_id,
                    objectives=tuple(
                        None if value is None else -float(value)
                        for value in item.objectives
                    ),
                    feature=item.feature,
                    metadata={**dict(item.metadata), "oriented": "negated"},
                )
                for item in history
            ]
        )

    def select(self, candidates, representations, *, count: int = 1):
        return self.inner.select(candidates, representations, count=count)


def build_selector(**kwargs: Any) -> MinimisingGPUCBSelector:
    """GP-UCB selector over the measured history, oriented for minimisation.

    ``beta`` is the exploration weight on the posterior std. It defaults above
    1.0 because the GP starts almost empty here -- unlike a cheap objective,
    this task cannot afford many observations, so early rounds should be
    treated as genuinely uncertain rather than trusted.
    """

    inner = RBFGPUCBSelector(
        objective_name=rl_task.OBJECTIVE_NAME,
        beta=float(kwargs.get("gp_beta", 2.0)),
        lengthscale=float(kwargs.get("gp_lengthscale", 1.5)),
        noise=float(kwargs.get("gp_noise", kwargs.get("noise_floor", 1.3e-3) ** 2)),
        prior_std=float(kwargs.get("gp_prior_std", 0.05)),
        feature_version=rl_encoder.FEATURE_VERSION,
    )
    return MinimisingGPUCBSelector(inner)


def build_runner(**kwargs: Any) -> rl_eval.RealNanogptEvaluator:
    """Construct the real trainer runner from episode ``real_kwargs``."""

    output_dir = kwargs.get("output_dir") or os.path.join(
        os.path.expanduser("~"), ".cache", "nanogpt_rl"
    )
    return rl_eval.RealNanogptEvaluator(
        output_dir=output_dir,
        repo_root=kwargs.get("repo_root"),
        time_budget=float(
            kwargs.get("time_budget", knobs_mod.DEFAULT_TIME_BUDGET)
        ),
        # Shared across every episode and every worker of a run, so a config is
        # measured once per run rather than once per episode.
        cache_file=kwargs.get("eval_cache_file"),
        eval_gpus=kwargs.get("eval_gpus"),
        run_command=kwargs.get("run_command"),
        timeout_slack=float(
            kwargs.get("timeout_slack", rl_eval.DEFAULT_TIMEOUT_SLACK)
        ),
        gpu_claim_timeout=float(kwargs.get("gpu_claim_timeout", 3600.0)),
        autoresearch_cache_dir=kwargs.get("autoresearch_cache_dir"),
        keep_run_dirs=bool(kwargs.get("keep_run_dirs", True)),
    )


def reference_config(instance: rl_task.KnobInstance) -> dict[str, Any]:
    """The config an episode is scored against: free knobs at their defaults."""

    return knobs_mod.validate(
        knobs_mod.with_defaults(
            instance.free_knobs,
            {name: knobs_mod.DEFAULTS[name] for name in instance.free_knobs},
            instance.pinned,
        )
    )


def measure_reference(
    instance: rl_task.KnobInstance,
    runner: rl_eval.RealNanogptEvaluator,
) -> rl_eval.RunOutcome:
    """Measure the instance's reference config (cached after the first call).

    This is the fixed zero point of the ``improvement`` reward. Measuring it
    once per instance -- offline, before training -- is what stops the episode's
    own first proposal from becoming the reference, which would pay the policy
    for opening badly. See the ``baseline is None`` branch of
    ``ldm_rl.env.LDMEnv._reward``.
    """

    return runner.evaluate_config(reference_config(instance), instance.time_budget)


def build_real_components(**kwargs: Any) -> Any:
    """Real-mode ``EnvComponents`` for one instance."""

    from ldm_rl.components import EnvComponents

    instance = rl_task.KnobInstance.from_kwargs(**kwargs)
    reservoir_size = int(kwargs.get("reservoir_size", 1) or 1)
    use_gp = bool(kwargs.get("use_surrogate", True))

    runner = build_runner(**kwargs)
    spec = rl_task.describe_rl_task(
        instance, reservoir_size=reservoir_size, surrogate=use_gp
    )

    selector = encoder = None
    if use_gp:
        selector = build_selector(**kwargs)
        encoder = rl_encoder.NanogptSurrogateEncoder()

    def parse_action(text: str) -> list[Any]:
        return rl_task.parse_knob_proposals(text, expected_count=reservoir_size)

    context = instance.context()
    reference = kwargs.get("reference_val_bpb")
    if reference is not None:
        context["reference_val_bpb"] = float(reference)
        context["reference_note"] = (
            "measured val_bpb of this instance's default configuration; your "
            "reward is how far below this you get"
        )

    return EnvComponents(
        task_spec=spec,
        domain=rl_task.NanogptKnobDomain(instance),
        evaluator=NanogptCandidateEvaluator(runner),
        parse_action=parse_action,
        context=context,
        selector=selector,
        surrogate_encoder=encoder,
    )


__all__ = [
    "MinimisingGPUCBSelector",
    "NanogptCandidateEvaluator",
    "build_real_components",
    "build_runner",
    "build_selector",
    "measure_reference",
    "reference_config",
]
