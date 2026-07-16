"""Sequential Bayesian optimization over a generated string candidate pool."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from .acquisition import acquisition_score
from .gp import ExactStringGaussianProcess
from .kernels import NGramStringKernel
from .space import SearchSpace
from .trial import FrozenTrial, Trial


@dataclass(frozen=True)
class StrBOConfig:
    seed: int = 42
    n_initial: int = 3
    candidate_pool_size: int = 512
    local_fraction: float = 0.5
    mutation_rate: float = 0.55
    mutation_scale: float = 0.25
    acquisition: str = "ei"
    xi: float = 0.01
    beta: float = 1.96
    kernel_max_ngram: int = 5
    kernel_normalize: bool = True
    gp_noise: float = 1e-6


class BayesianOptimizationStudy:
    def __init__(
        self,
        *,
        study_name: str,
        space: SearchSpace,
        direction: str = "minimize",
        config: StrBOConfig | None = None,
    ) -> None:
        if direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'.")
        self.study_name = study_name
        self.space = space
        self.direction = direction
        self.config = config or StrBOConfig()
        self.trials: list[FrozenTrial] = []
        self._rng = random.Random(self.config.seed)

    @property
    def best_trial(self) -> FrozenTrial:
        completed = self._completed_trials()
        if not completed:
            raise ValueError("No completed trials are available.")
        reverse = self.direction == "maximize"
        return sorted(completed, key=lambda trial: trial.value, reverse=reverse)[0]

    @property
    def best_value(self) -> float:
        return self.best_trial.value

    @property
    def best_params(self) -> dict[str, Any]:
        return dict(self.best_trial.params)

    def optimize(self, objective: Callable[[Trial], float], *, n_trials: int) -> None:
        for number in range(n_trials):
            trial = self.ask(number)
            value = objective(trial)
            self.trials.append(trial.freeze(value))

    def ask(self, number: int) -> Trial:
        params = self._suggest_params()
        return Trial(number, params)

    def _suggest_params(self) -> dict[str, Any]:
        completed = self._completed_trials()
        n_initial = max(1, self.config.n_initial)
        if len(completed) < n_initial or len(completed) < 2:
            return self._random_unseen()

        try:
            return self._bayesian_suggestion(completed)
        except ValueError:
            return self._random_unseen()

    def _bayesian_suggestion(self, completed: list[FrozenTrial]) -> dict[str, Any]:
        model_values = [self._model_value(trial.value) for trial in completed]
        train_strings = [self.space.stringify(trial.params) for trial in completed]
        kernel = NGramStringKernel(max_ngram=self.config.kernel_max_ngram, normalize=self.config.kernel_normalize)
        gp = ExactStringGaussianProcess(kernel=kernel, noise=self.config.gp_noise)
        gp.fit(train_strings, model_values)

        candidates = self._candidate_pool(completed)
        candidate_strings = [self.space.stringify(candidate) for candidate in candidates]
        predictions = gp.predict_many(candidate_strings)
        best_model_value = min(model_values)

        scored = []
        for candidate, prediction in zip(candidates, predictions):
            score = acquisition_score(
                prediction.mean,
                prediction.stddev,
                best_model_value,
                kind=self.config.acquisition,
                xi=self.config.xi,
                beta=self.config.beta,
            )
            scored.append((score, self._rng.random(), candidate))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def _candidate_pool(self, completed: list[FrozenTrial]) -> list[dict[str, Any]]:
        target = max(1, self.config.candidate_pool_size)
        existing_keys = {self.space.key(trial.params) for trial in completed}
        pool: list[dict[str, Any]] = []
        pool_keys: set[str] = set()

        local_target = min(target, max(0, int(target * self.config.local_fraction)))
        best_trials = self._best_trials_for_local_search(completed, limit=min(5, len(completed)))
        attempts = 0
        while len(pool) < local_target and attempts < target * 30 and best_trials:
            attempts += 1
            base = self._rng.choice(best_trials).params
            candidate = self.space.mutate(
                base,
                self._rng,
                mutation_rate=self.config.mutation_rate,
                scale=self.config.mutation_scale,
            )
            self._add_candidate(candidate, pool, pool_keys, existing_keys)

        attempts = 0
        while len(pool) < target and attempts < target * 50:
            attempts += 1
            self._add_candidate(self.space.sample(self._rng), pool, pool_keys, existing_keys)

        if not pool:
            return [self.space.sample(self._rng)]
        return pool

    def _random_unseen(self) -> dict[str, Any]:
        existing_keys = {self.space.key(trial.params) for trial in self.trials}
        for _ in range(1000):
            candidate = self.space.sample(self._rng)
            if self.space.key(candidate) not in existing_keys:
                return self.space.canonicalize(candidate)
        return self.space.sample(self._rng)

    def _add_candidate(
        self,
        candidate: dict[str, Any],
        pool: list[dict[str, Any]],
        pool_keys: set[str],
        existing_keys: set[str],
    ) -> None:
        canonical = self.space.canonicalize(candidate)
        key = self.space.key(canonical)
        if key in existing_keys or key in pool_keys:
            return
        pool.append(canonical)
        pool_keys.add(key)

    def _best_trials_for_local_search(self, completed: list[FrozenTrial], *, limit: int) -> list[FrozenTrial]:
        reverse = self.direction == "maximize"
        return sorted(completed, key=lambda trial: trial.value, reverse=reverse)[:limit]

    def _completed_trials(self) -> list[FrozenTrial]:
        return [trial for trial in self.trials if trial.state == "COMPLETE" and math.isfinite(trial.value)]

    def _model_value(self, value: float) -> float:
        return float(value) if self.direction == "minimize" else -float(value)


def create_study(
    *,
    study_name: str,
    space: SearchSpace,
    direction: str = "minimize",
    config: StrBOConfig | None = None,
) -> BayesianOptimizationStudy:
    return BayesianOptimizationStudy(study_name=study_name, space=space, direction=direction, config=config)
