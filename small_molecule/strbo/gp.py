"""A tiny exact Gaussian process for string inputs.

The implementation is intentionally dependency-free. The BO loop evaluates only
one expensive objective at a time, so the training sets stay small enough for a
plain Cholesky decomposition in Python lists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kernels import NGramStringKernel


@dataclass(frozen=True)
class Prediction:
    mean: float
    variance: float

    @property
    def stddev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


class ExactStringGaussianProcess:
    def __init__(
        self,
        *,
        kernel: NGramStringKernel | None = None,
        noise: float = 1e-6,
        jitter: float = 1e-8,
        min_variance: float = 1e-12,
    ) -> None:
        if noise <= 0:
            raise ValueError("noise must be positive.")
        self.kernel = kernel or NGramStringKernel()
        self.noise = noise
        self.jitter = jitter
        self.min_variance = min_variance
        self._train_x: list[str] = []
        self._alpha: list[float] = []
        self._chol: list[list[float]] = []
        self._y_mean = 0.0
        self._y_scale = 1.0

    def fit(self, strings: list[str], values: list[float]) -> None:
        if not strings:
            raise ValueError("At least one training point is required.")
        if len(strings) != len(values):
            raise ValueError("strings and values must have the same length.")

        self._train_x = [str(item) for item in strings]
        self._y_mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - self._y_mean) ** 2 for value in values) / (len(values) - 1)
            self._y_scale = math.sqrt(variance) if variance > 1e-24 else 1.0
        else:
            self._y_scale = 1.0
        standardized = [(value - self._y_mean) / self._y_scale for value in values]

        base = self.kernel.matrix(self._train_x)
        diag_noise = self.noise
        last_error: Exception | None = None
        for attempt in range(8):
            matrix = [row[:] for row in base]
            extra_jitter = 0.0 if attempt == 0 else self.jitter * (10 ** (attempt - 1))
            for index in range(len(matrix)):
                matrix[index][index] += diag_noise + extra_jitter
            try:
                self._chol = _cholesky(matrix)
                self._alpha = _solve_cholesky(self._chol, standardized)
                return
            except ValueError as exc:
                last_error = exc
        raise ValueError(f"Could not factor GP kernel matrix: {last_error}")

    def predict(self, string: str) -> Prediction:
        if not self._train_x:
            raise ValueError("The GP must be fit before prediction.")

        x = str(string)
        cross = [self.kernel(train_item, x) for train_item in self._train_x]
        mean_standardized = sum(weight * alpha for weight, alpha in zip(cross, self._alpha))
        v = _forward_substitution(self._chol, cross)
        variance_standardized = self.kernel(x, x) - sum(item * item for item in v)
        variance_standardized = max(variance_standardized, self.min_variance)
        return Prediction(
            mean=self._y_mean + self._y_scale * mean_standardized,
            variance=(self._y_scale**2) * variance_standardized,
        )

    def predict_many(self, strings: list[str]) -> list[Prediction]:
        return [self.predict(string) for string in strings]


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    lower = [[0.0 for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for col in range(row + 1):
            accum = sum(lower[row][idx] * lower[col][idx] for idx in range(col))
            if row == col:
                value = matrix[row][row] - accum
                if value <= 0 or not math.isfinite(value):
                    raise ValueError("matrix is not positive definite")
                lower[row][col] = math.sqrt(value)
            else:
                if lower[col][col] == 0:
                    raise ValueError("matrix is singular")
                lower[row][col] = (matrix[row][col] - accum) / lower[col][col]
    return lower


def _forward_substitution(lower: list[list[float]], vector: list[float]) -> list[float]:
    result = [0.0 for _ in vector]
    for row in range(len(vector)):
        accum = sum(lower[row][col] * result[col] for col in range(row))
        result[row] = (vector[row] - accum) / lower[row][row]
    return result


def _back_substitution_from_lower(lower: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    result = [0.0 for _ in vector]
    for row in range(n - 1, -1, -1):
        accum = sum(lower[col][row] * result[col] for col in range(row + 1, n))
        result[row] = (vector[row] - accum) / lower[row][row]
    return result


def _solve_cholesky(lower: list[list[float]], vector: list[float]) -> list[float]:
    intermediate = _forward_substitution(lower, vector)
    return _back_substitution_from_lower(lower, intermediate)
