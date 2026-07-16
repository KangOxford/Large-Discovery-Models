"""Search-space utilities for mixed ReaSyn parameters."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Dimension:
    name: str
    kind: str
    low: int | float | None = None
    high: int | float | None = None
    choices: tuple[Any, ...] = ()

    @classmethod
    def integer(cls, name: str, low: int, high: int) -> "Dimension":
        return cls(name=name, kind="int", low=low, high=high)

    @classmethod
    def floating(cls, name: str, low: float, high: float) -> "Dimension":
        return cls(name=name, kind="float", low=low, high=high)

    @classmethod
    def categorical(cls, name: str, choices: Iterable[Any]) -> "Dimension":
        return cls(name=name, kind="categorical", choices=tuple(choices))

    def __post_init__(self) -> None:
        if self.kind not in {"int", "float", "categorical"}:
            raise ValueError(f"Unsupported dimension kind: {self.kind}")
        if self.kind in {"int", "float"}:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name} requires low/high bounds.")
            if self.low > self.high:
                raise ValueError(f"{self.name} bounds must be ascending.")
        if self.kind == "categorical" and not self.choices:
            raise ValueError(f"{self.name} requires at least one choice.")

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "float":
            return rng.uniform(float(self.low), float(self.high))
        return rng.choice(self.choices)

    def mutate(self, value: Any, rng: random.Random, *, scale: float = 0.25) -> Any:
        if self.kind == "categorical":
            if len(self.choices) == 1:
                return self.choices[0]
            alternatives = [choice for choice in self.choices if choice != value]
            return rng.choice(alternatives or self.choices)
        if self.kind == "int":
            low = int(self.low)
            high = int(self.high)
            span = max(high - low, 1)
            step = int(round(rng.gauss(0.0, max(1.0, span * scale))))
            if step == 0:
                step = rng.choice([-1, 1])
            return min(high, max(low, int(value) + step))

        low = float(self.low)
        high = float(self.high)
        span = max(high - low, 1e-12)
        mutated = float(value) + rng.gauss(0.0, span * scale)
        return min(high, max(low, mutated))

    def canonicalize(self, value: Any) -> Any:
        if self.kind == "int":
            return int(value)
        if self.kind == "float":
            return float(value)
        return value

    def encode_value(self, value: Any) -> str:
        value = self.canonicalize(value)
        if self.kind == "categorical":
            return _safe_token(value)
        if self.kind == "int":
            low = int(self.low)
            high = int(self.high)
            width = max(len(str(abs(low))), len(str(abs(high))), 2)
            span = max(high - low, 1)
            bucket = round((int(value) - low) / span * 20)
            return f"{int(value):0{width}d}:q{bucket:02d}"

        low = float(self.low)
        high = float(self.high)
        span = max(high - low, 1e-12)
        bucket = round((float(value) - low) / span * 20)
        return f"{float(value):.6g}:q{bucket:02d}"


@dataclass(frozen=True)
class SearchSpace:
    dimensions: tuple[Dimension, ...]

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise ValueError("SearchSpace requires at least one dimension.")
        names = [dimension.name for dimension in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("Dimension names must be unique.")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {dimension.name: dimension.sample(rng) for dimension in self.dimensions}

    def mutate(
        self,
        config: dict[str, Any],
        rng: random.Random,
        *,
        mutation_rate: float = 0.55,
        scale: float = 0.25,
    ) -> dict[str, Any]:
        mutated: dict[str, Any] = {}
        changed = False
        for dimension in self.dimensions:
            current = config[dimension.name]
            if rng.random() < mutation_rate:
                mutated_value = dimension.mutate(current, rng, scale=scale)
                changed = changed or mutated_value != current
            else:
                mutated_value = current
            mutated[dimension.name] = dimension.canonicalize(mutated_value)

        if not changed:
            dimension = rng.choice(self.dimensions)
            mutated[dimension.name] = dimension.mutate(mutated[dimension.name], rng, scale=scale)
        return self.canonicalize(mutated)

    def canonicalize(self, config: dict[str, Any]) -> dict[str, Any]:
        return {dimension.name: dimension.canonicalize(config[dimension.name]) for dimension in self.dimensions}

    def key(self, config: dict[str, Any]) -> str:
        return json.dumps(self.canonicalize(config), sort_keys=True, separators=(",", ":"))

    def stringify(self, config: dict[str, Any]) -> str:
        canonical = self.canonicalize(config)
        parts = [
            f"{dimension.name}:{dimension.kind}:{dimension.encode_value(canonical[dimension.name])}"
            for dimension in self.dimensions
        ]
        return "|".join(parts)


def _safe_token(value: Any) -> str:
    token = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return token.replace("|", "/").replace(":", "=")
