"""Trial objects with the suggest_* API used by the ReaSyn objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FrozenTrial:
    number: int
    value: float
    params: dict[str, Any]
    user_attrs: dict[str, Any] = field(default_factory=dict)
    state: str = "COMPLETE"


class Trial:
    def __init__(self, number: int, params: dict[str, Any]) -> None:
        self.number = number
        self.params: dict[str, Any] = {}
        self.user_attrs: dict[str, Any] = {}
        self._suggested = dict(params)

    def suggest_int(self, name: str, low: int, high: int) -> int:
        value = self._require(name)
        value = int(value)
        if value < low or value > high:
            raise ValueError(f"Suggested value for {name}={value} is outside [{low}, {high}].")
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float) -> float:
        value = float(self._require(name))
        if value < low or value > high:
            raise ValueError(f"Suggested value for {name}={value} is outside [{low}, {high}].")
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        value = self._require(name)
        if value not in choices:
            raise ValueError(f"Suggested value for {name}={value!r} is not in {choices!r}.")
        self.params[name] = value
        return value

    def set_user_attr(self, name: str, value: Any) -> None:
        self.user_attrs[name] = value

    def freeze(self, value: float, *, state: str = "COMPLETE") -> FrozenTrial:
        params = dict(self._suggested)
        params.update(self.params)
        return FrozenTrial(
            number=self.number,
            value=float(value),
            params=params,
            user_attrs=dict(self.user_attrs),
            state=state,
        )

    def _require(self, name: str) -> Any:
        if name not in self._suggested:
            raise KeyError(f"No StrBO suggestion was prepared for parameter {name!r}.")
        return self._suggested[name]
