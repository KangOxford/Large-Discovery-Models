"""String kernels inspired by GAUCHE's bag-of-SMILES examples."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


class NGramStringKernel:
    """Dot-product kernel over character n-gram count dictionaries.

    GAUCHE's lightweight SMILES string-kernel examples featurize strings as
    character n-gram counts and then use a kernel over those feature vectors.
    This class keeps that idea but avoids a scikit-learn dependency.
    """

    def __init__(self, *, max_ngram: int = 5, normalize: bool = True, output_scale: float = 1.0) -> None:
        if max_ngram < 1:
            raise ValueError("max_ngram must be at least 1.")
        if output_scale <= 0:
            raise ValueError("output_scale must be positive.")
        self.max_ngram = max_ngram
        self.normalize = normalize
        self.output_scale = output_scale
        self._feature_cache: dict[str, Counter[str]] = {}
        self._norm_cache: dict[str, float] = {}

    def features(self, value: str) -> Counter[str]:
        text = str(value)
        cached = self._feature_cache.get(text)
        if cached is not None:
            return cached

        counts: Counter[str] = Counter()
        for size in range(1, self.max_ngram + 1):
            if len(text) < size:
                break
            for start in range(0, len(text) - size + 1):
                counts[text[start : start + size]] += 1
        self._feature_cache[text] = counts
        return counts

    def self_similarity(self, value: str) -> float:
        text = str(value)
        cached = self._norm_cache.get(text)
        if cached is not None:
            return cached
        feats = self.features(text)
        norm = sum(count * count for count in feats.values())
        self._norm_cache[text] = float(norm)
        return float(norm)

    def similarity(self, left: str, right: str) -> float:
        left_features = self.features(left)
        right_features = self.features(right)
        if len(left_features) > len(right_features):
            left_features, right_features = right_features, left_features
        dot = sum(count * right_features.get(token, 0) for token, count in left_features.items())
        value = float(dot)
        if self.normalize:
            denom = math.sqrt(self.self_similarity(left) * self.self_similarity(right))
            if denom <= 0:
                value = 0.0
            else:
                value /= denom
        return self.output_scale * value

    def matrix(self, left: Iterable[str], right: Iterable[str] | None = None) -> list[list[float]]:
        left_items = list(left)
        right_items = left_items if right is None else list(right)
        return [[self.similarity(left_item, right_item) for right_item in right_items] for left_item in left_items]

    def __call__(self, left: str, right: str) -> float:
        return self.similarity(left, right)
