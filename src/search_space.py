from __future__ import annotations

from typing import Sequence

import numpy as np


class SearchSpace:


    def __init__(self, bounds: Sequence[Sequence[float]]) -> None:
        if not bounds:
            raise ValueError("bounds must not be empty")

        validated_bounds: list[tuple[float, float]] = []

        for i, bound in enumerate(bounds):
            if len(bound) != 2:
                raise ValueError(
                    f"Each bound must have exactly 2 values, but bounds[{i}] = {bound}"
                )

            low, high = bound
            low = float(low)
            high = float(high)

            if low >= high:
                raise ValueError(
                    f"Each bound must satisfy low < high, but bounds[{i}] = ({low}, {high})"
                )

            validated_bounds.append((low, high))

        self.bounds = validated_bounds
        self.dim = len(validated_bounds)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be an instance of numpy.random.Generator"
            )

        # Separate lows and highs into arrays of shape (dim,)
        lows = np.array([low for low, _ in self.bounds], dtype=float)
        highs = np.array([high for _, high in self.bounds], dtype=float)

        # NumPy broadcasts lows/highs across n samples, producing shape (n, dim)
        samples = rng.uniform(low=lows, high=highs, size=(n, self.dim))

        return samples

    def __repr__(self) -> str:
        return f"SearchSpace(dim={self.dim}, bounds={self.bounds})"