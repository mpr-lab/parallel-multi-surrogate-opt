from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ChoiceDimension:
    """One dimension of a ParameterSpace: a fixed, ordered set of allowed
    values.

    This single mechanism covers integers, discrete numeric choices, and
    categorical choices -- all three are "pick one of N options"
    underneath, so there is no need for three separate dimension types.
    """

    name: str
    choices: Sequence[Any]

    def __post_init__(self):
        if len(self.choices) == 0:
            raise ValueError(f"dimension {self.name!r} must have at least one choice")

    def decode(self, raw_value: float) -> Any:
        """Map a raw numeric value in [0, 1) to one of `choices`."""
        n = len(self.choices)
        index = int(raw_value * n)
        index = min(max(index, 0), n - 1)
        return self.choices[index]

    def encode(self, value: Any) -> float:
        """Inverse of decode: return a representative raw value (the
        midpoint of that choice's bucket) for a given choice."""
        index = list(self.choices).index(value)
        return (index + 0.5) / len(self.choices)


class ParameterSpace:
    """Canonical encoding/decoding between a raw numeric vector (what
    SearchSpace/surrogates see) and a decoded parameter dict (what
    objectives see).

    `bounds` is always the unit interval per dimension, since every
    ChoiceDimension's raw encoding lives in [0, 1) -- this is exactly what
    gets handed to the existing, unmodified SearchSpace.
    """

    def __init__(self, dimensions: list[ChoiceDimension]):
        if len(dimensions) == 0:
            raise ValueError("ParameterSpace needs at least one dimension")
        names = [d.name for d in dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"dimension names must be unique, got {names}")
        self.dimensions = dimensions

    @property
    def dim(self) -> int:
        return len(self.dimensions)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * self.dim

    @property
    def total_configurations(self) -> int:
        """Size of the entire finite decoded configuration space -- the
        product of each dimension's choice count. Every ChoiceDimension
        has a finite, fixed choice list, so this is always a finite int.
        """
        total = 1
        for d in self.dimensions:
            total *= len(d.choices)
        return total

    def decode(self, x: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"expected a vector of length {self.dim}, got {x.shape[0]}")
        return {d.name: d.decode(v) for d, v in zip(self.dimensions, x)}

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        missing = [d.name for d in self.dimensions if d.name not in params]
        if missing:
            raise ValueError(f"missing parameter(s) for encoding: {missing}")
        return np.array([d.encode(params[d.name]) for d in self.dimensions], dtype=float)
