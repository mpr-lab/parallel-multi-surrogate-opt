import numpy as np


class SearchSpace:
    """A continuous, box-bounded hyperparameter space."""

    def __init__(self, bounds: list[tuple[float, float]]):
        if len(bounds) == 0:
            raise ValueError("bounds must contain at least one dimension")

        for i, bound in enumerate(bounds):
            if len(bound) != 2:
                raise ValueError(
                    f"bounds[{i}] must be a (low, high) pair, got {bound!r}"
                )
            low, high = bound
            if not low < high:
                raise ValueError(
                    f"bounds[{i}] = ({low}, {high}) must satisfy low < high"
                )

        self._low = np.array([b[0] for b in bounds], dtype=float)
        self._high = np.array([b[1] for b in bounds], dtype=float)

    @property
    def dim(self) -> int:
        return len(self._low)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return an array of shape (n, dim) sampled uniformly within bounds."""
        return rng.uniform(self._low, self._high, size=(n, self.dim))
