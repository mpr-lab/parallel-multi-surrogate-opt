"""
Search space definition.

A SearchSpace is an axis-aligned box in R^d defined by per-dimension
lower and upper bounds.  Keeping this in one place ensures that every
surrogate and every benchmark uses exactly the same definition.

Reproducibility note:
    sample() requires an explicit numpy Generator (from make_rng) rather
    than using global random state.  This makes sampling fully reproducible:
    the same rng seeded with the same value always produces the same points.
"""

import numpy as np


class SearchSpace:
    """
    An axis-aligned box in R^d defined by per-dimension lower and upper bounds.

    Example:
        # 2D Branin search space
        space = SearchSpace([(-5.0, 10.0), (0.0, 15.0)])
        rng = make_rng(42)
        X = space.sample(100, rng)   # shape (100, 2), always the same 100 points
    """

    def __init__(self, bounds: list[tuple[float, float]]):
        """
        Args:
            bounds: List of (low, high) pairs, one per dimension.
                    E.g. [(-5.0, 10.0), (0.0, 15.0)] for 2D Branin.
        """
        self.bounds = bounds
        self.dim    = len(bounds)
        self.lows   = np.array([b[0] for b in bounds], dtype=float)
        self.highs  = np.array([b[1] for b in bounds], dtype=float)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n points uniformly at random from the search space.

        We require an explicit rng rather than using global random state
        so the caller controls exactly which random numbers are consumed.
        This is essential for reproducibility: the same rng seeded with
        the same value will always produce the same sample.

        Args:
            n:   Number of points to sample.
            rng: A numpy Generator from make_rng(seed).  Do NOT pass
                 np.random as the rng — use make_rng() explicitly.

        Returns:
            Array of shape (n, dim).  Every row is a point in the space.
        """
        # rng.random() gives uniform [0, 1) — scale to [low, high)
        unit = rng.random((n, self.dim))
        return self.lows + unit * (self.highs - self.lows)

    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        Clip an array of points to lie within the search space bounds.

        Useful after any operation that might push a point outside the
        bounds (e.g., GP acquisition function optimization restarts).

        Args:
            X: Array of shape (n, dim) or (dim,).

        Returns:
            Clipped array, same shape as input.
        """
        return np.clip(X, self.lows, self.highs)

    def __repr__(self) -> str:
        return f"SearchSpace(dim={self.dim}, bounds={self.bounds})"
