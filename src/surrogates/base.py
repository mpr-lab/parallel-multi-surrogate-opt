"""
Abstract base class for all surrogate models.

All surrogates (RandomForest, GP, etc.) implement this same interface so
the orchestrator can treat them interchangeably.

The propose() method is provided here as a default implementation that:
  1. Samples a large pool of random candidates from the search space.
  2. Scores them with Expected Improvement using the surrogate's predict().
  3. Returns the top-n candidates.

Subclasses can override propose() if they need a different strategy.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm

from src.search_space import SearchSpace


class BaseSurrogate(ABC):
    """
    Interface that every surrogate model must implement.
    """

    def __init__(self, search_space: SearchSpace, seed: int):
        """
        Args:
            search_space: The optimization domain (bounds for each dimension).
            seed:         Random seed for this surrogate's internal random state.
                          Pass this to random_state= in sklearn models — do NOT
                          rely on the global seed having been set.
        """
        self.search_space = search_space
        self.seed         = seed
        self._is_fitted   = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the surrogate on all observed data.

        Args:
            X: Tried hyperparameter configs, shape (n_obs, dim).
            y: Observed objective values,    shape (n_obs,).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and predictive std at candidate points.

        Args:
            X: Candidate points, shape (n_points, dim).

        Returns:
            Tuple of (mean, std), each shape (n_points,).
        """
        ...

    def propose(
        self,
        n_return: int,
        rng: np.random.Generator,
        current_best_y: float,
        n_pool: int = 1000,
    ) -> np.ndarray:
        """
        Propose the most promising candidates by scoring a random pool with EI.

        If the model hasn't been trained yet, falls back to pure random sampling.

        Steps:
          1. Sample n_pool points uniformly from the search space (using rng).
          2. Score each with Expected Improvement via predict().
          3. Return the top n_return points.

        Args:
            n_return:       How many candidates to return.
            rng:            Seeded numpy Generator — passed from the orchestrator
                            so candidate sampling is reproducible.
            current_best_y: Best objective value observed so far.
            n_pool:         Size of the random pool to evaluate.

        Returns:
            Array of shape (n_return, dim).
        """
        if not self._is_fitted:
            # Not trained yet — just return random candidates
            return self.search_space.sample(n_return, rng)

        # 1. Sample a random pool
        candidates = self.search_space.sample(n_pool, rng)

        # 2. Score with Expected Improvement
        mean, std = self.predict(candidates)
        ei = _expected_improvement(mean, std, current_best_y)

        # 3. Return top-n by EI score (argsort descending)
        top_idx = np.argsort(ei)[::-1][:n_return]
        return candidates[top_idx]


# --------------------------------------------------------------------------- #
#  Expected Improvement — used by propose() above                              #
# --------------------------------------------------------------------------- #

def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected Improvement (EI) acquisition function.

    EI(x) = E[max(f(x) - (best_y + xi), 0)]

    Closed-form under a Gaussian predictive distribution N(mean, std^2):
        z  = (mean - best_y - xi) / std
        EI = (mean - best_y - xi) * Phi(z) + std * phi(z)

    where Phi = normal CDF, phi = normal PDF.

    The xi parameter adds a small exploration bonus: xi > 0 means the
    model must predict at least xi above the current best to be attractive.

    Args:
        mean:   Predicted mean,  shape (n,).
        std:    Predicted std,   shape (n,).
        best_y: Current best observed objective value.
        xi:     Exploration bonus (default 0.01).

    Returns:
        EI scores, shape (n,).  Larger = more promising.
    """
    std = np.maximum(std, 1e-9)   # avoid division by zero
    improvement = mean - best_y - xi
    z  = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    ei[std < 1e-9] = 0.0          # deterministic points have zero EI
    return ei
