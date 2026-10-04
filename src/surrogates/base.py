from abc import ABC, abstractmethod

import numpy as np

from search_space import SearchSpace


class BaseSurrogate(ABC):
    """Minimal contract every surrogate must satisfy to plug into the
    optimization loop.

    Only `name`, `fit`, and `propose` are required. `predict()` is
    deliberately NOT part of this contract: the orchestrator never calls
    it. A concrete surrogate may still implement `predict()` (or anything
    else) for its own internal use -- e.g. to compute `propose()` -- but
    nothing outside the surrogate may depend on it existing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable label used as `surrogate_source` for attribution."""
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on the full shared observation history (X, y)."""
        ...

    @abstractmethod
    def propose(
        self,
        search_space: SearchSpace,
        n_candidates: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return up to n_candidates configurations, shape
        (n_candidates, search_space.dim), ordered best-first."""
        ...
