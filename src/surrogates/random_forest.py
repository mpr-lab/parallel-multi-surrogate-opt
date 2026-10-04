import numpy as np
from sklearn.ensemble import RandomForestRegressor

from search_space import SearchSpace
from surrogates.base import BaseSurrogate


class RandomForestSurrogate(BaseSurrogate):
    """Random-forest surrogate.

    Proposal strategy: sample a pool of candidates uniformly from the
    search space, score every pool point with the fitted forest's mean
    prediction, and return the top `n_candidates` by predicted score.
    This is a sample-then-rank baseline -- no acquisition-function
    optimization, no uncertainty term, just mean-prediction ranking over a
    random pool.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        pool_size: int = 200,
        random_state: int | None = None,
    ):
        self.pool_size = pool_size
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            # Fixed, not exposed: model-level threading must not interfere
            # with later worker-pool parallelism experiments.
            n_jobs=1,
        )
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "random_forest"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._is_fitted = True

    def propose(
        self,
        search_space: SearchSpace,
        n_candidates: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("propose() called before fit()")

        pool_size = max(self.pool_size, n_candidates)
        pool = search_space.sample(pool_size, rng)
        predicted = self._model.predict(pool)

        # Higher predicted y = better; stable sort keeps ties deterministic.
        best_first = np.argsort(-predicted, kind="stable")[:n_candidates]
        return pool[best_first]
