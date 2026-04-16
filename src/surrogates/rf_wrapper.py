"""
Random Forest surrogate model — the 'fast' surrogate.

The RF is retrained every round because it is cheap (seconds or less).
Uncertainty is estimated from the variance across individual trees.

Reproducibility:
    random_state=self.seed is passed to RandomForestRegressor explicitly.
    This means: same training data + same seed => identical model.
    We do NOT rely on the global seed having been set (though it is).
    n_jobs=1 is required for determinism — parallelising tree building
    can produce slightly different results depending on thread scheduling.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.surrogates.base import BaseSurrogate
from src.search_space import SearchSpace


class RandomForestSurrogate(BaseSurrogate):
    """
    Surrogate model backed by a scikit-learn RandomForestRegressor.

    The predictive uncertainty (std) is approximated as the standard
    deviation of predictions across all trees in the forest.  This is
    a common heuristic used in BO with tree-based surrogates.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        seed: int,
        n_estimators: int = 100,
    ):
        """
        Args:
            search_space: The optimization domain.
            seed:         Passed to RandomForestRegressor(random_state=seed).
                          Controls tree structure and bootstrap sampling.
            n_estimators: Number of trees.  100 is a reasonable default.
        """
        super().__init__(search_space, seed)
        self.n_estimators = n_estimators
        self._model: RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train (or retrain) the Random Forest on the current dataset.

        A new model is created each time with the same random_state.
        This is intentional: the seed controls tree structure given the
        data, not running state, so same data => same model.

        Args:
            X: Observed configs, shape (n_obs, dim).
            y: Observed values,  shape (n_obs,).
        """
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,   # <-- explicit seed, not relying on global
            n_jobs=1,                 # single thread = deterministic on all platforms
        )
        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at candidate points.

        Mean  = average prediction across all trees.
        Std   = standard deviation of predictions across trees.
               (Larger std means the trees disagree — higher uncertainty.)

        Args:
            X: Candidate points, shape (n_points, dim).

        Returns:
            (mean, std), each shape (n_points,).
        """
        if not self._is_fitted:
            raise RuntimeError("RandomForestSurrogate.predict() called before fit().")

        # Collect prediction from every individual tree: shape (n_trees, n_points)
        tree_preds = np.array([
            tree.predict(X) for tree in self._model.estimators_
        ])

        mean = tree_preds.mean(axis=0)   # shape (n_points,)
        std  = tree_preds.std(axis=0)    # shape (n_points,)
        return mean, std
