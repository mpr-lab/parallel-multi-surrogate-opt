import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from search_space import SearchSpace
from surrogates.base import BaseSurrogate


class GaussianProcessSurrogate(BaseSurrogate):
    """Gaussian-process surrogate.

    Proposal strategy: sample a pool of candidates uniformly from the
    search space, score every pool point with the fitted GP's mean
    prediction, and return the top `n_candidates` by predicted score.
    This mirrors RandomForestSurrogate's proposal rule exactly -- no
    acquisition-function optimization, no use of the GP's predictive
    uncertainty, just mean-prediction ranking over a random pool.
    """

    def __init__(
        self,
        pool_size: int = 200,
        n_restarts_optimizer: int = 5,
        random_state: int | None = None,
    ):
        self.pool_size = pool_size
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,
            random_state=random_state,
        )
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "gaussian_process"

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
