import numpy as np

from search_space import SearchSpace
from surrogates.base import BaseSurrogate

_LOG_EPS = 1e-12
_MIN_BANDWIDTH = 1e-6


class TPESurrogate(BaseSurrogate):
    """Tree-structured Parzen Estimator surrogate.

    Proposal strategy (genuinely different from the sample-then-rank
    RF/GP baseline): split fit history into a "good" group (top `gamma`
    fraction by y) and a "bad" group (the rest), fit an independent 1-D
    Gaussian KDE per dimension for each group, then sample a candidate
    pool BY DRAWING FROM the good-group density (not a uniform pool) and
    rank those candidates by the good/bad log-density ratio. Only the
    top-N ranked candidates are returned, in the same
    `(n_candidates, dim)` best-first ndarray format RF/GP already use.
    """

    def __init__(self, gamma: float = 0.25, pool_size: int = 200):
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.gamma = gamma
        self.pool_size = pool_size
        self._good_X: np.ndarray | None = None
        self._bad_X: np.ndarray | None = None
        self._good_bandwidth: np.ndarray | None = None
        self._bad_bandwidth: np.ndarray | None = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "tpe"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < 2:
            raise ValueError(
                f"TPESurrogate.fit() needs at least 2 observations to form "
                f"both a 'good' and a 'bad' group, got {n}"
            )

        # Higher y = better; descending sort puts the best observations first.
        order = np.argsort(-y, kind="stable")
        n_good = int(round(self.gamma * n))
        n_good = min(max(n_good, 1), n - 1)  # keep both groups non-empty

        good_idx = order[:n_good]
        bad_idx = order[n_good:]
        self._good_X = X[good_idx]
        self._bad_X = X[bad_idx]
        self._good_bandwidth = self._silverman_bandwidth(self._good_X, fallback=X)
        self._bad_bandwidth = self._silverman_bandwidth(self._bad_X, fallback=X)
        self._is_fitted = True

    @staticmethod
    def _silverman_bandwidth(group: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        m = group.shape[0]
        std = group.std(axis=0, ddof=0)
        bandwidth = 1.06 * std * max(m, 1) ** (-1.0 / 5.0)

        # A group with zero spread (e.g. m == 1, or degenerate duplicate
        # values) would otherwise collapse the KDE to a delta function;
        # fall back to a fraction of the full history's spread instead.
        fallback_std = fallback.std(axis=0, ddof=0)
        bandwidth = np.where(bandwidth > _MIN_BANDWIDTH, bandwidth, fallback_std * 0.5)
        bandwidth = np.maximum(bandwidth, _MIN_BANDWIDTH)
        return bandwidth

    def propose(
        self,
        search_space: SearchSpace,
        n_candidates: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("propose() called before fit()")

        pool_size = max(self.pool_size, n_candidates)
        dim = search_space.dim
        low = search_space.low
        high = search_space.high

        n_good = self._good_X.shape[0]
        candidates = np.empty((pool_size, dim), dtype=float)
        for d in range(dim):
            center_idx = rng.integers(0, n_good, size=pool_size)
            centers = self._good_X[center_idx, d]
            noise = rng.normal(0.0, self._good_bandwidth[d], size=pool_size)
            candidates[:, d] = np.clip(centers + noise, low[d], high[d])

        log_good = self._log_density(candidates, self._good_X, self._good_bandwidth)
        log_bad = self._log_density(candidates, self._bad_X, self._bad_bandwidth)
        score = log_good - log_bad

        best_first = np.argsort(-score, kind="stable")[:n_candidates]
        return candidates[best_first]

    @staticmethod
    def _log_density(query: np.ndarray, group: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        """Vectorized log-density of `query` under a product of 1-D
        Gaussian KDEs (one per dimension, independent bandwidths)."""
        m = group.shape[0]
        total_log_density = np.zeros(query.shape[0], dtype=float)
        for d in range(group.shape[1]):
            h = bandwidth[d]
            diffs = (query[:, d][:, None] - group[:, d][None, :]) / h
            density = np.exp(-0.5 * diffs**2).sum(axis=1) / (m * h * np.sqrt(2 * np.pi))
            total_log_density += np.log(density + _LOG_EPS)
        return total_log_density
