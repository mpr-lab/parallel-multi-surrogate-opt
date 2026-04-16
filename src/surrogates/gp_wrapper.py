"""
Gaussian Process surrogate model — the 'slow' surrogate.

The GP is only retrained every K rounds (cost-aware refresh policy).
It provides proper predictive uncertainty but is expensive to train: O(N^3)
in the number of observations.

Reproducibility:
    random_state=self.seed is passed to GaussianProcessRegressor.
    This controls the random starting points for kernel hyperparameter
    optimization (the n_restarts_optimizer restarts).
    Input features are standardized before training to help the GP kernel.
    The StandardScaler is re-fitted each time fit() is called, using only
    the training data at that point.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from src.surrogates.base import BaseSurrogate
from src.search_space import SearchSpace


class GaussianProcessSurrogate(BaseSurrogate):
    """
    Surrogate model backed by a scikit-learn GaussianProcessRegressor.

    Uses a Matern 5/2 kernel (standard for BO) with a noise term.
    Inputs are standardized to help the kernel length-scale.

    This is the expensive surrogate in the prototype.  Its O(N^3) fitting
    cost is why we only retrain it every K rounds (the 'cost-aware GP
    refresh' in the prototype title).  The scheduler in src/scheduler.py
    decides when to call fit().
    """

    def __init__(
        self,
        search_space: SearchSpace,
        seed: int,
        n_restarts_optimizer: int = 5,
    ):
        """
        Args:
            search_space:         The optimization domain.
            seed:                 Passed to GaussianProcessRegressor(random_state=seed).
                                  Controls the random restarts of the kernel optimizer.
            n_restarts_optimizer: How many random restarts to use when optimizing
                                  kernel hyperparameters.  More = better but slower.
        """
        super().__init__(search_space, seed)
        self.n_restarts_optimizer = n_restarts_optimizer
        self._model:  GaussianProcessRegressor | None = None
        self._scaler: StandardScaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the GP on the current dataset.

        Inputs are standardized to zero mean and unit variance per dimension.
        This is important: the Matern kernel's length-scale parameters work
        much better on standardized inputs than on raw feature values.

        Args:
            X: Observed configs, shape (n_obs, dim).
            y: Observed values,  shape (n_obs,).
        """
        # Standardize inputs: fit the scaler on the current training data.
        X_scaled = self._scaler.fit_transform(X)

        # Matern(nu=2.5) kernel is the standard choice for BO.
        # ConstantKernel scales the output variance.
        # WhiteKernel adds a small noise term for numerical stability.
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(nu=2.5, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))
        )

        self._model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.seed,   # <-- controls random restarts
            normalize_y=True,         # subtracts mean of y before fitting
        )
        self._model.fit(X_scaled, y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at candidate points.

        The GP provides a proper posterior predictive distribution
        N(mean(x), std(x)^2), which is used directly by the EI acquisition.

        Args:
            X: Candidate points, shape (n_points, dim).

        Returns:
            (mean, std), each shape (n_points,).
        """
        if not self._is_fitted:
            raise RuntimeError("GaussianProcessSurrogate.predict() called before fit().")

        # Apply the same standardization we used during training
        X_scaled = self._scaler.transform(X)
        mean, std = self._model.predict(X_scaled, return_std=True)
        return mean, std
