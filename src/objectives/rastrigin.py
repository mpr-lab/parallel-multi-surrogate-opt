import numpy as np


def _rastrigin_minimize(x: np.ndarray) -> float:
    """Standard Rastrigin surface, in its native MINIMIZE form.

    Global minimum = 0 at x = 0, for any dimensionality d >= 1.
    """
    d = x.shape[0]
    return float(10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def rastrigin(x: np.ndarray, seed: int | None = None) -> float:
    """Rastrigin benchmark, MAXIMIZE convention.

    Unlike Branin and Hartmann6, Rastrigin is defined for any dimensionality
    d >= 1 — there is no fixed DIM constant to check against, only that x is
    non-empty.

    The benchmark is natively a minimization problem (its global optimum is
    its lowest value, 0, at the origin). It is negated here, at the
    objective boundary, so that higher y = better everywhere downstream.
    y = -f_minimize(x).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] == 0:
        raise ValueError("rastrigin expects a non-empty x")
    return -_rastrigin_minimize(x)
