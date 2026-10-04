import numpy as np

# Standard Hartmann-6 constants (see Dixon & Szego, 1978).
_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)

_P = 1e-4 * np.array(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ]
)

DIM = 6


def _hartmann6_minimize(x: np.ndarray) -> float:
    """Standard Hartmann-6 surface, in its native MINIMIZE form.

    Global minimum ~ -3.32237 at
    x* ~ (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    over the unit hypercube [0, 1]^6.
    """
    inner = np.sum(_A * (x - _P) ** 2, axis=1)
    return float(-np.sum(_ALPHA * np.exp(-inner)))


def hartmann6(x: np.ndarray, seed: int | None = None) -> float:
    """Hartmann-6 benchmark, MAXIMIZE convention.

    The benchmark is natively a minimization problem (its global optimum is
    the most negative value of f_minimize). It is negated here, at the
    objective boundary, so that higher y = better everywhere downstream.
    y = -f_minimize(x).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape != (DIM,):
        raise ValueError(f"hartmann6 expects x of shape ({DIM},), got {x.shape}")
    return -_hartmann6_minimize(x)
