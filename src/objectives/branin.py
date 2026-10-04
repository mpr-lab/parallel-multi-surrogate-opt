import numpy as np

# Standard Branin-Hoo constants (see Dixon & Szego, 1978).
_A = 1.0
_B = 5.1 / (4 * np.pi**2)
_C = 5.0 / np.pi
_R = 6.0
_S = 10.0
_T = 1.0 / (8 * np.pi)

DIM = 2


def _branin_minimize(x: np.ndarray) -> float:
    """Standard Branin-Hoo surface, in its native MINIMIZE form.

    Global minima ~0.397887 at (-pi, 12.275), (pi, 2.275), (9.42478, 2.475).
    """
    x1, x2 = x
    term1 = _A * (x2 - _B * x1**2 + _C * x1 - _R) ** 2
    term2 = _S * (1 - _T) * np.cos(x1)
    return float(term1 + term2 + _S)


def branin(x: np.ndarray, seed: int | None = None) -> float:
    """Branin-Hoo benchmark, MAXIMIZE convention.

    The benchmark is natively a minimization problem (best value is its
    lowest point). It is negated here, at the objective boundary, so that
    higher y = better everywhere downstream. y = -f_minimize(x).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape != (DIM,):
        raise ValueError(f"branin expects x of shape ({DIM},), got {x.shape}")
    return -_branin_minimize(x)
