"""
Synthetic benchmark functions used for testing optimization algorithms.

All functions here are DETERMINISTIC — given the same input x they always
return the same value.  No randomness is involved in the evaluation itself.
Randomness only appears in how we sample initial points and candidates,
which is handled by the search space sampler.

Conventions used throughout:
  - We NEGATE all functions so the optimizer can always MAXIMIZE.
    (Most BO theory maximizes; most test functions are minimized.)
  - The search space bounds are defined alongside the function in the
    BENCHMARK_REGISTRY dict at the bottom of this file.
  - "optimum" refers to the best (highest) value we can expect to find,
    i.e. the negated global minimum.

References:
    Branin:    Dixon & Szegö (1978)
    Hartmann:  Dixon & Szegö (1978)
    Rastrigin: Rastrigin (1974)
"""

import numpy as np


# --------------------------------------------------------------------------- #
#  Benchmark functions                                                          #
# --------------------------------------------------------------------------- #

def branin(x: np.ndarray) -> float:
    """
    Branin-Hoo function.  2-dimensional.
    Global minimum ≈ 0.397887 at three locations.

    Standard domain: x1 ∈ [-5, 10], x2 ∈ [0, 15].
    We negate so the optimizer maximizes (best returned value ≈ -0.3979).

    Args:
        x: 1D array of length 2: [x1, x2].

    Returns:
        Negated Branin value (float).
    """
    x1, x2 = float(x[0]), float(x[1])

    # Standard Branin parameters
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    val = (a * (x2 - b * x1**2 + c * x1 - r)**2
           + s * (1 - t) * np.cos(x1) + s)
    return -float(val)  # negate: we maximize


def hartmann6(x: np.ndarray) -> float:
    """
    Hartmann 6-dimensional function.
    Global minimum ≈ -3.32237 at approx (0.201, 0.150, 0.477, 0.275, 0.312, 0.657).

    Domain: x_i ∈ [0, 1] for all i = 1..6.
    We negate so the returned optimum ≈ +3.32237.

    Args:
        x: 1D array of length 6, all components in [0, 1].

    Returns:
        Negated Hartmann value (float).
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([
        [10,    3,    17,   3.5,  1.7,  8  ],
        [0.05,  10,   17,   0.1,  8,    14 ],
        [3,     3.5,  1.7,  10,   17,   8  ],
        [17,    8,    0.05, 10,   0.1,  14 ],
    ], dtype=float)

    P = 1e-4 * np.array([
        [1312, 1696, 5569,  124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091,  381],
    ], dtype=float)

    result = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i]) ** 2)
        result -= alpha[i] * np.exp(-inner)

    return -float(result)  # negate: we maximize


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function.  Works for any dimension d.
    Global minimum = 0 at the origin.  Highly multimodal (many local minima).

    Domain: x_i ∈ [-5.12, 5.12] for all i.
    We negate so the returned optimum = 0.

    Args:
        x: 1D array of any length (we use 6D in the experiments).

    Returns:
        Negated Rastrigin value (float).  Best possible = 0.
    """
    d   = len(x)
    A   = 10.0
    val = A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return -float(val)  # negate: we maximize


# --------------------------------------------------------------------------- #
#  Benchmark registry                                                           #
# --------------------------------------------------------------------------- #

# Maps benchmark name -> {fn, bounds, dim, optimum}
# "optimum" is the best value we can hope to find (negated global minimum).
BENCHMARK_REGISTRY = {
    "branin": {
        "fn":      branin,
        "bounds":  [(-5.0, 10.0), (0.0, 15.0)],
        "dim":     2,
        "optimum": -0.397887,   # negated minimum ≈ -0.397887
    },
    "hartmann6": {
        "fn":      hartmann6,
        "bounds":  [(0.0, 1.0)] * 6,
        "dim":     6,
        "optimum": 3.32237,     # negated minimum ≈ +3.32237
    },
    "rastrigin": {
        "fn":      rastrigin,
        "bounds":  [(-5.12, 5.12)] * 6,   # 6D version
        "dim":     6,
        "optimum": 0.0,         # negated minimum = 0 (at origin)
    },
}


def get_benchmark(name: str) -> tuple:
    """
    Return (objective_fn, bounds, known_optimum) for a named benchmark.

    Args:
        name: One of "branin", "hartmann6", "rastrigin".

    Returns:
        Tuple of:
            fn        - the objective function (accepts 1D numpy array)
            bounds    - list of (low, high) pairs for SearchSpace
            optimum   - the best value we can expect (for reference in logs)

    Raises:
        ValueError if the name is not recognized.
    """
    if name not in BENCHMARK_REGISTRY:
        available = list(BENCHMARK_REGISTRY.keys())
        raise ValueError(
            f"Unknown benchmark: {name!r}.  "
            f"Available benchmarks: {available}"
        )
    entry = BENCHMARK_REGISTRY[name]
    return entry["fn"], entry["bounds"], entry["optimum"]
