"""
Acquisition functions for Bayesian Optimization.

These score a candidate point based on the surrogate model's prediction.
Higher score = more promising to evaluate next.

The default acquisition used throughout the prototype is Expected Improvement
(EI), which is already implemented in src/surrogates/base.py and used by the
propose() method.  This file provides standalone versions for direct use and
extends the collection with UCB for comparison.

Note: there is no randomness in these functions — they are pure mathematical
transformations of (mean, std) -> score.
"""

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected Improvement (EI) acquisition function.

    EI(x) = E[max(f(x) - (best_y + xi), 0)]
           = (mean - best_y - xi) * Phi(z) + std * phi(z)
    where z = (mean - best_y - xi) / std

    The xi term is a small exploration bonus:
      - xi=0.0: pure exploitation (greedy, prone to getting stuck)
      - xi=0.01: slight encouragement to explore (default, works well in practice)

    Args:
        mean:   Surrogate predicted mean,  shape (n,).
        std:    Surrogate predicted std,   shape (n,).
        best_y: Current best observed objective value.
        xi:     Exploration bonus.  Default 0.01.

    Returns:
        EI scores, shape (n,).  Higher = more promising to evaluate.
    """
    std = np.maximum(std, 1e-9)
    improvement = mean - best_y - xi
    z  = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    ei[std < 1e-9] = 0.0
    return ei


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    kappa: float = 2.0,
) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) acquisition function.

    UCB(x) = mean(x) + kappa * std(x)

    Higher kappa -> more exploration (prefer uncertain regions).
    Lower  kappa -> more exploitation (prefer predicted-good regions).
    kappa=2 corresponds to roughly a 95% confidence bound.

    Args:
        mean:  Surrogate predicted mean,  shape (n,).
        std:   Surrogate predicted std,   shape (n,).
        kappa: Exploration-exploitation trade-off.  Default 2.0.

    Returns:
        UCB scores, shape (n,).  Higher = more promising.
    """
    return mean + kappa * std
