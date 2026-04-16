"""
Central seed utility for reproducible experiments.

This is the single source of truth for all randomness in the project.
Every component that involves randomness should use one of these three
functions rather than seeding itself independently.

Typical usage in a run entry point:
    from src.utils.seeds import set_global_seed, make_rng, derive_worker_seed

    set_global_seed(42)        # call exactly once at the top of the run
    rng = make_rng(42)         # for explicit per-module RNG usage
    ws  = derive_worker_seed(42, round_id=3, candidate_id=7)
"""

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for Python's random module, NumPy, and
    PyTorch (if installed).  Call this exactly ONCE at the top of a run.

    Why do we need to seed all three separately?
    Each library maintains its own independent internal RNG state.
    Setting only numpy.random.seed() will NOT affect Python's random
    module or torch, and vice versa.  We must set all of them.

    Args:
        seed: Any non-negative integer.  42 and 0 are both valid choices.
              Use the same value that is in your experiment config so the
              run can always be reproduced.
    """
    # Python's built-in random module.
    # Used by e.g. random.shuffle(), random.choice(), and some sklearn internals.
    random.seed(seed)

    # NumPy's legacy global random state.
    # Many scientific libraries (including older sklearn code) call
    # np.random.* directly, so we need to seed this too.
    np.random.seed(seed)

    # PyTorch — only imported if it is already installed.
    # We use a try/except so the rest of the codebase does not need to
    # worry about whether torch is present in the environment.
    try:
        import torch
        torch.manual_seed(seed)
        # Also seed CUDA if a GPU is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic (slight performance cost, but required for
        # bit-exact reproducibility on GPU)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch is not installed — that is fine for this prototype


def make_rng(seed: int) -> np.random.Generator:
    """
    Return a fresh, independent numpy Generator seeded from `seed`.

    Unlike the legacy np.random.seed() approach (which modifies global state),
    a Generator object carries its own isolated state.  You can safely pass it
    around between functions without worrying about other code accidentally
    changing its state.

    Use this whenever a function needs explicit RNG control — for example,
    when sampling candidate points or performing train/val splits.

    Args:
        seed: Any non-negative integer.

    Returns:
        A numpy.random.Generator (default algorithm: PCG64).

    Example:
        rng = make_rng(42)
        X = rng.uniform(0, 1, size=(100, 6))  # always the same 100 points
    """
    return np.random.default_rng(seed)


def derive_worker_seed(base_seed: int, round_id: int, candidate_id: int) -> int:
    """
    Derive a unique, deterministic seed for a single worker evaluation.

    Each (base_seed, round_id, candidate_id) triple maps to exactly one
    integer.  This guarantees:
      - Two runs with the same base_seed produce identical worker seeds.
      - Two workers in the same round always get DIFFERENT seeds.
      - Changing the round number changes all seeds even if candidate_id
        is the same — worker 0 in round 3 != worker 0 in round 5.

    How it works:
        We mix the three values using large prime multipliers.
        The primes spread the values out and reduce collisions.
        Taking modulo 2^31 keeps the result in a valid seed range.

    Args:
        base_seed:    The run-level seed from the experiment config.
        round_id:     Which optimization round (0-indexed).
                      Use -1 for the initial random sampling phase.
        candidate_id: Which candidate within that round (0-indexed).

    Returns:
        A non-negative integer suitable for set_global_seed() or
        random_state= parameters in sklearn models.

    Example:
        seeds = [derive_worker_seed(42, round_id=2, candidate_id=i)
                 for i in range(8)]
        # Each seed is unique; same inputs always give the same output.
    """
    # Large primes as mixing constants.
    # These are coprime with 2^31 and with each other, which prevents
    # systematic collisions for small round/candidate values.
    P1 = 1_000_003
    P2 = 999_983

    # Mix all three values.  Adding (not XOR) avoids symmetry where
    # round_id==candidate_id would give the same contribution.
    mixed = (base_seed + round_id * P1 + candidate_id * P2) % (2 ** 31)

    return abs(mixed)  # always non-negative
