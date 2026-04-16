"""
Parallel evaluation of candidate hyperparameter configurations.

Each candidate is evaluated by the objective function.  Each worker gets a
unique, deterministic seed so the evaluation is reproducible regardless of
the order in which OS schedules the worker processes.

Key reproducibility design:
    For candidate i in round r with base_seed s:
        worker_seed = derive_worker_seed(s, r, i)

    Inside the worker:
        1. set_global_seed(worker_seed)   -- seeds all global RNG state
        2. objective_fn(x, seed=worker_seed)  -- pass seed explicitly if accepted

    This means: same base_seed + same data => same candidate => same evaluation.
    Even if two workers run in different order, each INDIVIDUAL evaluation
    always produces the same result.

    ProcessPoolExecutor is used for true parallelism.  Sequential mode
    (use_parallel=False) is provided for debugging and synthetic benchmarks
    where the evaluation is microseconds and pool overhead would dominate.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

from src.utils.seeds import derive_worker_seed, set_global_seed
from src.utils.timing import Timer


# --------------------------------------------------------------------------- #
#  Worker function (runs in a child process)                                   #
# --------------------------------------------------------------------------- #

def _worker_fn(
    x: np.ndarray,
    objective_fn: Callable,
    worker_seed: int,
    pass_seed_to_objective: bool,
) -> tuple[np.ndarray, float, float]:
    """
    Evaluate one candidate in a worker process.

    This function is the only thing that runs inside each subprocess.  It:
      1. Sets the global random seed (ensures any randomness is reproducible).
      2. Calls the objective function (with or without seed= argument).
      3. Returns (config, value, elapsed_seconds).

    Why set_global_seed inside the worker?
    Child processes in ProcessPoolExecutor inherit a copy of the parent's
    memory at fork time, but the random states are not guaranteed to be
    independent.  By explicitly seeding each worker, we ensure that even if
    workers run in different orders, each evaluation is deterministic.

    Args:
        x:                      Candidate hyperparameter configuration (1D array).
        objective_fn:           The function to call.  Must be a module-level
                                function (not a lambda) to be picklable.
        worker_seed:            Unique seed derived from base_seed + round + candidate.
        pass_seed_to_objective: If True, call objective_fn(x, seed=worker_seed).
                                If False, call objective_fn(x).

    Returns:
        (x, y, elapsed_seconds)
    """
    # Step 1: seed all global random state in this subprocess
    set_global_seed(worker_seed)

    # Step 2: evaluate the objective
    with Timer() as t:
        if pass_seed_to_objective:
            y = objective_fn(x, seed=worker_seed)
        else:
            y = objective_fn(x)

    return x, float(y), t.elapsed


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def evaluate_candidates(
    candidates: np.ndarray,
    sources: list[str],
    objective_fn: Callable,
    base_seed: int,
    round_id: int,
    n_workers: int = 4,
    pass_seed_to_objective: bool = False,
    use_parallel: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    """
    Evaluate a batch of candidates, optionally in parallel.

    Worker seed for candidate i:
        worker_seed_i = derive_worker_seed(base_seed, round_id, i)

    This deterministic per-candidate seed means:
      - Two runs with the same base_seed produce identical evaluations.
      - Changing round_id or candidate position changes the seed, so
        worker 0 in round 3 is independent of worker 0 in round 4.

    Args:
        candidates:             Array of shape (n, dim) to evaluate.
        sources:                Source label list ("rf" or "gp") per candidate.
        objective_fn:           Must be a module-level function (picklable).
        base_seed:              Run-level seed from config.
        round_id:               Current round (used in seed derivation).
        n_workers:              Number of parallel processes.
        pass_seed_to_objective: Pass seed=worker_seed to objective_fn?
                                True for stochastic objectives (XGBoost).
                                False for deterministic ones (Branin, Hartmann).
        use_parallel:           If False, evaluate sequentially.  Useful for
                                cheap synthetic benchmarks and debugging.

    Returns:
        Tuple of:
            X_new:       Evaluated configs, shape (n, dim).  Same order as input.
            y_new:       Objective values,   shape (n,).
            eval_times:  Elapsed seconds per evaluation, list of length n.
            sources:     Same source labels, same order as input.
    """
    n = len(candidates)

    # Derive a unique deterministic seed per candidate
    worker_seeds = [
        derive_worker_seed(base_seed, round_id, i) for i in range(n)
    ]

    X_new      = np.zeros_like(candidates)
    y_new      = np.zeros(n)
    eval_times = [0.0] * n

    if (not use_parallel) or n_workers <= 1:
        # ------------------------------------------------------------------ #
        #  Sequential mode — simple, easy to debug                            #
        # ------------------------------------------------------------------ #
        for i, (x, wseed) in enumerate(zip(candidates, worker_seeds)):
            _, y, t = _worker_fn(x, objective_fn, wseed, pass_seed_to_objective)
            X_new[i]      = x
            y_new[i]      = y
            eval_times[i] = t

    else:
        # ------------------------------------------------------------------ #
        #  Parallel mode — use ProcessPoolExecutor                            #
        # ------------------------------------------------------------------ #
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, (x, wseed) in enumerate(zip(candidates, worker_seeds)):
                future = executor.submit(
                    _worker_fn, x, objective_fn, wseed, pass_seed_to_objective
                )
                futures[future] = i

            # Collect results as they complete (order may differ from submission)
            for future in as_completed(futures):
                i = futures[future]
                x_i, y_i, t_i = future.result()
                X_new[i]      = x_i
                y_new[i]      = y_i
                eval_times[i] = t_i

    return X_new, y_new, eval_times, list(sources)
