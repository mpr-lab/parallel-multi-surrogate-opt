"""
Main optimization loop: Parallel Multi-Surrogate BO with Cost-Aware GP Refresh.

This module ties everything together:
  Phase 1 -- Initial random observations (warm start before training surrogates)
  Phase 2 -- Optimization rounds:
      a. Retrain fast surrogate (RF)          -- every round, always
      b. Maybe retrain slow surrogate (GP)    -- every K rounds (scheduler)
      c. Both surrogates propose candidates   -- using EI over a random pool
      d. Evaluate candidates (parallel OK)    -- each gets a unique worker seed
      e. Add results to shared dataset        -- both surrogates read the same data
      f. Log round metrics

The run-level RNG (seeded from the config seed) is passed explicitly to any
function that needs to sample random numbers.  This ensures the sequence of
random decisions is fully deterministic given the same config and seed.
"""

import time
import numpy as np

from src.utils.seeds import make_rng
from src.utils.timing import Timer
from src.search_space import SearchSpace
from src.surrogates.rf_wrapper import RandomForestSurrogate
from src.surrogates.gp_wrapper import GaussianProcessSurrogate
from src.scheduler import GPRefreshScheduler
from src.datastore import DataStore
from src.candidate_merge import merge_candidates
from src.worker_pool import evaluate_candidates
from src.metrics import RunMetrics, RoundMetrics


def run_optimization(
    cfg: dict,
    objective_fn,
    bounds: list[tuple[float, float]],
) -> tuple[DataStore, RunMetrics]:
    """
    Run the Parallel Multi-Surrogate BO with Cost-Aware GP Refresh.

    Args:
        cfg:          Full config dict loaded from YAML.  Must include 'seed'.
        objective_fn: The function to optimize.  Signature: fn(x) -> float,
                      or fn(x, seed=int) -> float if stochastic.
        bounds:       Search space bounds, list of (low, high) per dimension.

    Returns:
        Tuple of (DataStore, RunMetrics) -- the full observation history and
        per-round metrics.
    """
    # ------------------------------------------------------------------ #
    #  Unpack config with safe defaults                                    #
    # ------------------------------------------------------------------ #
    seed            = int(cfg["seed"])
    n_initial       = int(cfg.get("n_initial",            10))
    n_rounds        = int(cfg.get("n_rounds",             20))
    n_fast          = int(cfg.get("fast_worker_budget",    6))
    n_slow          = int(cfg.get("slow_worker_budget",    2))
    gp_every        = int(cfg.get("gp_refresh_every",      5))
    n_pool          = int(cfg.get("n_random_candidates", 1000))
    n_eval_workers  = int(cfg.get("n_eval_workers",        4))
    output_dir      = str(cfg.get("output_dir",       "results"))
    benchmark_name  = str(cfg.get("benchmark",         "unknown"))
    method_name     = str(cfg.get("method",     "parallel_msbo"))
    use_parallel    = bool(cfg.get("use_parallel",      False))
    use_gp          = bool(cfg.get("use_gp",             True))
    # Should we pass seed= to the objective? (True for XGBoost, False for synthetics)
    pass_seed       = bool(cfg.get("pass_seed_to_objective", False))

    run_name = f"{method_name}_{benchmark_name}_seed{seed}"

    # ------------------------------------------------------------------ #
    #  Build core objects                                                  #
    # ------------------------------------------------------------------ #
    search_space = SearchSpace(bounds)

    # The run-level RNG is used for candidate pool sampling and initial points.
    # Seeding it from the base seed ensures it always produces the same sequence.
    rng = make_rng(seed)

    # Surrogates: each receives the base seed explicitly.
    # random_state=seed in sklearn means: same data => same model.
    rf = RandomForestSurrogate(search_space, seed=seed)
    gp = GaussianProcessSurrogate(search_space, seed=seed) if use_gp else None

    # The scheduler decides when to retrain the GP
    scheduler = GPRefreshScheduler(refresh_every=gp_every)

    # DataStore for logging observations
    dataset = DataStore(dim=len(bounds), output_dir=output_dir, run_name=run_name)

    # Metrics tracker
    run_metrics = RunMetrics(
        seed=seed, method=method_name, benchmark=benchmark_name
    )

    run_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    #  Phase 1: Initial random warm start                                  #
    # ------------------------------------------------------------------ #
    print(f"\n[{run_name}] Warm start: collecting {n_initial} random observations...")

    # sample() uses the run-level rng -- always the same n_initial points
    X_init = search_space.sample(n_initial, rng)

    # round_id=-1 marks the initial phase (before any surrogate training)
    X_eval, y_eval, _, _ = evaluate_candidates(
        candidates=X_init,
        sources=["random"] * n_initial,
        objective_fn=objective_fn,
        base_seed=seed,
        round_id=-1,
        n_workers=n_eval_workers,
        pass_seed_to_objective=pass_seed,
        use_parallel=use_parallel,
    )
    dataset.add_observations(X_eval, y_eval, round_id=-1, source="random")
    print(f"  Initial best: {dataset.best_y:.4f}")

    # ------------------------------------------------------------------ #
    #  Phase 2: Optimization rounds                                        #
    # ------------------------------------------------------------------ #
    for round_id in range(n_rounds):

        # --- 2a. Retrain fast surrogate (RF) -- ALWAYS ---
        with Timer() as rf_timer:
            rf.fit(dataset.X, dataset.y)

        # --- 2b. Maybe retrain slow surrogate (GP) ---
        gp_fit_time      = 0.0
        gp_was_refreshed = False
        if use_gp and gp is not None and scheduler.should_refresh_gp(round_id):
            with Timer() as gp_timer:
                gp.fit(dataset.X, dataset.y)
            gp_fit_time      = gp_timer.elapsed
            gp_was_refreshed = True

        # --- 2c. Both surrogates propose candidates ---
        # rng is passed explicitly so candidate sampling is reproducible.
        rf_candidates = rf.propose(
            n_return=n_fast,
            rng=rng,
            current_best_y=dataset.best_y,
            n_pool=n_pool,
        )

        gp_candidates = None
        if use_gp and gp is not None and gp._is_fitted:
            gp_candidates = gp.propose(
                n_return=n_slow,
                rng=rng,
                current_best_y=dataset.best_y,
                n_pool=n_pool,
            )

        # --- 2d. Merge candidate lists ---
        candidates, sources = merge_candidates(rf_candidates, gp_candidates)

        # --- 2e. Evaluate candidates ---
        # Each candidate gets a unique seed: derive_worker_seed(seed, round_id, i)
        X_new, y_new, eval_times, ordered_sources = evaluate_candidates(
            candidates=candidates,
            sources=sources,
            objective_fn=objective_fn,
            base_seed=seed,
            round_id=round_id,
            n_workers=n_eval_workers,
            pass_seed_to_objective=pass_seed,
            use_parallel=use_parallel,
        )

        # --- 2f. Add to shared dataset ---
        for i in range(len(y_new)):
            dataset.add_observations(
                X_new[i:i+1], y_new[i:i+1],
                round_id=round_id,
                source=ordered_sources[i],
            )

        # --- 2g. Record metrics ---
        elapsed = time.perf_counter() - run_start
        round_m = RoundMetrics(
            round_id             = round_id,
            best_y               = dataset.best_y,
            n_obs                = dataset.n_obs,
            rf_fit_time          = rf_timer.elapsed,
            gp_fit_time          = gp_fit_time,
            gp_was_refreshed     = gp_was_refreshed,
            eval_times           = list(eval_times),
            n_rf_candidates      = len(rf_candidates),
            n_gp_candidates      = len(gp_candidates) if gp_candidates is not None else 0,
            cumulative_wall_time = elapsed,
        )
        run_metrics.add_round(round_m)

        # --- 2h. Print progress ---
        gp_str = (f" | GP fit: {gp_fit_time:.2f}s"
                  if gp_was_refreshed else " | GP: stale")
        print(
            f"  Round {round_id:3d}: best={dataset.best_y:.4f}"
            f" | RF fit: {rf_timer.elapsed:.3f}s{gp_str}"
            f" | n_obs={dataset.n_obs}"
        )

    total_time = time.perf_counter() - run_start
    print(f"\n[{run_name}] Done. Total time: {total_time:.1f}s | "
          f"Final best: {dataset.best_y:.4f}")

    return dataset, run_metrics
