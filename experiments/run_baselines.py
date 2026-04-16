"""
Run baseline comparisons for the prototype.

The prototype (parallel RF + GP with cost-aware refresh) is compared against
three baselines that isolate different design choices:

  1. rf_only         -- RF surrogate only, updated every round.
                       Tests: does adding the GP help at all?

  2. gp_every_round  -- RF + GP with GP refreshed EVERY round.
                       Tests: what is the optimization quality ceiling?
                       (This is the expensive 'always retrain' baseline.)

  3. gp_every_k      -- RF + GP with GP refreshed every K rounds.
                       This IS the prototype.  Included here for convenience.

Together these let us answer:
  - Does the periodic GP refresh lose much optimization quality?  (compare 2 vs 3)
  - How much time does it save?  (compare gp_fit_time in 2 vs 3)
  - Is the GP even helping?  (compare 1 vs 3)

Usage:
    python experiments/run_baselines.py --config configs/branin.yaml
    python experiments/run_baselines.py --config configs/branin.yaml \
        --seeds 42 123 456 --baselines rf_only gp_every_round gp_every_k
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seeds import set_global_seed
from src.utils.serialization import load_yaml, build_run_metadata, save_json
from src.objectives.synthetic import get_benchmark
from src.objectives.ml_tasks import xgboost_task, XGBOOST_BOUNDS
from src.orchestrator import run_optimization


# Config overrides that define each baseline variant.
# These are merged on top of the base config loaded from the YAML file.
BASELINE_OVERRIDES = {
    "rf_only": {
        "use_gp":             False,
        "fast_worker_budget": 8,   # all 8 candidates from RF
        "slow_worker_budget": 0,
        "gp_refresh_every":   1,   # irrelevant, GP disabled
        "method":             "rf_only",
    },
    "gp_every_round": {
        "use_gp":             True,
        "fast_worker_budget": 6,
        "slow_worker_budget": 2,
        "gp_refresh_every":   1,   # GP retrained EVERY round
        "method":             "gp_every_round",
    },
    "gp_every_k": {
        "use_gp":             True,
        "fast_worker_budget": 6,
        "slow_worker_budget": 2,
        # gp_refresh_every is taken from the base config (default 5)
        "method":             "gp_every_k",
    },
}


def run_baseline(
    cfg_base: dict,
    config_path: str,
    baseline_name: str,
    seed: int,
) -> dict:
    """
    Run one baseline variant with the given seed.

    Args:
        cfg_base:      Base config loaded from YAML.
        config_path:   Path to the config file (for metadata).
        baseline_name: One of the keys in BASELINE_OVERRIDES.
        seed:          Random seed for this run.

    Returns:
        Summary dict with key metrics.
    """
    # Build the final config: base overridden by baseline-specific settings
    cfg = {**cfg_base, **BASELINE_OVERRIDES[baseline_name]}
    cfg["seed"] = seed

    # Set the global seed for this run
    set_global_seed(seed)

    benchmark_name = cfg.get("benchmark", "branin")
    method_name    = cfg["method"]

    print(f"  [{method_name}] seed={seed} | benchmark={benchmark_name}")

    # Load objective
    if benchmark_name == "xgboost":
        objective_fn  = xgboost_task
        bounds        = XGBOOST_BOUNDS
        known_optimum = 1.0
        cfg["pass_seed_to_objective"] = True
    else:
        objective_fn, bounds, known_optimum = get_benchmark(benchmark_name)
        cfg.setdefault("pass_seed_to_objective", False)

    # Run optimization
    dataset, run_metrics = run_optimization(cfg, objective_fn, bounds)

    # Save results
    output_dir = cfg.get("output_dir", "results")
    run_name   = f"{method_name}_{benchmark_name}_seed{seed}"
    dataset.run_name = run_name

    csv_path     = dataset.save_results_csv()
    metadata     = build_run_metadata(cfg, config_path=config_path)
    json_path    = dataset.save_metadata_json(metadata)
    metrics_path = os.path.join(output_dir, f"{run_name}_metrics.json")
    save_json(run_metrics.to_dict(), metrics_path)

    return {
        "baseline":       baseline_name,
        "benchmark":      benchmark_name,
        "seed":           seed,
        "best_y":         dataset.best_y,
        "known_optimum":  known_optimum,
        "gp_fit_total":   run_metrics.cumulative_gp_fit_time(),
        "rf_fit_total":   run_metrics.cumulative_rf_fit_time(),
        "n_gp_refreshes": run_metrics.n_gp_refreshes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline comparisons for the prototype"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (benchmark, n_rounds, etc.)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Seeds to run for each baseline (default: 42)"
    )
    parser.add_argument(
        "--baselines", type=str, nargs="+",
        default=list(BASELINE_OVERRIDES.keys()),
        choices=list(BASELINE_OVERRIDES.keys()),
        help="Which baselines to run (default: all)"
    )
    args = parser.parse_args()

    cfg_base = load_yaml(args.config)
    results  = []

    print(f"\nRunning {len(args.baselines)} baselines x {len(args.seeds)} seeds "
          f"on benchmark '{cfg_base.get('benchmark', '?')}'")
    print("="*72)

    for baseline in args.baselines:
        for seed in args.seeds:
            r = run_baseline(cfg_base, args.config, baseline, seed)
            results.append(r)

    # Print summary
    print("\n\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Baseline':<20} {'Seed':>6} {'Best Y':>10} "
          f"{'GP time':>9} {'RF time':>9} {'GP refreshes':>13}")
    print("-"*80)
    for r in results:
        print(
            f"{r['baseline']:<20} {r['seed']:>6} {r['best_y']:>10.4f} "
            f"{r['gp_fit_total']:>9.3f} {r['rf_fit_total']:>9.3f} "
            f"{r['n_gp_refreshes']:>13}"
        )
    print("="*80)
    print("\nKey comparison:")
    print("  gp_every_round vs gp_every_k: does periodic refresh hurt quality?")
    print("  rf_only        vs gp_every_k: does adding the GP help?")
    print("  GP time:  lower in gp_every_k = the cost-aware refresh is working.")


if __name__ == "__main__":
    main()
