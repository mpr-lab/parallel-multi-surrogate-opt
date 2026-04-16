"""
Run the Parallel Multi-Surrogate BO experiment across seeds and benchmarks.

This is the main experimental script for the paper.  It runs the prototype
method (RF + GP with cost-aware refresh) across one or more benchmarks and
seeds, saving CSV/JSON results for each run.

Usage examples:
    # Single run with a specific config and seed
    python experiments/run_parallel_msbo.py --config configs/branin.yaml --seed 42

    # Multiple seeds on one benchmark
    python experiments/run_parallel_msbo.py --config configs/branin.yaml \
        --seeds 42 123 456 789 1024

    # All benchmarks, multiple seeds
    python experiments/run_parallel_msbo.py --all-benchmarks --seeds 42 123 456
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


# Maps benchmark name -> config file path
BENCHMARK_CONFIGS = {
    "branin":    "configs/branin.yaml",
    "hartmann6": "configs/hartmann.yaml",
    "rastrigin": "configs/rastrigin.yaml",
    "xgboost":   "configs/xgboost.yaml",
}


def run_one(config_path: str, seed: int) -> dict:
    """
    Run a single experiment with the given config and seed.

    The seed in the config is overridden by the seed argument so we can
    run multiple seeds without editing any files.

    Returns:
        Summary dict: {benchmark, method, seed, best_y, known_optimum}
    """
    cfg = load_yaml(config_path)
    cfg["seed"] = seed   # override config seed with the requested seed

    # Set the global seed RIGHT HERE before anything else runs
    set_global_seed(seed)

    benchmark_name = cfg.get("benchmark", "branin")

    print(f"\n{'='*60}")
    print(f"Benchmark : {benchmark_name}")
    print(f"Method    : {cfg.get('method', 'parallel_msbo')}")
    print(f"Seed      : {seed}")
    print(f"Config    : {config_path}")
    print(f"{'='*60}")

    # Load the objective
    if benchmark_name == "xgboost":
        objective_fn  = xgboost_task
        bounds        = XGBOOST_BOUNDS
        known_optimum = 1.0
        cfg["pass_seed_to_objective"] = True
    else:
        objective_fn, bounds, known_optimum = get_benchmark(benchmark_name)
        cfg.setdefault("pass_seed_to_objective", False)

    # Run the optimizer
    dataset, run_metrics = run_optimization(cfg, objective_fn, bounds)

    # Save results
    output_dir = cfg.get("output_dir", "results")
    run_name   = f"{cfg.get('method', 'parallel_msbo')}_{benchmark_name}_seed{seed}"
    dataset.run_name = run_name  # ensure run_name reflects the actual seed used

    csv_path     = dataset.save_results_csv()
    metadata     = build_run_metadata(cfg, config_path=config_path)
    json_path    = dataset.save_metadata_json(metadata)
    metrics_path = os.path.join(output_dir, f"{run_name}_metrics.json")
    save_json(run_metrics.to_dict(), metrics_path)

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}")
    print(f"  Saved: {metrics_path}")

    return {
        "benchmark":    benchmark_name,
        "method":       cfg.get("method", "parallel_msbo"),
        "seed":         seed,
        "best_y":       dataset.best_y,
        "known_optimum": known_optimum,
        "gp_fit_total":  run_metrics.cumulative_gp_fit_time(),
        "rf_fit_total":  run_metrics.cumulative_rf_fit_time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Parallel Multi-Surrogate BO experiments"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a single YAML config file"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="One or more seeds to run (default: 42)"
    )
    parser.add_argument(
        "--all-benchmarks", action="store_true",
        help="Run all benchmarks (branin, hartmann6, rastrigin)"
    )
    args = parser.parse_args()

    results = []

    if args.all_benchmarks:
        # Run all synthetic benchmarks (skip xgboost in --all-benchmarks
        # because it is slow; run it explicitly with --config configs/xgboost.yaml)
        for bench in ["branin", "hartmann6", "rastrigin"]:
            for seed in args.seeds:
                r = run_one(BENCHMARK_CONFIGS[bench], seed)
                results.append(r)
    elif args.config:
        for seed in args.seeds:
            r = run_one(args.config, seed)
            results.append(r)
    else:
        parser.print_help()
        sys.exit(1)

    # Print summary table
    print("\n\n" + "="*72)
    print("EXPERIMENT SUMMARY")
    print("="*72)
    print(f"{'Benchmark':<12} {'Method':<18} {'Seed':>6} "
          f"{'Best Y':>10} {'Optimum':>10} {'GP time':>9} {'RF time':>9}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['benchmark']:<12} {r['method']:<18} {r['seed']:>6} "
            f"{r['best_y']:>10.4f} {r['known_optimum']:>10.4f} "
            f"{r['gp_fit_total']:>9.2f} {r['rf_fit_total']:>9.2f}"
        )
    print("="*72)


if __name__ == "__main__":
    main()
