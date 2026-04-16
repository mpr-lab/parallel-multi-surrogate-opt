"""
Entry point for a single optimization run.

Usage:
    python src/main.py --config configs/branin.yaml
    python src/main.py --config configs/branin.yaml --seed 123
    python src/main.py --config configs/hartmann.yaml --seed 42
    python src/main.py --config configs/xgboost.yaml --seed 7

The --seed flag overrides whatever seed is in the config file.  This makes
it easy to run multiple seeds from a shell loop without editing any files:

    for seed in 42 123 456; do
        python src/main.py --config configs/branin.yaml --seed $seed
    done
"""

import argparse
import os
import sys

# Make the repo root importable when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seeds import set_global_seed
from src.utils.serialization import load_yaml, build_run_metadata, save_json
from src.objectives.synthetic import get_benchmark, BENCHMARK_REGISTRY
from src.objectives.ml_tasks import xgboost_task, XGBOOST_BOUNDS
from src.orchestrator import run_optimization


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Multi-Surrogate BO with Cost-Aware GP Refresh"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a YAML config file, e.g. configs/branin.yaml"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides the seed value in the config file)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Load config                                                         #
    # ------------------------------------------------------------------ #
    cfg = load_yaml(args.config)

    # CLI --seed takes priority over whatever is in the YAML
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Fall back to a default seed if neither CLI nor config specifies one
    if "seed" not in cfg:
        cfg["seed"] = 42
        print("Warning: no 'seed' in config; using default seed=42")

    seed = int(cfg["seed"])

    # ------------------------------------------------------------------ #
    #  Set global seed -- ONCE, right here                                  #
    # ------------------------------------------------------------------ #
    # This single call seeds Python random, NumPy, and PyTorch (if installed).
    # It must be called before any library that uses global random state.
    set_global_seed(seed)
    print(f"Global seed: {seed}")

    # ------------------------------------------------------------------ #
    #  Load benchmark                                                      #
    # ------------------------------------------------------------------ #
    benchmark_name = cfg.get("benchmark", "branin")

    if benchmark_name == "xgboost":
        objective_fn = xgboost_task
        bounds       = XGBOOST_BOUNDS
        cfg["pass_seed_to_objective"] = True   # XGBoost needs explicit seed
        known_optimum = 1.0                    # max possible AUC
    else:
        objective_fn, bounds, known_optimum = get_benchmark(benchmark_name)
        cfg.setdefault("pass_seed_to_objective", False)

    print(f"Benchmark: {benchmark_name} | known optimum ~= {known_optimum}")

    # ------------------------------------------------------------------ #
    #  Run optimization                                                    #
    # ------------------------------------------------------------------ #
    dataset, run_metrics = run_optimization(cfg, objective_fn, bounds)

    # ------------------------------------------------------------------ #
    #  Save outputs                                                        #
    # ------------------------------------------------------------------ #
    output_dir = cfg.get("output_dir", "results")
    run_name   = f"{cfg.get('method', 'parallel_msbo')}_{benchmark_name}_seed{seed}"

    # CSV: all (x, y) observations with seed embedded in run_name column
    csv_path = dataset.save_results_csv()
    print(f"\nResults CSV  : {csv_path}")

    # JSON: run-level metadata (everything needed to reproduce this run)
    metadata  = build_run_metadata(cfg, config_path=args.config)
    json_path = dataset.save_metadata_json(metadata)
    print(f"Metadata JSON: {json_path}")

    # JSON: per-round metrics (convergence curve, timing breakdown)
    metrics_path = os.path.join(output_dir, f"{run_name}_metrics.json")
    save_json(run_metrics.to_dict(), metrics_path)
    print(f"Metrics JSON : {metrics_path}")

    print(f"\nFinal best value : {dataset.best_y:.6f}")
    print(f"Known optimum    : {known_optimum:.6f}")


if __name__ == "__main__":
    main()
