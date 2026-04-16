"""
Lightweight reproducibility check.

Runs the same config + seed TWICE and verifies that the results match exactly
(or within a small floating-point tolerance).

This is NOT a unit test -- it is a practical smoke test you run before
submitting paper results to make sure:
  1. The same seed always produces the same observations.
  2. The same best-y trajectory is produced both times.

Usage:
    python scripts/check_reproducibility.py --config configs/branin.yaml --seed 42
    python scripts/check_reproducibility.py --config configs/hartmann.yaml --seed 123
    python scripts/check_reproducibility.py --config configs/branin.yaml --tol 1e-8

Exit code:
    0 = PASS (runs are reproducible)
    1 = FAIL (runs differ)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seeds import set_global_seed
from src.utils.serialization import load_yaml
from src.objectives.synthetic import get_benchmark
from src.orchestrator import run_optimization


def run_once(cfg: dict, objective_fn, bounds: list, label: str):
    """
    Run the optimizer once with cfg["seed"] and return the raw results.

    We call set_global_seed() here (not just in the orchestrator) to ensure
    the global state is reset to exactly the same starting point for run 1
    and run 2.

    Returns:
        (X, y, best_y_per_round) -- all observations and the convergence curve.
    """
    seed = int(cfg["seed"])
    set_global_seed(seed)
    print(f"  [{label}] seed={seed} ...", end=" ", flush=True)
    dataset, run_metrics = run_optimization(cfg, objective_fn, bounds)
    best = dataset.best_y
    print(f"done.  best_y={best:.6f}  n_obs={dataset.n_obs}")
    return dataset.X, dataset.y, run_metrics.best_y_per_round()


def check_reproducibility(
    config_path: str,
    seed: int,
    tol: float = 1e-10,
) -> bool:
    """
    Run the same experiment twice and check that results match.

    Args:
        config_path: Path to YAML config file.
        seed:        Seed to use for both runs.
        tol:         Absolute tolerance for floating-point comparisons.
                     Default 1e-10 is tight but realistic for same-platform runs.
                     Use 1e-6 if running on different hardware.

    Returns:
        True if all checks pass, False otherwise.
    """
    cfg = load_yaml(config_path)
    cfg["seed"]       = seed
    cfg["output_dir"] = os.path.join(cfg.get("output_dir", "results"), "_repro_check")

    benchmark_name = cfg.get("benchmark", "branin")
    objective_fn, bounds, known_optimum = get_benchmark(benchmark_name)

    print(f"\nReproducibility check")
    print(f"  Config     : {config_path}")
    print(f"  Benchmark  : {benchmark_name}  (optimum ~= {known_optimum})")
    print(f"  Seed       : {seed}")
    print(f"  Tolerance  : {tol:.0e}")
    print("-"*50)

    X1, y1, best1 = run_once(cfg, objective_fn, bounds, "Run 1")
    X2, y2, best2 = run_once(cfg, objective_fn, bounds, "Run 2")

    # ------------------------------------------------------------------ #
    #  Run the checks                                                      #
    # ------------------------------------------------------------------ #
    checks = []  # list of (name, passed: bool, detail: str)

    # Check 1: same number of observations
    n1, n2 = len(y1), len(y2)
    ok = (n1 == n2)
    checks.append(("n_observations", ok,
                    f"{n1} obs in both" if ok
                    else f"Run 1: {n1}, Run 2: {n2}"))

    # Check 2: X configs are identical
    if X1.shape == X2.shape:
        x_diff = float(np.max(np.abs(X1 - X2)))
        ok = (x_diff <= tol)
        checks.append(("X_configs_match", ok,
                        f"max |diff| = {x_diff:.2e}  (tol={tol:.0e})"))
    else:
        checks.append(("X_configs_match", False,
                        f"shapes differ: {X1.shape} vs {X2.shape}"))

    # Check 3: y values are identical
    if len(y1) == len(y2):
        y_diff = float(np.max(np.abs(y1 - y2)))
        ok = (y_diff <= tol)
        checks.append(("y_values_match", ok,
                        f"max |diff| = {y_diff:.2e}  (tol={tol:.0e})"))

    # Check 4: convergence curves are identical
    if len(best1) == len(best2):
        curve_diff = float(np.max(np.abs(np.array(best1) - np.array(best2))))
        ok = (curve_diff <= tol)
        checks.append(("best_y_curve_match", ok,
                        f"max |diff| = {curve_diff:.2e}  (tol={tol:.0e})"))
    else:
        checks.append(("best_y_curve_match", False,
                        f"different number of rounds: {len(best1)} vs {len(best2)}"))

    # ------------------------------------------------------------------ #
    #  Print results                                                       #
    # ------------------------------------------------------------------ #
    all_passed = all(ok for _, ok, _ in checks)

    print("\nCheck results:")
    for name, passed, detail in checks:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name:<26}  {detail}")

    print()
    if all_passed:
        print("OVERALL: PASS -- both runs produced identical results.")
        print("         The implementation is reproducible given the same seed.")
    else:
        print("OVERALL: FAIL -- runs differ.")
        print("         Look for any randomized component that does not receive")
        print("         an explicit seed (check random_state= in sklearn calls,")
        print("         np.random calls not using the rng object, etc.).")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check experiment reproducibility")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed to use for both runs (default: 42)"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-10,
        help="Absolute tolerance for comparing floating-point results "
             "(default: 1e-10).  Use 1e-6 if comparing across machines."
    )
    args = parser.parse_args()

    passed = check_reproducibility(args.config, args.seed, args.tol)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
