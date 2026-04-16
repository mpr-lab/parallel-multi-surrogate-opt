"""
Analyze and summarize experiment results.

Reads all *_metrics.json files from a results directory and prints a summary
table showing:
  - best_y achieved per method per seed
  - mean +/- std across seeds
  - cumulative GP fit time (the key metric for the paper's cost analysis)

Usage:
    python experiments/analyze_results.py --results-dir results/branin
    python experiments/analyze_results.py --results-dir results/
    python experiments/analyze_results.py --results-dir results/ --benchmark branin
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_metrics_files(results_dir: str, benchmark_filter: str = "") -> list[dict]:
    """Load all *_metrics.json files from results_dir (recursively)."""
    metrics = []
    for root, _, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith("_metrics.json"):
                continue
            if benchmark_filter and benchmark_filter not in fname:
                continue
            path = os.path.join(root, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                    data["_file"] = path
                    metrics.append(data)
            except Exception as e:
                print(f"Warning: could not read {path}: {e}")
    return metrics


def summarize(metrics_list: list[dict]) -> None:
    """Print a summary table grouped by method and benchmark."""
    if not metrics_list:
        print("No metrics files found.")
        return

    # Group by (benchmark, method)
    groups = defaultdict(list)
    for m in metrics_list:
        key = (m.get("benchmark", "?"), m.get("method", "?"))
        groups[key].append(m)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Benchmark':<12} {'Method':<20} {'N':>4} "
          f"{'Best Y mean':>12} {'+/-std':>8} "
          f"{'GP time mean':>13} {'RF time mean':>13}")
    print("-"*80)

    for (benchmark, method), runs in sorted(groups.items()):
        best_ys    = [r["best_y_final"] for r in runs if r.get("best_y_final") is not None]
        gp_times   = [r.get("cumulative_gp_fit_time", 0.0) for r in runs]
        rf_times   = [r.get("cumulative_rf_fit_time", 0.0) for r in runs]

        if not best_ys:
            continue

        mean_y  = np.mean(best_ys)
        std_y   = np.std(best_ys)
        mean_gp = np.mean(gp_times)
        mean_rf = np.mean(rf_times)

        print(
            f"{benchmark:<12} {method:<20} {len(best_ys):>4} "
            f"{mean_y:>12.4f} {std_y:>8.4f} "
            f"{mean_gp:>13.3f} {mean_rf:>13.3f}"
        )

    print("="*80)

    # Per-run detail
    print("\n\nPer-run detail (sorted by method, then seed):")
    print(f"{'Benchmark':<12} {'Method':<20} {'Seed':>8} {'Best Y':>10} "
          f"{'GP time':>9} {'n_obs':>7}")
    print("-"*70)
    for m in sorted(metrics_list,
                    key=lambda x: (x.get("benchmark",""), x.get("method",""),
                                   x.get("seed", 0))):
        n_rounds = len(m.get("rounds", []))
        n_obs    = m["rounds"][-1]["n_obs"] if m.get("rounds") else "?"
        print(
            f"{m.get('benchmark','?'):<12} {m.get('method','?'):<20} "
            f"{m.get('seed','?'):>8} "
            f"{m.get('best_y_final', float('nan')):>10.4f} "
            f"{m.get('cumulative_gp_fit_time', 0.0):>9.3f} "
            f"{n_obs:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to search for *_metrics.json files"
    )
    parser.add_argument(
        "--benchmark", type=str, default="",
        help="Filter to a specific benchmark name (e.g. 'branin')"
    )
    args = parser.parse_args()

    metrics_list = load_metrics_files(args.results_dir, args.benchmark)
    print(f"Found {len(metrics_list)} metrics files in {args.results_dir!r}")
    summarize(metrics_list)


if __name__ == "__main__":
    main()
