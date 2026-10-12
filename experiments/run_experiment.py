#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datastore import DataStore  # noqa: E402
from objectives import branin, hartmann6, rastrigin  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from search_space import SearchSpace  # noqa: E402
from surrogates import GaussianProcessSurrogate, RandomForestSurrogate  # noqa: E402
from worker_pool import WorkerPool  # noqa: E402

BENCHMARK_REGISTRY = {
    "branin": {
        "objective": branin,
        "bounds": [(-5.0, 10.0), (0.0, 15.0)],
    },
    "hartmann6": {
        "objective": hartmann6,
        "bounds": [(0.0, 1.0)] * 6,
    },
    "rastrigin": {
        "objective": rastrigin,
        "bounds": [(-5.12, 5.12)] * 5,
    },
}

# Each entry: (make_surrogate_a, make_surrogate_b), both `Callable[[int | None], BaseSurrogate]`.
# Registering a new pair means adding an entry here -- the Orchestrator never
# sees this registry and does not change when a pair is added.
SURROGATE_PAIR_REGISTRY = {
    "gp_rf": (
        lambda seed: GaussianProcessSurrogate(random_state=seed),
        lambda seed: RandomForestSurrogate(random_state=seed),
    ),
}


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _write_candidates_csv(records: list[dict], path: Path) -> None:
    if not records:
        path.write_text("")
        return
    dim = len(records[0]["x"])
    fieldnames = [f"x_{i}" for i in range(dim)] + [k for k in records[0] if k != "x"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {f"x_{i}": v for i, v in enumerate(record["x"])}
            row.update({k: v for k, v in record.items() if k != "x"})
            writer.writerow(row)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one controlled multi-surrogate optimization experiment."
    )
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_REGISTRY))
    parser.add_argument("--pair", required=True, choices=sorted(SURROGATE_PAIR_REGISTRY))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.batch_size % 2 != 0:
        raise ValueError(
            f"--batch-size must be even for the fixed 50/50 allocation, got {args.batch_size}"
        )
    n_proposals_per_surrogate = args.batch_size // 2

    benchmark = BENCHMARK_REGISTRY[args.benchmark]
    make_surrogate_a, make_surrogate_b = SURROGATE_PAIR_REGISTRY[args.pair]

    search_space = SearchSpace(bounds=benchmark["bounds"])
    surrogate_a = make_surrogate_a(args.seed)
    surrogate_b = make_surrogate_b(args.seed)
    datastore = DataStore()
    worker_pool = WorkerPool(n_workers=args.workers, experiment_seed=args.seed)

    orchestrator = Orchestrator(
        search_space=search_space,
        objective_fn=benchmark["objective"],
        surrogate_a=surrogate_a,
        surrogate_b=surrogate_b,
        datastore=datastore,
        worker_pool=worker_pool,
        n_initial=args.n_initial,
        n_proposals_per_surrogate=n_proposals_per_surrogate,
        n_rounds=args.n_rounds,
        experiment_seed=args.seed,
    )

    run_name = f"{surrogate_a.name}_{surrogate_b.name}_{args.benchmark}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    orchestrator.run()
    wall_time = time.perf_counter() - wall_start

    _write_candidates_csv(datastore.get_all(), run_dir / "candidates.csv")
    orchestrator.metrics.to_json(str(run_dir / "round_metrics.json"))

    metadata = {
        "run_name": run_name,
        "benchmark": args.benchmark,
        "surrogate_pair": args.pair,
        "surrogate_a_name": surrogate_a.name,
        "surrogate_b_name": surrogate_b.name,
        "seed": args.seed,
        "n_initial": args.n_initial,
        "n_rounds": args.n_rounds,
        "batch_size": args.batch_size,
        "n_proposals_per_surrogate": n_proposals_per_surrogate,
        "worker_count": args.workers,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "total_evaluations": len(datastore),
        "total_wall_time": wall_time,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"run: {run_name}")
    print(f"  evaluations: {len(datastore)}")
    print(f"  best_y: {datastore.best_y:.6f}")
    print(f"  wall_time: {wall_time:.3f}s")
    print(f"  output: {run_dir}")


if __name__ == "__main__":
    main()
