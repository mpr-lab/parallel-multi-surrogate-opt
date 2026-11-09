#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datastore import DataStore  # noqa: E402
from objectives import branin, hartmann6, rastrigin  # noqa: E402
from objectives.classical_ml import (  # noqa: E402
    DEFAULT_SMOKE_TEST_DATASET as DEFAULT_CLASSICAL_ML_SMOKE_TEST_DATASET,
    LOGISTIC_REGRESSION_EXPANDED_PARAMETER_SPACE,
    LOGISTIC_REGRESSION_PARAMETER_SPACE,
    RANDOM_FOREST_EXPANDED_PARAMETER_SPACE,
    RANDOM_FOREST_PARAMETER_SPACE,
    XGBOOST_EXPANDED_PARAMETER_SPACE,
    XGBOOST_PARAMETER_SPACE,
    make_logistic_regression_objective,
    make_random_forest_objective,
    make_xgboost_objective,
)
from objectives.huggingface_distilbert import (  # noqa: E402
    HUGGINGFACE_EXPANDED_PARAMETER_SPACE,
    HUGGINGFACE_PARAMETER_SPACE,
    N_EVAL_EXAMPLES as HF_DEFAULT_EVAL_EXAMPLES,
    N_TRAIN_EXAMPLES as HF_DEFAULT_TRAIN_EXAMPLES,
    make_distilbert_imdb_objective,
)
from orchestrator import Orchestrator  # noqa: E402
from parameter_space import ParameterSpace  # noqa: E402
from search_space import SearchSpace  # noqa: E402
from surrogates import GaussianProcessSurrogate, RandomForestSurrogate, TPESurrogate  # noqa: E402
from worker_pool import WorkerPool  # noqa: E402

# "Legacy" shape: a bare objective + plain continuous bounds, used
# unchanged by every synthetic benchmark. SearchSpace is built directly
# from "bounds" and no decode step is involved anywhere downstream.
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

# "Parameterized" shape: a ParameterSpace (mixed/discrete/categorical
# dimensions) plus an objective FACTORY. The factory is called once, at
# experiment-setup time (before Orchestrator.run()), so one-time dataset/
# model loading never contaminates any single candidate's objective-
# evaluation timing. `make_objective(config, seed) -> Callable`.
PARAMETERIZED_BENCHMARK_REGISTRY = {
    "logistic_regression": {
        "parameter_space": LOGISTIC_REGRESSION_PARAMETER_SPACE,
        "make_objective": lambda config, seed: make_logistic_regression_objective(
            dataset=config.dataset or DEFAULT_CLASSICAL_ML_SMOKE_TEST_DATASET,
            data_dir=config.data_dir,
            seed=seed,
        ),
    },
    "random_forest": {
        "parameter_space": RANDOM_FOREST_PARAMETER_SPACE,
        "make_objective": lambda config, seed: make_random_forest_objective(
            dataset=config.dataset or DEFAULT_CLASSICAL_ML_SMOKE_TEST_DATASET,
            data_dir=config.data_dir,
            seed=seed,
        ),
    },
    "xgboost": {
        "parameter_space": XGBOOST_PARAMETER_SPACE,
        "make_objective": lambda config, seed: make_xgboost_objective(
            dataset=config.dataset or DEFAULT_CLASSICAL_ML_SMOKE_TEST_DATASET,
            data_dir=config.data_dir,
            seed=seed,
        ),
    },
    # "Expanded" search spaces for the matched 15-evaluation experiment
    # protocol (n_initial=5, n_rounds=5, batch_size=2). Separate benchmark
    # names from their historical-grid counterparts above -- those are
    # unchanged. Each reuses the identical make_*_objective factory (same
    # dataset/objective/scaling behavior), only the ParameterSpace differs.
    "logistic_regression_expanded": {
        "parameter_space": LOGISTIC_REGRESSION_EXPANDED_PARAMETER_SPACE,
        "make_objective": lambda config, seed: make_logistic_regression_objective(
            dataset=config.dataset or DEFAULT_CLASSICAL_ML_SMOKE_TEST_DATASET,
            data_dir=config.data_dir,
            seed=seed,
        ),
    },
    "random_forest_expanded": {
        "parameter_space": RANDOM_FOREST_EXPANDED_PARAMETER_SPACE,
        "make_objective": lambda config, seed: make_random_forest_objective(
            dataset=config.dataset or DEFAULT_CLASSICAL_ML_SMOKE_TEST_DATASET,
            data_dir=config.data_dir,
            seed=seed,
        ),
    },
    "xgboost_expanded": {
        "parameter_space": XGBOOST_EXPANDED_PARAMETER_SPACE,
        "make_objective": lambda config, seed: make_xgboost_objective(
            dataset=config.dataset or DEFAULT_CLASSICAL_ML_SMOKE_TEST_DATASET,
            data_dir=config.data_dir,
            seed=seed,
        ),
    },
    "distilbert_imdb": {
        "parameter_space": HUGGINGFACE_PARAMETER_SPACE,
        "make_objective": lambda config, seed: make_distilbert_imdb_objective(
            cache_dir=config.cache_dir,
            output_dir=config.model_output_dir,
            device=config.device,
            seed=seed,
            train_subset_size=config.train_subset_size or HF_DEFAULT_TRAIN_EXAMPLES,
            eval_subset_size=config.eval_subset_size or HF_DEFAULT_EVAL_EXAMPLES,
        ),
    },
    # Expanded search space for the matched 15-evaluation protocol.
    # Separate benchmark name from distilbert_imdb above -- that
    # historical profiling benchmark's space/behavior is unchanged.
    # Reuses the identical make_distilbert_imdb_objective factory (same
    # model/dataset/subset sizes/max length/epochs/device/seed/timing);
    # only the ParameterSpace differs.
    "distilbert_imdb_expanded": {
        "parameter_space": HUGGINGFACE_EXPANDED_PARAMETER_SPACE,
        "make_objective": lambda config, seed: make_distilbert_imdb_objective(
            cache_dir=config.cache_dir,
            output_dir=config.model_output_dir,
            device=config.device,
            seed=seed,
            train_subset_size=config.train_subset_size or HF_DEFAULT_TRAIN_EXAMPLES,
            eval_subset_size=config.eval_subset_size or HF_DEFAULT_EVAL_EXAMPLES,
        ),
    },
}

_overlap = set(BENCHMARK_REGISTRY) & set(PARAMETERIZED_BENCHMARK_REGISTRY)
assert not _overlap, f"benchmark name(s) registered twice: {_overlap}"
ALL_BENCHMARK_NAMES = sorted(set(BENCHMARK_REGISTRY) | set(PARAMETERIZED_BENCHMARK_REGISTRY))

# Each entry: (make_surrogate_a, make_surrogate_b), both `Callable[[int | None], BaseSurrogate]`.
# Registering a new pair means adding an entry here -- the Orchestrator never
# sees this registry and does not change when a pair is added.
SURROGATE_PAIR_REGISTRY = {
    "gp_rf": (
        lambda seed: GaussianProcessSurrogate(random_state=seed),
        lambda seed: RandomForestSurrogate(random_state=seed),
    ),
    "gp_tpe": (
        lambda seed: GaussianProcessSurrogate(random_state=seed),
        # TPESurrogate has no random_state: fit() is fully deterministic
        # and all of its randomness flows through the rng the Orchestrator
        # already passes explicitly into propose().
        lambda seed: TPESurrogate(),
    ),
}


@dataclass(frozen=True)
class MLWorkloadConfig:
    """Options relevant only to classical-ML / Hugging Face benchmarks.
    Synthetic benchmarks (branin/hartmann6/rastrigin) never read this.
    Nothing here defaults to a hard-coded filesystem path -- cache/output
    directories are None unless the caller explicitly sets them via CLI.
    """

    dataset: str | None = None
    data_dir: Path | None = None
    cache_dir: Path | None = None
    model_output_dir: Path | None = None
    device: str = "cpu"
    train_subset_size: int | None = None
    eval_subset_size: int | None = None


def _build_benchmark(
    args: argparse.Namespace,
) -> tuple[SearchSpace, Callable, ParameterSpace | None]:
    """Returns (search_space, objective_fn, parameter_space). parameter_space
    is None for every legacy (synthetic) benchmark -- Orchestrator's
    default behavior is then byte-for-byte what it was before this
    existed (no decoding, no evaluated-config tracking, no oversampling).
    """
    if args.benchmark in BENCHMARK_REGISTRY:
        entry = BENCHMARK_REGISTRY[args.benchmark]
        return SearchSpace(bounds=entry["bounds"]), entry["objective"], None

    entry = PARAMETERIZED_BENCHMARK_REGISTRY[args.benchmark]
    parameter_space = entry["parameter_space"]
    config = MLWorkloadConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        model_output_dir=args.model_output_dir,
        device=args.device,
        train_subset_size=args.train_subset_size,
        eval_subset_size=args.eval_subset_size,
    )
    objective_fn = entry["make_objective"](config, args.seed)
    return SearchSpace(bounds=parameter_space.bounds), objective_fn, parameter_space


def _check_finite_space_budget(args: argparse.Namespace) -> None:
    """Fail fast, before any experiment setup (dataset/model loading
    included), if the requested evaluation budget cannot possibly fit in
    a ParameterSpace-backed benchmark's finite configuration space.

    Deliberately a pure registry lookup -- it does not call
    `_build_benchmark`/`make_objective`, so a oversized budget is
    rejected before paying for e.g. Hugging Face dataset/model loading.
    Continuous synthetic benchmarks (no ParameterSpace) have no finite
    ceiling to check against and are skipped entirely.
    """
    if args.benchmark not in PARAMETERIZED_BENCHMARK_REGISTRY:
        return

    parameter_space = PARAMETERIZED_BENCHMARK_REGISTRY[args.benchmark]["parameter_space"]
    total_configurations = parameter_space.total_configurations
    requested_evaluations = args.n_initial + args.n_rounds * args.batch_size

    if requested_evaluations > total_configurations:
        raise ValueError(
            f"requested evaluations ({requested_evaluations} = "
            f"--n-initial {args.n_initial} + --n-rounds {args.n_rounds} * "
            f"--batch-size {args.batch_size}) exceed the finite parameter "
            f"space for benchmark '{args.benchmark}' "
            f"({total_configurations} total unique configurations). "
            f"Reduce --n-initial, --n-rounds, and/or --batch-size so the "
            f"total stays at or below {total_configurations}."
        )


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
    # decoded_params is only ever present when a ParameterSpace/decode_fn
    # was used (ML benchmarks) -- for every synthetic benchmark this is
    # absent and the column set below is identical to before this existed.
    has_decoded = "decoded_params" in records[0]
    param_names = sorted(records[0]["decoded_params"]) if has_decoded else []

    fieldnames = (
        [f"x_{i}" for i in range(dim)]
        + [f"param_{name}" for name in param_names]
        + [k for k in records[0] if k not in ("x", "decoded_params")]
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {f"x_{i}": v for i, v in enumerate(record["x"])}
            if has_decoded:
                row.update({f"param_{k}": v for k, v in record["decoded_params"].items()})
            row.update({k: v for k, v in record.items() if k not in ("x", "decoded_params")})
            writer.writerow(row)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one controlled multi-surrogate optimization experiment."
    )
    parser.add_argument("--benchmark", required=True, choices=ALL_BENCHMARK_NAMES)
    parser.add_argument("--pair", required=True, choices=sorted(SURROGATE_PAIR_REGISTRY))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")

    # ML-workload-only options. Ignored by synthetic benchmarks. None of
    # these default to a hard-coded filesystem path.
    parser.add_argument(
        "--dataset", type=str, default=None,
        help=(
            "Dataset name for classical-ML benchmarks. Defaults to a "
            "SMOKE-TEST dataset only (objectives.classical_ml."
            "DEFAULT_SMOKE_TEST_DATASET) -- the real experiment dataset is "
            "a separate, not-yet-made decision; pass this explicitly for "
            "any non-smoke-test run."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Hugging Face cache directory for models/datasets (default: transformers/datasets' own default cache)",
    )
    parser.add_argument("--model-output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-subset-size", type=int, default=None)
    parser.add_argument("--eval-subset-size", type=int, default=None)

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.batch_size % 2 != 0:
        raise ValueError(
            f"--batch-size must be even for the fixed 50/50 allocation, got {args.batch_size}"
        )
    n_proposals_per_surrogate = args.batch_size // 2
    _check_finite_space_budget(args)

    make_surrogate_a, make_surrogate_b = SURROGATE_PAIR_REGISTRY[args.pair]
    search_space, objective_fn, parameter_space = _build_benchmark(args)

    surrogate_a = make_surrogate_a(args.seed)
    surrogate_b = make_surrogate_b(args.seed)
    datastore = DataStore()
    worker_pool = WorkerPool(n_workers=args.workers, experiment_seed=args.seed)

    orchestrator = Orchestrator(
        search_space=search_space,
        objective_fn=objective_fn,
        surrogate_a=surrogate_a,
        surrogate_b=surrogate_b,
        datastore=datastore,
        worker_pool=worker_pool,
        n_initial=args.n_initial,
        n_proposals_per_surrogate=n_proposals_per_surrogate,
        n_rounds=args.n_rounds,
        experiment_seed=args.seed,
        parameter_space=parameter_space,
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
