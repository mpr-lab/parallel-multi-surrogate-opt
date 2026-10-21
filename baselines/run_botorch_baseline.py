#!/usr/bin/env python3
"""Standalone BoTorch GP + (Log)Expected Improvement baseline.

This is intentionally NOT part of the GP+RF multi-surrogate framework: it
does not import or call Orchestrator, WorkerPool, DataStore, CandidateMerge,
or any surrogate class. It only reuses the plain objective functions and
SearchSpace from src/, and (read-only) a role constant from orchestrator.py
so the warm-start reconstruction uses the exact same derivation the GP+RF
pilot used -- no framework file is modified.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from objectives import branin, hartmann6  # noqa: E402
from orchestrator import _ROLE_WARM_START  # noqa: E402 (read-only reuse, not a modification)
from run_data import load_candidates  # noqa: E402
from search_space import SearchSpace  # noqa: E402

BENCHMARK_REGISTRY = {
    "branin": {
        "objective": branin,
        "bounds": [(-5.0, 10.0), (0.0, 15.0)],
        "total_evaluations": 15,
    },
    "hartmann6": {
        "objective": hartmann6,
        "bounds": [(0.0, 1.0)] * 6,
        "total_evaluations": 21,
    },
}

N_WARM_START = 5
NUM_RESTARTS = 10
RAW_SAMPLES = 256

# Role codes for torch seeding, analogous in spirit to Orchestrator's
# _ROLE_* constants (experiment_conventions.md SS G) but local to this
# standalone script's own seed namespace.
_ROLE_GP_FIT = 100
_ROLE_ACQF_OPTIMIZE = 101


def _sub_seed(experiment_seed: int, role: int, index: int) -> int:
    return int(np.random.SeedSequence([experiment_seed, role, index]).generate_state(1)[0])


def _bounds_tensor(bounds: list[tuple[float, float]]) -> torch.Tensor:
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    return torch.tensor([lows, highs], dtype=torch.double)


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


def reconstruct_warm_start(objective_fn, bounds, seed: int, n_warm: int = N_WARM_START):
    """Reconstruct the warm-start set using the exact same derivation
    Orchestrator._run_warm_start uses: SeedSequence([seed, _ROLE_WARM_START,
    round_id=0]) feeding SearchSpace.sample(). Verified (see conversation)
    to reproduce the existing GP+RF pilot's warm-start rows bit-for-bit."""
    space = SearchSpace(bounds=bounds)
    rng = np.random.default_rng(np.random.SeedSequence([seed, _ROLE_WARM_START, 0]))
    X_warm = space.sample(n_warm, rng)
    y_warm = [float(objective_fn(x)) for x in X_warm]
    return X_warm, y_warm


def _cross_check_against_existing_pilot(benchmark: str, X_warm, y_warm) -> bool | None:
    """Best-effort consistency check against the existing GP+RF pilot's
    stored warm start, if that (gitignored, locally-generated) run
    directory happens to be present. Returns True if matched, None if no
    reference run was available to check against. Raises if present but
    mismatched -- that would indicate the two pilots have desynced."""
    reference_dir = REPO_ROOT / "results" / f"gaussian_process_random_forest_{benchmark}_seed42"
    if not reference_dir.exists():
        return None

    reference_records = load_candidates(reference_dir)
    reference_warm = sorted(
        (r for r in reference_records if r["round_id"] == 0),
        key=lambda r: r["candidate_rank_within_source"],
    )
    if len(reference_warm) != len(X_warm):
        raise ValueError(
            f"reference pilot warm start has {len(reference_warm)} rows, "
            f"expected {len(X_warm)}"
        )
    for i, ref in enumerate(reference_warm):
        if not np.allclose(ref["x"], X_warm[i]):
            raise ValueError(f"warm-start x mismatch at index {i}: {ref['x']} vs {X_warm[i]}")
        if not np.isclose(ref["y"], y_warm[i]):
            raise ValueError(f"warm-start y mismatch at index {i}: {ref['y']} vs {y_warm[i]}")
    return True


def _write_candidates_csv(records: list[dict], path: Path, dim: int) -> None:
    fieldnames = (
        ["evaluation_index", "is_warm_start"]
        + [f"x_{i}" for i in range(dim)]
        + [
            "y",
            "best_y_so_far",
            "gp_fit_time",
            "acquisition_optimization_time",
            "objective_eval_time",
            "total_iteration_time",
            "seed",
        ]
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {"evaluation_index": r["evaluation_index"], "is_warm_start": r["is_warm_start"]}
            row.update({f"x_{i}": v for i, v in enumerate(r["x"])})
            row.update(
                {
                    "y": r["y"],
                    "best_y_so_far": r["best_y_so_far"],
                    "gp_fit_time": r["gp_fit_time"],
                    "acquisition_optimization_time": r["acquisition_optimization_time"],
                    "objective_eval_time": r["objective_eval_time"],
                    "total_iteration_time": r["total_iteration_time"],
                    "seed": r["seed"],
                }
            )
            writer.writerow(row)


def run_baseline(
    benchmark: str,
    seed: int,
    output_dir: Path,
    total_evaluations: int | None = None,
) -> tuple[Path, list[dict], dict]:
    if benchmark not in BENCHMARK_REGISTRY:
        raise ValueError(f"unknown benchmark {benchmark!r}, expected one of {sorted(BENCHMARK_REGISTRY)}")

    entry = BENCHMARK_REGISTRY[benchmark]
    objective_fn = entry["objective"]
    bounds = entry["bounds"]
    if total_evaluations is None:
        total_evaluations = entry["total_evaluations"]
    if total_evaluations <= N_WARM_START:
        raise ValueError(
            f"total_evaluations ({total_evaluations}) must exceed n_warm_start ({N_WARM_START})"
        )

    dim = len(bounds)
    bounds_t = _bounds_tensor(bounds)
    n_bo_iterations = total_evaluations - N_WARM_START

    X_warm, y_warm = reconstruct_warm_start(objective_fn, bounds, seed)
    cross_checked = _cross_check_against_existing_pilot(benchmark, X_warm, y_warm)

    records: list[dict] = []
    running_best = None
    run_start = time.perf_counter()

    for i in range(N_WARM_START):
        y_i = y_warm[i]
        running_best = y_i if running_best is None else max(running_best, y_i)
        records.append(
            {
                "evaluation_index": i,
                "is_warm_start": True,
                "x": tuple(float(v) for v in X_warm[i]),
                "y": float(y_i),
                "best_y_so_far": float(running_best),
                "gp_fit_time": 0.0,
                "acquisition_optimization_time": 0.0,
                "objective_eval_time": 0.0,
                "total_iteration_time": 0.0,
                "seed": seed,
            }
        )

    train_X = torch.tensor(np.asarray(X_warm), dtype=torch.double)
    train_Y = torch.tensor(np.asarray(y_warm), dtype=torch.double).unsqueeze(-1)

    for bo_iter in range(n_bo_iterations):
        iteration_start = time.perf_counter()

        torch.manual_seed(_sub_seed(seed, _ROLE_GP_FIT, bo_iter))
        fit_start = time.perf_counter()
        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(d=dim, bounds=bounds_t),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        gp_fit_time = time.perf_counter() - fit_start

        torch.manual_seed(_sub_seed(seed, _ROLE_ACQF_OPTIMIZE, bo_iter))
        acqf_start = time.perf_counter()
        acqf = LogExpectedImprovement(model=model, best_f=train_Y.max())
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds_t,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
        acquisition_optimization_time = time.perf_counter() - acqf_start

        x_next = candidate.squeeze(0).detach().numpy()

        eval_start = time.perf_counter()
        y_next = float(objective_fn(x_next))
        objective_eval_time = time.perf_counter() - eval_start

        total_iteration_time = time.perf_counter() - iteration_start

        train_X = torch.cat([train_X, candidate.to(torch.double)], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[y_next]], dtype=torch.double)], dim=0)

        running_best = max(running_best, y_next)
        records.append(
            {
                "evaluation_index": N_WARM_START + bo_iter,
                "is_warm_start": False,
                "x": tuple(float(v) for v in x_next),
                "y": y_next,
                "best_y_so_far": float(running_best),
                "gp_fit_time": gp_fit_time,
                "acquisition_optimization_time": acquisition_optimization_time,
                "objective_eval_time": objective_eval_time,
                "total_iteration_time": total_iteration_time,
                "seed": seed,
            }
        )

    total_wall_time = time.perf_counter() - run_start

    run_name = f"botorch_gp_ei_{benchmark}_seed{seed}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_candidates_csv(records, run_dir / "candidates.csv", dim)

    metadata = {
        "run_name": run_name,
        "benchmark": benchmark,
        "method": "botorch_gp_ei",
        "acquisition_function": "LogExpectedImprovement",
        "seed": seed,
        "bounds": bounds,
        "n_warm_start": N_WARM_START,
        "total_evaluations": total_evaluations,
        "num_restarts": NUM_RESTARTS,
        "raw_samples": RAW_SAMPLES,
        "warm_start_cross_checked_against_existing_pilot": cross_checked is True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "total_wall_time": total_wall_time,
        "best_y": float(running_best),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return run_dir, records, metadata


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone BoTorch GP+EI baseline (sequential, q=1). "
        "Not part of the GP+RF multi-surrogate framework."
    )
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_REGISTRY))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir, records, metadata = run_baseline(args.benchmark, args.seed, args.output_dir)
    print(f"run: {metadata['run_name']}")
    print(f"  evaluations: {len(records)}")
    print(f"  best_y: {metadata['best_y']:.6f}")
    print(f"  wall_time: {metadata['total_wall_time']:.3f}s")
    print(f"  warm_start_cross_checked: {metadata['warm_start_cross_checked_against_existing_pilot']}")
    print(f"  output: {run_dir}")


if __name__ == "__main__":
    main()
