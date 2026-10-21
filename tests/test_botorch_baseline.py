import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from run_botorch_baseline import BENCHMARK_REGISTRY, reconstruct_warm_start, run_baseline
from run_data import load_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = REPO_ROOT / "baselines" / "run_botorch_baseline.py"


@pytest.mark.parametrize("benchmark", ["branin", "hartmann6"])
def test_warm_start_matches_existing_pilot_run(benchmark):
    reference_dir = REPO_ROOT / "results" / f"gaussian_process_random_forest_{benchmark}_seed42"
    if not reference_dir.exists():
        pytest.skip(f"{reference_dir} not present (gitignored, locally-generated pilot run)")

    entry = BENCHMARK_REGISTRY[benchmark]
    X_warm, y_warm = reconstruct_warm_start(entry["objective"], entry["bounds"], seed=42)

    reference_records = load_candidates(reference_dir)
    reference_warm = sorted(
        (r for r in reference_records if r["round_id"] == 0),
        key=lambda r: r["candidate_rank_within_source"],
    )

    assert len(reference_warm) == len(X_warm) == 5
    for i, ref in enumerate(reference_warm):
        assert np.allclose(ref["x"], X_warm[i])
        assert np.isclose(ref["y"], y_warm[i])


@pytest.mark.parametrize("benchmark,total_evaluations", [("branin", 7), ("hartmann6", 7)])
def test_correct_evaluation_counts(benchmark, total_evaluations, tmp_path):
    _, records, metadata = run_baseline(
        benchmark, seed=42, output_dir=tmp_path, total_evaluations=total_evaluations
    )

    assert len(records) == total_evaluations
    assert metadata["total_evaluations"] == total_evaluations
    assert sum(1 for r in records if r["is_warm_start"]) == 5
    assert sum(1 for r in records if not r["is_warm_start"]) == total_evaluations - 5
    assert [r["evaluation_index"] for r in records] == list(range(total_evaluations))


@pytest.mark.parametrize("benchmark", ["branin", "hartmann6"])
def test_proposed_points_remain_within_bounds(benchmark, tmp_path):
    entry = BENCHMARK_REGISTRY[benchmark]
    _, records, _ = run_baseline(benchmark, seed=42, output_dir=tmp_path, total_evaluations=7)

    low = np.array([b[0] for b in entry["bounds"]])
    high = np.array([b[1] for b in entry["bounds"]])
    for r in records:
        x = np.array(r["x"])
        assert np.all(x >= low - 1e-9)
        assert np.all(x <= high + 1e-9)


@pytest.mark.parametrize("benchmark", ["branin", "hartmann6"])
def test_best_y_so_far_is_monotonic_and_matches_running_max(benchmark, tmp_path):
    _, records, _ = run_baseline(benchmark, seed=42, output_dir=tmp_path, total_evaluations=7)

    best_values = [r["best_y_so_far"] for r in records]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best_values, best_values[1:]))

    running_max = -float("inf")
    for r in records:
        running_max = max(running_max, r["y"])
        assert r["best_y_so_far"] == pytest.approx(running_max)


def test_deterministic_given_fixed_seed(tmp_path):
    _, records_a, _ = run_baseline("branin", seed=42, output_dir=tmp_path / "a", total_evaluations=7)
    _, records_b, _ = run_baseline("branin", seed=42, output_dir=tmp_path / "b", total_evaluations=7)

    assert [r["x"] for r in records_a] == [r["x"] for r in records_b]
    assert [r["y"] for r in records_a] == [r["y"] for r in records_b]


def test_output_files_created_correctly(tmp_path):
    run_dir, records, metadata = run_baseline(
        "branin", seed=42, output_dir=tmp_path, total_evaluations=7
    )

    assert run_dir == tmp_path / "botorch_gp_ei_branin_seed42"
    candidates_path = run_dir / "candidates.csv"
    metadata_path = run_dir / "metadata.json"
    assert candidates_path.exists()
    assert metadata_path.exists()

    with open(candidates_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7
    assert rows[0]["is_warm_start"] == "True"
    assert rows[-1]["is_warm_start"] == "False"
    assert "x_0" in rows[0] and "x_1" in rows[0]

    loaded_metadata = json.loads(metadata_path.read_text())
    assert loaded_metadata["benchmark"] == "branin"
    assert loaded_metadata["method"] == "botorch_gp_ei"
    assert loaded_metadata["acquisition_function"] == "LogExpectedImprovement"
    assert loaded_metadata["seed"] == 42
    assert "git_commit" in loaded_metadata
    assert "total_wall_time" in loaded_metadata


def test_cli_runs_branin_at_full_fixed_budget(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(BASELINE_SCRIPT),
            "--benchmark", "branin",
            "--seed", "42",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "botorch_gp_ei_branin_seed42"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["total_evaluations"] == 15


def test_cli_runs_hartmann6_at_full_fixed_budget(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(BASELINE_SCRIPT),
            "--benchmark", "hartmann6",
            "--seed", "42",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "botorch_gp_ei_hartmann6_seed42"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["total_evaluations"] == 21


def test_rejects_unknown_benchmark(tmp_path):
    with pytest.raises(ValueError):
        run_baseline("not_a_real_benchmark", seed=42, output_dir=tmp_path)
