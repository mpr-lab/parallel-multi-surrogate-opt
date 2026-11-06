import json
import subprocess
import sys
from pathlib import Path

import pytest

import run_experiment
from parameter_space import ParameterSpace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "run_experiment.py"


def test_all_expected_benchmark_names_are_registered():
    assert run_experiment.ALL_BENCHMARK_NAMES == sorted(
        [
            "branin",
            "hartmann6",
            "rastrigin",
            "logistic_regression",
            "random_forest",
            "xgboost",
            "logistic_regression_expanded",
            "random_forest_expanded",
            "xgboost_expanded",
            "distilbert_imdb",
        ]
    )


def test_legacy_and_parameterized_registries_do_not_overlap():
    legacy = set(run_experiment.BENCHMARK_REGISTRY)
    parameterized = set(run_experiment.PARAMETERIZED_BENCHMARK_REGISTRY)
    assert legacy.isdisjoint(parameterized)


def test_every_parameterized_entry_has_parameter_space_and_factory():
    for name, entry in run_experiment.PARAMETERIZED_BENCHMARK_REGISTRY.items():
        assert isinstance(entry["parameter_space"], ParameterSpace), name
        assert callable(entry["make_objective"]), name


def test_legacy_benchmark_build_has_no_parameter_space():
    args = run_experiment.parse_args(
        ["--benchmark", "branin", "--pair", "gp_rf", "--seed", "0"]
    )
    search_space, objective_fn, parameter_space = run_experiment._build_benchmark(args)

    assert parameter_space is None
    assert search_space.dim == 2
    assert objective_fn is run_experiment.branin


def test_parameterized_benchmark_build_has_parameter_space_and_unit_bounds():
    args = run_experiment.parse_args(
        ["--benchmark", "logistic_regression", "--pair", "gp_rf", "--seed", "0"]
    )
    search_space, objective_fn, parameter_space = run_experiment._build_benchmark(args)

    assert parameter_space is not None
    assert search_space.dim == 2
    # Every ChoiceDimension's raw encoding lives in [0, 1).
    assert search_space._low.tolist() == [0.0, 0.0]
    assert search_space._high.tolist() == [1.0, 1.0]
    decoded = parameter_space.decode([0.0, 0.0])
    assert decoded == {"C": 0.01, "penalty": "l1"}


def test_cli_runs_random_forest_benchmark_end_to_end(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--benchmark", "random_forest",
            "--pair", "gp_rf",
            "--seed", "0",
            "--dataset", "synthetic_classification",
            "--n-initial", "3",
            "--n-rounds", "1",
            "--batch-size", "2",
            "--workers", "1",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "gaussian_process_random_forest_random_forest_seed0"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["total_evaluations"] == 5

    candidate_lines = (run_dir / "candidates.csv").read_text().strip().splitlines()
    header = candidate_lines[0].split(",")
    assert "param_n_estimators" in header
    assert "param_max_depth" in header


# -- finite-space budget pre-check ----------------------------------------


def test_budget_precheck_rejects_requested_evaluations_exceeding_space_size():
    # logistic_regression's ParameterSpace has 4 * 2 = 8 configurations.
    args = run_experiment.parse_args(
        [
            "--benchmark", "logistic_regression", "--pair", "gp_rf", "--seed", "0",
            "--n-initial", "5", "--n-rounds", "3", "--batch-size", "2",  # 5 + 3*2 = 11 > 8
        ]
    )
    with pytest.raises(ValueError) as exc_info:
        run_experiment._check_finite_space_budget(args)

    message = str(exc_info.value)
    assert "11" in message  # requested evaluations
    assert "8" in message  # total unique configurations
    assert "--n-initial" in message and "--n-rounds" in message and "--batch-size" in message


def test_budget_precheck_allows_exact_fit_budget():
    args = run_experiment.parse_args(
        [
            "--benchmark", "logistic_regression", "--pair", "gp_rf", "--seed", "0",
            "--n-initial", "4", "--n-rounds", "2", "--batch-size", "2",  # 4 + 2*2 = 8 == 8
        ]
    )
    run_experiment._check_finite_space_budget(args)  # must not raise


def test_budget_precheck_allows_under_budget():
    args = run_experiment.parse_args(
        [
            "--benchmark", "logistic_regression", "--pair", "gp_rf", "--seed", "0",
            "--n-initial", "3", "--n-rounds", "1", "--batch-size", "2",  # 3 + 2 = 5 < 8
        ]
    )
    run_experiment._check_finite_space_budget(args)  # must not raise


@pytest.mark.parametrize("benchmark", ["branin", "hartmann6", "rastrigin"])
def test_budget_precheck_skips_continuous_synthetic_benchmarks(benchmark):
    args = run_experiment.parse_args(
        [
            "--benchmark", benchmark, "--pair", "gp_rf", "--seed", "0",
            # Deliberately enormous -- must still pass, no finite ceiling exists.
            "--n-initial", "10000", "--n-rounds", "10000", "--batch-size", "10000",
        ]
    )
    run_experiment._check_finite_space_budget(args)  # must not raise


@pytest.mark.parametrize(
    "benchmark,total_configurations",
    [
        ("logistic_regression_expanded", 80),
        ("random_forest_expanded", 152),
        ("xgboost_expanded", 96),
    ],
)
def test_budget_precheck_allows_the_matched_15_evaluation_protocol(benchmark, total_configurations):
    # n_initial=5, n_rounds=5, batch_size=2 -> 5 + 5*2 = 15 real evaluations,
    # comfortably under each expanded space's total_configurations.
    args = run_experiment.parse_args(
        [
            "--benchmark", benchmark, "--pair", "gp_rf", "--seed", "42",
            "--n-initial", "5", "--n-rounds", "5", "--batch-size", "2",
        ]
    )
    run_experiment._check_finite_space_budget(args)  # must not raise

    entry = run_experiment.PARAMETERIZED_BENCHMARK_REGISTRY[benchmark]
    assert entry["parameter_space"].total_configurations == total_configurations
    assert 15 <= total_configurations


def test_historical_grid_benchmarks_are_unchanged_by_the_expanded_additions():
    assert run_experiment.LOGISTIC_REGRESSION_PARAMETER_SPACE.total_configurations == 8
    assert run_experiment.RANDOM_FOREST_PARAMETER_SPACE.total_configurations == 9
    assert run_experiment.XGBOOST_PARAMETER_SPACE.total_configurations == 9


def test_cli_runs_logistic_regression_expanded_with_matched_protocol(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--benchmark", "logistic_regression_expanded",
            "--pair", "gp_rf",
            "--seed", "42",
            "--dataset", "synthetic_classification",
            "--n-initial", "5",
            "--n-rounds", "5",
            "--batch-size", "2",
            "--workers", "1",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "gaussian_process_random_forest_logistic_regression_expanded_seed42"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["total_evaluations"] == 15

    candidate_lines = (run_dir / "candidates.csv").read_text().strip().splitlines()
    assert len(candidate_lines) == 1 + 15  # header + 15 rows
    header = candidate_lines[0].split(",")
    assert "param_C" in header and "param_penalty" in header


def test_cli_rejects_over_budget_run_before_building_benchmark(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--benchmark", "logistic_regression",
            "--pair", "gp_rf",
            "--seed", "0",
            "--dataset", "synthetic_classification",
            "--n-initial", "5",
            "--n-rounds", "3",
            "--batch-size", "2",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "exceed the finite parameter space" in result.stderr
    # No run directory should have been created -- it failed before any
    # experiment setup (dataset loading included).
    assert not any(tmp_path.iterdir())
