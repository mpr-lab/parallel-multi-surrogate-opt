import csv
import json
import subprocess
import sys
from pathlib import Path

import run_experiment
from surrogates import GaussianProcessSurrogate, TPESurrogate

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "run_experiment.py"


def test_gp_tpe_is_a_registered_pair_choice():
    assert "gp_tpe" in run_experiment.SURROGATE_PAIR_REGISTRY
    assert "gp_tpe" in sorted(run_experiment.SURROGATE_PAIR_REGISTRY)


def test_cli_accepts_gp_tpe_pair():
    args = run_experiment.parse_args(
        ["--benchmark", "branin", "--pair", "gp_tpe", "--seed", "0"]
    )
    assert args.pair == "gp_tpe"


def test_gp_tpe_registry_entry_constructs_gp_and_tpe():
    make_surrogate_a, make_surrogate_b = run_experiment.SURROGATE_PAIR_REGISTRY["gp_tpe"]
    surrogate_a = make_surrogate_a(42)
    surrogate_b = make_surrogate_b(42)

    assert isinstance(surrogate_a, GaussianProcessSurrogate)
    assert isinstance(surrogate_b, TPESurrogate)
    assert surrogate_a.name == "gaussian_process"
    assert surrogate_b.name == "tpe"


def test_gp_rf_registry_entry_is_unaffected_by_gp_tpe_addition():
    make_surrogate_a, make_surrogate_b = run_experiment.SURROGATE_PAIR_REGISTRY["gp_rf"]
    surrogate_a = make_surrogate_a(42)
    surrogate_b = make_surrogate_b(42)

    assert surrogate_a.name == "gaussian_process"
    assert surrogate_b.name == "random_forest"


def test_cli_runs_gp_tpe_end_to_end_on_branin(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--benchmark", "branin",
            "--pair", "gp_tpe",
            "--seed", "42",
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

    run_dir = tmp_path / "gaussian_process_tpe_branin_seed42"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["surrogate_pair"] == "gp_tpe"
    assert metadata["surrogate_a_name"] == "gaussian_process"
    assert metadata["surrogate_b_name"] == "tpe"
    assert metadata["total_evaluations"] == 15

    with open(run_dir / "candidates.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15

    warm_start_rows = [r for r in rows if r["round_id"] == "0"]
    assert len(warm_start_rows) == 5
    assert all(r["surrogate_source"] == "random" for r in warm_start_rows)

    # Each of the 5 post-warm-start rounds must contribute exactly one
    # gaussian_process candidate and one tpe candidate (batch_size=2,
    # n_proposals_per_surrogate=1 at a 50/50 allocation).
    for round_id in ("1", "2", "3", "4", "5"):
        round_rows = [r for r in rows if r["round_id"] == round_id]
        sources = sorted(r["surrogate_source"] for r in round_rows)
        assert sources == ["gaussian_process", "tpe"], round_id

    round_metrics = json.loads((run_dir / "round_metrics.json").read_text())
    assert len(round_metrics["rounds"]) == 5
    for record in round_metrics["rounds"]:
        assert record["surrogate_a_name"] == "gaussian_process"
        assert record["surrogate_b_name"] == "tpe"
        assert record["surrogate_a_candidates_contributed"] == 1
        assert record["surrogate_b_candidates_contributed"] == 1
        for time_field in (
            "surrogate_a_fit_time",
            "surrogate_b_fit_time",
            "surrogate_a_proposal_time",
            "surrogate_b_proposal_time",
            "evaluation_wall_time",
            "total_round_wall_time",
        ):
            assert record[time_field] >= 0.0


def test_gp_tpe_deterministic_across_repeated_runs(tmp_path):
    def _run(output_dir):
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--benchmark", "branin",
                "--pair", "gp_tpe",
                "--seed", "42",
                "--n-initial", "5",
                "--n-rounds", "5",
                "--batch-size", "2",
                "--workers", "1",
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        run_dir = output_dir / "gaussian_process_tpe_branin_seed42"
        with open(run_dir / "candidates.csv") as f:
            rows = list(csv.DictReader(f))
        return [(r["x_0"], r["x_1"], r["y"], r["surrogate_source"]) for r in rows]

    first = _run(tmp_path / "run1")
    second = _run(tmp_path / "run2")

    assert first == second


def test_gp_tpe_with_parameter_space_benchmark_preserves_decoded_dedup(tmp_path):
    # logistic_regression's ParameterSpace has only 8 total configurations;
    # a successful 5-evaluation run with gp_tpe confirms decode-aware
    # deduplication/refill still functions for this pair, exactly as it
    # already does for gp_rf.
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--benchmark", "logistic_regression",
            "--pair", "gp_tpe",
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

    run_dir = tmp_path / "gaussian_process_tpe_logistic_regression_seed0"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["total_evaluations"] == 5

    with open(run_dir / "candidates.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert "param_C" in rows[0] and "param_penalty" in rows[0]
    # No duplicate decoded (C, penalty) configuration across all 5 evals.
    configs = {(r["param_C"], r["param_penalty"]) for r in rows}
    assert len(configs) == 5
