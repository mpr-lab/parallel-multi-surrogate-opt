import json
import subprocess
import sys
from pathlib import Path

import pytest

from plot_run import generate_all_plots

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT_SCRIPT = REPO_ROOT / "experiments" / "run_experiment.py"
PLOT_RUN_SCRIPT = REPO_ROOT / "analysis" / "plot_run.py"

EXPECTED_PLOTS = (
    "best_objective_vs_evaluations.png",
    "best_objective_vs_wall_time.png",
    "timing_breakdown_by_round.png",
    "surrogate_timing_by_round.png",
    "surrogate_contribution.png",
)


def _build_synthetic_run(run_dir: Path) -> None:
    header = (
        "x_0,y,round_id,candidate_id,surrogate_source,"
        "candidate_rank_within_source,eval_start_time,eval_end_time,"
        "eval_duration,seed,improved_global_best"
    )
    rows = [
        "1.0,-5.0,0,r0_c0,random,0,10.00,10.10,0.10,1,True",
        "2.0,-3.0,0,r0_c1,random,1,10.10,10.20,0.10,2,True",
        "3.0,-1.0,1,r1_c0,gaussian_process,0,10.30,10.40,0.10,3,True",
        "4.0,-2.0,1,r1_c1,random_forest,0,10.30,10.42,0.12,4,False",
    ]
    (run_dir / "candidates.csv").write_text(header + "\n" + "\n".join(rows) + "\n")

    round_metrics = {
        "warm_start": {
            "round_id": 0,
            "n_initial": 2,
            "evaluation_wall_time": 0.2,
            "total_round_wall_time": 0.25,
        },
        "rounds": [
            {
                "round_id": 1,
                "best_y_so_far": -1.0,
                "surrogate_a_name": "gaussian_process",
                "surrogate_b_name": "random_forest",
                "surrogate_a_fit_time": 0.05,
                "surrogate_b_fit_time": 0.07,
                "surrogate_a_proposal_time": 0.01,
                "surrogate_b_proposal_time": 0.02,
                "surrogate_a_candidates_contributed": 1,
                "surrogate_b_candidates_contributed": 1,
                "evaluation_wall_time": 0.12,
                "total_round_wall_time": 0.3,
            }
        ],
    }
    (run_dir / "round_metrics.json").write_text(json.dumps(round_metrics))

    metadata = {
        "run_name": "gaussian_process_random_forest_branin_seed0",
        "benchmark": "branin",
        "surrogate_a_name": "gaussian_process",
        "surrogate_b_name": "random_forest",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))


def test_generate_all_plots_creates_all_expected_files(tmp_path):
    _build_synthetic_run(tmp_path)

    outputs = generate_all_plots(tmp_path)

    assert [p.name for p in outputs] == list(EXPECTED_PLOTS)
    for name in EXPECTED_PLOTS:
        path = tmp_path / "plots" / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_generate_all_plots_fails_clearly_when_round_metrics_missing(tmp_path):
    _build_synthetic_run(tmp_path)
    (tmp_path / "round_metrics.json").unlink()

    with pytest.raises(FileNotFoundError):
        generate_all_plots(tmp_path)


def test_generate_all_plots_fails_clearly_with_zero_rounds(tmp_path):
    _build_synthetic_run(tmp_path)
    data = json.loads((tmp_path / "round_metrics.json").read_text())
    data["rounds"] = []
    (tmp_path / "round_metrics.json").write_text(json.dumps(data))

    with pytest.raises(ValueError):
        generate_all_plots(tmp_path)


def test_cli_end_to_end_against_a_real_experiment_run(tmp_path):
    run_result = subprocess.run(
        [
            sys.executable, str(RUN_EXPERIMENT_SCRIPT),
            "--benchmark", "branin",
            "--pair", "gp_rf",
            "--seed", "0",
            "--n-initial", "3",
            "--n-rounds", "1",
            "--batch-size", "2",
            "--workers", "1",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run_result.returncode == 0, run_result.stderr

    run_dir = tmp_path / "gaussian_process_random_forest_branin_seed0"

    plot_result = subprocess.run(
        [sys.executable, str(PLOT_RUN_SCRIPT), "--run-dir", str(run_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert plot_result.returncode == 0, plot_result.stderr

    for name in EXPECTED_PLOTS:
        path = run_dir / "plots" / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_cli_fails_clearly_when_candidates_csv_missing(tmp_path):
    run_dir = tmp_path / "incomplete_run"
    run_dir.mkdir()
    _build_synthetic_run(run_dir)
    (run_dir / "candidates.csv").unlink()

    result = subprocess.run(
        [sys.executable, str(PLOT_RUN_SCRIPT), "--run-dir", str(run_dir)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "candidates.csv" in result.stderr
