import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "run_experiment.py"


def test_branin_gp_rf_smoke(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
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

    assert result.returncode == 0, result.stderr
    assert "run: gaussian_process_random_forest_branin_seed0" in result.stdout

    run_dir = tmp_path / "gaussian_process_random_forest_branin_seed0"
    assert (run_dir / "candidates.csv").exists()
    assert (run_dir / "round_metrics.json").exists()
    assert (run_dir / "metadata.json").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["benchmark"] == "branin"
    assert metadata["surrogate_pair"] == "gp_rf"
    assert metadata["seed"] == 0
    assert metadata["total_evaluations"] == 3 + 1 * 2  # n_initial + n_rounds * batch_size
    assert metadata["git_commit"] is None or isinstance(metadata["git_commit"], str)

    candidate_lines = (run_dir / "candidates.csv").read_text().strip().splitlines()
    assert len(candidate_lines) == 1 + 5  # header + (n_initial + n_rounds * batch_size) rows

    round_metrics = json.loads((run_dir / "round_metrics.json").read_text())
    assert round_metrics["warm_start"]["n_initial"] == 3
    assert len(round_metrics["rounds"]) == 1
    assert round_metrics["rounds"][0]["surrogate_a_name"] == "gaussian_process"
    assert round_metrics["rounds"][0]["surrogate_b_name"] == "random_forest"


def test_odd_batch_size_rejected(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--benchmark", "branin",
            "--pair", "gp_rf",
            "--seed", "0",
            "--n-initial", "2",
            "--n-rounds", "1",
            "--batch-size", "3",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
