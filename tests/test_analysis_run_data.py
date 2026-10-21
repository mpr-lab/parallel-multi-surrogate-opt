import json

import pytest

from run_data import load_candidates, load_metadata, load_round_metrics

CANDIDATES_HEADER = (
    "x_0,x_1,y,round_id,candidate_id,surrogate_source,"
    "candidate_rank_within_source,eval_start_time,eval_end_time,"
    "eval_duration,seed,improved_global_best"
)
CANDIDATES_ROWS = [
    "1.0,2.0,-5.0,0,r0_c0,random,0,10.0,10.1,0.1,42,True",
    "3.0,4.0,-2.0,1,r1_c0,gp,0,10.2,10.3,0.1,43,True",
]


def _write_candidates_csv(run_dir, header=CANDIDATES_HEADER, rows=CANDIDATES_ROWS):
    (run_dir / "candidates.csv").write_text(header + "\n" + "\n".join(rows) + "\n")


def _write_round_metrics(run_dir, data=None):
    if data is None:
        data = {"warm_start": {"round_id": 0, "n_initial": 1}, "rounds": []}
    (run_dir / "round_metrics.json").write_text(json.dumps(data))


def _write_metadata(run_dir, data=None):
    if data is None:
        data = {
            "run_name": "gp_rf_branin_seed0",
            "benchmark": "branin",
            "surrogate_a_name": "gaussian_process",
            "surrogate_b_name": "random_forest",
        }
    (run_dir / "metadata.json").write_text(json.dumps(data))


def test_load_candidates_parses_types_correctly(tmp_path):
    _write_candidates_csv(tmp_path)

    records = load_candidates(tmp_path)

    assert len(records) == 2
    first = records[0]
    assert first["x"] == (1.0, 2.0)
    assert first["y"] == -5.0
    assert first["round_id"] == 0
    assert first["candidate_id"] == "r0_c0"
    assert first["surrogate_source"] == "random"
    assert first["candidate_rank_within_source"] == 0
    assert first["eval_start_time"] == 10.0
    assert first["eval_end_time"] == 10.1
    assert first["eval_duration"] == 0.1
    assert first["seed"] == 42
    assert first["improved_global_best"] is True


def test_load_candidates_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candidates(tmp_path)


def test_load_candidates_missing_required_column_raises(tmp_path):
    header = CANDIDATES_HEADER.replace("improved_global_best", "extra_unrelated_field")
    _write_candidates_csv(tmp_path, header=header, rows=["1.0,2.0,-5.0,0,r0_c0,random,0,10.0,10.1,0.1,42,True"])

    with pytest.raises(ValueError):
        load_candidates(tmp_path)


def test_load_round_metrics_parses(tmp_path):
    data = {
        "warm_start": {"round_id": 0, "n_initial": 4},
        "rounds": [{"round_id": 1, "surrogate_a_fit_time": 0.1}],
    }
    _write_round_metrics(tmp_path, data)

    loaded = load_round_metrics(tmp_path)

    assert loaded == data


def test_load_round_metrics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_round_metrics(tmp_path)


def test_load_round_metrics_missing_rounds_key_raises(tmp_path):
    (tmp_path / "round_metrics.json").write_text(json.dumps({"warm_start": {}}))

    with pytest.raises(ValueError):
        load_round_metrics(tmp_path)


def test_load_metadata_parses(tmp_path):
    _write_metadata(tmp_path)

    metadata = load_metadata(tmp_path)

    assert metadata["run_name"] == "gp_rf_branin_seed0"
    assert metadata["surrogate_a_name"] == "gaussian_process"


def test_load_metadata_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path)


def test_load_metadata_missing_required_field_raises(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"run_name": "x", "benchmark": "branin"}))

    with pytest.raises(ValueError):
        load_metadata(tmp_path)
