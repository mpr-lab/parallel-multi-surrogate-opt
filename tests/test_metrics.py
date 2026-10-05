import csv
import json

from metrics import MetricsLog


def _sample_round_kwargs(round_id=1):
    return dict(
        round_id=round_id,
        best_y_so_far=12.5,
        surrogate_a_name="random_forest",
        surrogate_b_name="gaussian_process",
        surrogate_a_fit_time=0.01,
        surrogate_b_fit_time=0.02,
        surrogate_a_proposal_time=0.03,
        surrogate_b_proposal_time=0.04,
        surrogate_a_candidates_contributed=5,
        surrogate_b_candidates_contributed=5,
        evaluation_wall_time=0.05,
        total_round_wall_time=0.15,
    )


def test_record_round_schema():
    log = MetricsLog()
    log.record_round(**_sample_round_kwargs())

    assert len(log.rounds) == 1
    record = log.rounds[0]
    assert set(record.keys()) == {
        "round_id",
        "best_y_so_far",
        "surrogate_a_name",
        "surrogate_b_name",
        "surrogate_a_fit_time",
        "surrogate_b_fit_time",
        "surrogate_a_proposal_time",
        "surrogate_b_proposal_time",
        "surrogate_a_candidates_contributed",
        "surrogate_b_candidates_contributed",
        "evaluation_wall_time",
        "total_round_wall_time",
    }


def test_record_warm_start_schema():
    log = MetricsLog()
    log.record_warm_start(n_initial=10, evaluation_wall_time=0.02, total_round_wall_time=0.03)

    assert log.warm_start == {
        "round_id": 0,
        "n_initial": 10,
        "evaluation_wall_time": 0.02,
        "total_round_wall_time": 0.03,
    }


def test_field_names_are_surrogate_agnostic():
    log = MetricsLog()
    log.record_round(**_sample_round_kwargs())

    for key in log.rounds[0]:
        assert "rf_" not in key and "gp_" not in key
        assert "random_forest" not in key and "gaussian_process" not in key


def test_to_json_round_trips(tmp_path):
    log = MetricsLog()
    log.record_warm_start(n_initial=4, evaluation_wall_time=0.01, total_round_wall_time=0.02)
    log.record_round(**_sample_round_kwargs(round_id=1))
    log.record_round(**_sample_round_kwargs(round_id=2))

    path = tmp_path / "metrics.json"
    log.to_json(str(path))

    with open(path) as f:
        loaded = json.load(f)

    assert loaded["warm_start"]["n_initial"] == 4
    assert len(loaded["rounds"]) == 2
    assert loaded["rounds"][1]["round_id"] == 2


def test_to_csv_round_trips(tmp_path):
    log = MetricsLog()
    log.record_round(**_sample_round_kwargs(round_id=1))
    log.record_round(**_sample_round_kwargs(round_id=2))

    path = tmp_path / "metrics.csv"
    log.to_csv(str(path))

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["round_id"] == "1"
    assert rows[1]["surrogate_a_name"] == "random_forest"


def test_to_csv_without_rounds_raises():
    log = MetricsLog()
    try:
        log.to_csv("unused.csv")
        assert False, "expected ValueError"
    except ValueError:
        pass
