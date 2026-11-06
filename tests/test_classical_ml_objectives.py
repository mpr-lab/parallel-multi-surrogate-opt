import inspect
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import objectives.classical_ml as classical_ml
from objectives.classical_ml import (
    LOGISTIC_REGRESSION_EXPANDED_PARAMETER_SPACE,
    LOGISTIC_REGRESSION_PARAMETER_SPACE,
    RANDOM_FOREST_EXPANDED_PARAMETER_SPACE,
    RANDOM_FOREST_PARAMETER_SPACE,
    XGBOOST_EXPANDED_PARAMETER_SPACE,
    XGBOOST_PARAMETER_SPACE,
    make_logistic_regression_objective,
    make_random_forest_objective,
    make_xgboost_objective,
    prepare_classical_ml_data,
)

# Small, bundled/synthetic datasets only -- no network access.
FAST_DATASET = "synthetic_classification"


def test_logistic_regression_objective_returns_valid_accuracy():
    objective = make_logistic_regression_objective(dataset=FAST_DATASET, seed=0)
    y = objective({"C": 1, "penalty": "l2"})
    assert 0.0 <= y <= 1.0


def test_random_forest_objective_returns_valid_accuracy():
    objective = make_random_forest_objective(dataset=FAST_DATASET, seed=0)
    y = objective({"n_estimators": 50, "max_depth": 5})
    assert 0.0 <= y <= 1.0


# -- explicit single-threaded resource control (experiment-level fairness) --
# RandomForestClassifier/XGBClassifier must be constructed with n_jobs=1 so
# the WorkerPool's own worker count controls parallelism, not each
# library's own threading default (sklearn's n_jobs=None -> 1 core,
# XGBoost's n_jobs=None -> all cores -- confounding wall-clock comparisons).


def test_random_forest_is_constructed_with_n_jobs_1(monkeypatch):
    captured_kwargs = {}
    real_random_forest_classifier = classical_ml.RandomForestClassifier

    class RecordingRandomForestClassifier(real_random_forest_classifier):
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(classical_ml, "RandomForestClassifier", RecordingRandomForestClassifier)

    objective = make_random_forest_objective(dataset=FAST_DATASET, seed=0)
    y = objective({"n_estimators": 50, "max_depth": 5})

    assert captured_kwargs["n_jobs"] == 1
    assert 0.0 <= y <= 1.0


def test_xgboost_is_constructed_with_n_jobs_1(monkeypatch):
    # Stub xgboost entirely so this is independent of whether the real
    # package is installed/working in this environment.
    captured_kwargs = {}

    class _FakeXGBClassifier:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def fit(self, X, y):
            self._y = y

        def score(self, X, y):
            return 0.75

    fake_xgboost = type(sys)("xgboost")
    fake_xgboost.XGBClassifier = _FakeXGBClassifier
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)

    objective = make_xgboost_objective(dataset=FAST_DATASET, seed=0)
    y = objective({"eta": 0.1, "max_depth": 6})

    assert captured_kwargs["n_jobs"] == 1
    assert y == 0.75  # objective wiring (fit + score) still works as before


def test_random_forest_n_jobs_does_not_change_the_fitted_result():
    # n_jobs controls parallelism only -- for a fixed random_state, the
    # trees built (and therefore the accuracy) must be identical whether
    # n_jobs is 1 or sklearn's default (None). Confirms this change only
    # affects wall-clock timing, not experiment results.
    data = prepare_classical_ml_data(dataset=FAST_DATASET, seed=0)

    model_default = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0, n_jobs=None)
    model_explicit = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0, n_jobs=1)
    model_default.fit(data.X_train, data.y_train)
    model_explicit.fit(data.X_train, data.y_train)

    assert model_default.score(data.X_val, data.y_val) == model_explicit.score(data.X_val, data.y_val)


def test_logistic_regression_parameter_space_matches_spec():
    assert LOGISTIC_REGRESSION_PARAMETER_SPACE.decode(np.array([0.0, 0.0])) == {
        "C": 0.01,
        "penalty": "l1",
    }


def test_random_forest_parameter_space_matches_spec():
    choices = {d.name: list(d.choices) for d in RANDOM_FOREST_PARAMETER_SPACE.dimensions}
    assert choices == {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}


def test_xgboost_parameter_space_matches_spec():
    choices = {d.name: list(d.choices) for d in XGBOOST_PARAMETER_SPACE.dimensions}
    assert choices == {"eta": [0.01, 0.1, 0.2], "max_depth": [3, 6, 9]}


# -- expanded search spaces (15-evaluation protocol) ----------------------
# These must NOT affect the historical-grid spaces above -- verified by
# the three tests immediately above still asserting the original,
# unchanged 4/3/3-choice grids.


def test_logistic_regression_expanded_has_80_configurations():
    assert LOGISTIC_REGRESSION_EXPANDED_PARAMETER_SPACE.total_configurations == 80
    c_dimension = LOGISTIC_REGRESSION_EXPANDED_PARAMETER_SPACE.dimensions[0]
    penalty_dimension = LOGISTIC_REGRESSION_EXPANDED_PARAMETER_SPACE.dimensions[1]
    assert c_dimension.name == "C"
    assert len(c_dimension.choices) == 40
    assert c_dimension.choices[0] == pytest.approx(1e-4)
    assert c_dimension.choices[-1] == pytest.approx(1e3)
    assert list(penalty_dimension.choices) == ["l1", "l2"]


def test_random_forest_expanded_has_152_configurations():
    assert RANDOM_FOREST_EXPANDED_PARAMETER_SPACE.total_configurations == 152
    n_estimators_dimension = RANDOM_FOREST_EXPANDED_PARAMETER_SPACE.dimensions[0]
    max_depth_dimension = RANDOM_FOREST_EXPANDED_PARAMETER_SPACE.dimensions[1]
    assert list(n_estimators_dimension.choices) == [
        50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
    ]
    assert list(max_depth_dimension.choices) == [3, 5, 7, 10, 15, 20, 25, 30]


def test_xgboost_expanded_has_96_configurations():
    assert XGBOOST_EXPANDED_PARAMETER_SPACE.total_configurations == 96
    eta_dimension = XGBOOST_EXPANDED_PARAMETER_SPACE.dimensions[0]
    max_depth_dimension = XGBOOST_EXPANDED_PARAMETER_SPACE.dimensions[1]
    assert len(eta_dimension.choices) == 12
    assert eta_dimension.choices[0] == pytest.approx(0.01)
    assert eta_dimension.choices[-1] == pytest.approx(0.3)
    assert list(max_depth_dimension.choices) == [3, 4, 5, 6, 7, 8, 9, 10]


def test_expanded_spaces_decode_to_the_same_param_names_as_historical_counterparts():
    # This is exactly why the existing make_*_objective factories are
    # reusable unchanged for the expanded benchmarks.
    assert {d.name for d in LOGISTIC_REGRESSION_EXPANDED_PARAMETER_SPACE.dimensions} == {
        d.name for d in LOGISTIC_REGRESSION_PARAMETER_SPACE.dimensions
    }
    assert {d.name for d in RANDOM_FOREST_EXPANDED_PARAMETER_SPACE.dimensions} == {
        d.name for d in RANDOM_FOREST_PARAMETER_SPACE.dimensions
    }
    assert {d.name for d in XGBOOST_EXPANDED_PARAMETER_SPACE.dimensions} == {
        d.name for d in XGBOOST_PARAMETER_SPACE.dimensions
    }


def test_random_forest_objective_works_with_expanded_space_decoded_params():
    objective = make_random_forest_objective(dataset=FAST_DATASET, seed=0)
    y = objective({"n_estimators": 500, "max_depth": 30})  # only valid in the expanded space
    assert 0.0 <= y <= 1.0


def test_prepare_classical_ml_data_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        prepare_classical_ml_data(dataset="not_a_real_dataset", seed=0)


def test_deterministic_train_validation_split_given_same_seed():
    data_a = prepare_classical_ml_data(dataset=FAST_DATASET, seed=42)
    data_b = prepare_classical_ml_data(dataset=FAST_DATASET, seed=42)

    np.testing.assert_array_equal(data_a.X_train, data_b.X_train)
    np.testing.assert_array_equal(data_a.y_train, data_b.y_train)
    np.testing.assert_array_equal(data_a.X_val, data_b.X_val)
    np.testing.assert_array_equal(data_a.y_val, data_b.y_val)


def test_different_seed_changes_split():
    data_a = prepare_classical_ml_data(dataset=FAST_DATASET, seed=1)
    data_b = prepare_classical_ml_data(dataset=FAST_DATASET, seed=2)
    assert not np.array_equal(data_a.X_train, data_b.X_train)


def test_model_random_state_is_fixed_per_run_not_per_call_seed():
    # The per-call `seed` (what WorkerPool derives uniquely per candidate)
    # must NOT affect model-internal randomness -- only the hyperparameters
    # being searched should change the result for the same decoded params.
    objective = make_logistic_regression_objective(dataset=FAST_DATASET, seed=0)
    params = {"C": 1, "penalty": "l2"}
    assert objective(params, seed=111) == objective(params, seed=999)


def test_objective_accepts_seed_kwarg_for_worker_pool_compatibility():
    objective = make_logistic_regression_objective(dataset=FAST_DATASET, seed=0)
    assert "seed" in inspect.signature(objective).parameters


def test_xgboost_missing_dependency_raises_clear_import_error(monkeypatch):
    # Force the import to fail regardless of whether xgboost happens to be
    # installed/working in this particular environment.
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ImportError, match="xgboost"):
        make_xgboost_objective(dataset=FAST_DATASET, seed=0)
