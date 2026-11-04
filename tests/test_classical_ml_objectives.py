import inspect
import sys

import numpy as np
import pytest

from objectives.classical_ml import (
    LOGISTIC_REGRESSION_PARAMETER_SPACE,
    RANDOM_FOREST_PARAMETER_SPACE,
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
