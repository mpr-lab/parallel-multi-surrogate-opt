import sys

import numpy as np
import pytest

import objectives.classical_ml as classical_ml
from objectives.classical_ml import (
    make_logistic_regression_objective,
    make_random_forest_objective,
    make_xgboost_objective,
    prepare_classical_ml_data,
)


@pytest.fixture
def higgs_fixture_dir(tmp_path, monkeypatch):
    """A tiny, fully synthetic stand-in for HIGGS.csv -- never the real
    multi-GB dataset. Also shrinks HIGGS_SUBSAMPLE_SIZE so the (otherwise
    50,000-row) subsample logic runs against this tiny fixture."""
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 100)  # 200 rows, balanced for stratify=y
    features = rng.normal(size=(200, 28))
    fixture = np.column_stack([labels, features])
    np.savetxt(tmp_path / "HIGGS.csv", fixture, delimiter=",")

    monkeypatch.setattr(classical_ml, "HIGGS_SUBSAMPLE_SIZE", 50)
    return tmp_path


def test_higgs_loader_reads_and_subsamples_fixture(higgs_fixture_dir):
    X, y = classical_ml._load_higgs(seed=0, data_dir=higgs_fixture_dir)
    assert X.shape == (50, 28)
    assert y.shape == (50,)
    assert set(np.unique(y)) == {0.0, 1.0}


def test_higgs_loader_requires_data_dir():
    with pytest.raises(ValueError, match="data-dir"):
        classical_ml._load_higgs(seed=0, data_dir=None)


def test_higgs_loader_raises_clear_error_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="HIGGS.csv"):
        classical_ml._load_higgs(seed=0, data_dir=tmp_path)


def test_higgs_uses_historical_80_20_split(higgs_fixture_dir):
    data = prepare_classical_ml_data(dataset="higgs", seed=0, data_dir=higgs_fixture_dir)
    total = len(data.X_train) + len(data.X_val)
    assert len(data.X_val) / total == pytest.approx(0.2)


def test_smoke_test_datasets_keep_default_validation_fraction():
    data = prepare_classical_ml_data(dataset="synthetic_classification", seed=0)
    total = len(data.X_train) + len(data.X_val)
    assert len(data.X_val) / total == pytest.approx(0.25)


def test_higgs_split_is_deterministic_given_same_seed(higgs_fixture_dir):
    data_a = prepare_classical_ml_data(dataset="higgs", seed=7, data_dir=higgs_fixture_dir)
    data_b = prepare_classical_ml_data(dataset="higgs", seed=7, data_dir=higgs_fixture_dir)
    np.testing.assert_array_equal(data_a.X_train, data_b.X_train)
    np.testing.assert_array_equal(data_a.y_val, data_b.y_val)


def test_logistic_regression_objective_works_on_higgs_fixture(higgs_fixture_dir):
    objective = make_logistic_regression_objective(dataset="higgs", data_dir=higgs_fixture_dir, seed=0)
    y = objective({"C": 1, "penalty": "l2"})
    assert 0.0 <= y <= 1.0


def test_random_forest_objective_works_on_higgs_fixture(higgs_fixture_dir):
    objective = make_random_forest_objective(dataset="higgs", data_dir=higgs_fixture_dir, seed=0)
    y = objective({"n_estimators": 50, "max_depth": 5})
    assert 0.0 <= y <= 1.0


def test_logistic_regression_scales_higgs_features(higgs_fixture_dir, monkeypatch):
    calls = []

    class RecordingScaler:
        def fit_transform(self, X):
            calls.append("fit_transform")
            return X

        def transform(self, X):
            calls.append("transform")
            return X

    monkeypatch.setattr(classical_ml, "MinMaxScaler", RecordingScaler)

    make_logistic_regression_objective(dataset="higgs", data_dir=higgs_fixture_dir, seed=0)

    assert calls == ["fit_transform", "transform"]


def test_random_forest_does_not_scale_higgs_features(higgs_fixture_dir, monkeypatch):
    class ExplodingScaler:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MinMaxScaler must not be used for Random Forest")

    monkeypatch.setattr(classical_ml, "MinMaxScaler", ExplodingScaler)

    make_random_forest_objective(dataset="higgs", data_dir=higgs_fixture_dir, seed=0)  # must not raise


def test_xgboost_does_not_scale_higgs_features(higgs_fixture_dir, monkeypatch):
    class ExplodingScaler:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MinMaxScaler must not be used for XGBoost")

    monkeypatch.setattr(classical_ml, "MinMaxScaler", ExplodingScaler)

    # Stub out xgboost itself so this test is independent of whether the
    # real package is installed/working in this environment.
    class _FakeXGBClassifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            self._y = y

        def score(self, X, y):
            return float((y == y[0]).mean())  # arbitrary, just needs to run

    fake_xgboost = type(sys)("xgboost")
    fake_xgboost.XGBClassifier = _FakeXGBClassifier
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)

    make_xgboost_objective(dataset="higgs", data_dir=higgs_fixture_dir, seed=0)  # must not raise
