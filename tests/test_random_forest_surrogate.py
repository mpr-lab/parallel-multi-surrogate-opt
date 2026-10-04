import numpy as np
import pytest

from search_space import SearchSpace
from surrogates import BaseSurrogate, RandomForestSurrogate


def _linear_training_data():
    # y increases with x, so a fitted model should prefer high-x candidates.
    X = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]])
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    return X, y


def test_is_a_base_surrogate():
    surrogate = RandomForestSurrogate()
    assert isinstance(surrogate, BaseSurrogate)


def test_name():
    assert RandomForestSurrogate().name == "random_forest"


def test_n_jobs_is_fixed_to_one():
    surrogate = RandomForestSurrogate(n_estimators=5)
    assert surrogate._model.n_jobs == 1


def test_propose_before_fit_raises():
    space = SearchSpace(bounds=[(0.0, 1.0)])
    surrogate = RandomForestSurrogate()

    with pytest.raises(RuntimeError):
        surrogate.propose(space, n_candidates=3, rng=np.random.default_rng(0))


def test_fit_accepts_shared_history_shapes():
    X, y = _linear_training_data()
    surrogate = RandomForestSurrogate(n_estimators=10, random_state=0)

    surrogate.fit(X, y)  # should not raise

    assert surrogate._is_fitted


def test_propose_output_shape_and_bounds():
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = RandomForestSurrogate(n_estimators=10, pool_size=50, random_state=0)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=4, rng=np.random.default_rng(0))

    assert candidates.shape == (4, 1)
    assert np.all(candidates >= 0.0) and np.all(candidates <= 10.0)


def test_propose_ranks_candidates_by_descending_predicted_score():
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = RandomForestSurrogate(n_estimators=20, pool_size=50, random_state=0)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=10, rng=np.random.default_rng(0))
    predicted = surrogate._model.predict(candidates)

    assert np.all(np.diff(predicted) <= 1e-9)  # non-increasing: best-first


def test_propose_prefers_high_value_region():
    # Since y increases with x, the top-ranked proposals should cluster
    # toward the high end of the domain rather than being spread uniformly.
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = RandomForestSurrogate(n_estimators=20, pool_size=200, random_state=0)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=5, rng=np.random.default_rng(0))

    assert candidates.mean() > 5.0


def test_deterministic_given_same_seed_end_to_end():
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])

    surrogate_a = RandomForestSurrogate(n_estimators=15, pool_size=30, random_state=42)
    surrogate_b = RandomForestSurrogate(n_estimators=15, pool_size=30, random_state=42)
    surrogate_a.fit(X, y)
    surrogate_b.fit(X, y)

    candidates_a = surrogate_a.propose(space, 6, np.random.default_rng(123))
    candidates_b = surrogate_b.propose(space, 6, np.random.default_rng(123))

    np.testing.assert_array_equal(candidates_a, candidates_b)
