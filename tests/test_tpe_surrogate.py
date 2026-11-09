import numpy as np
import pytest

from search_space import SearchSpace
from surrogates import BaseSurrogate, TPESurrogate


def _linear_training_data():
    # y increases with x, so "good" (high-y) observations cluster at high x.
    X = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]])
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    return X, y


def test_is_a_base_surrogate():
    assert isinstance(TPESurrogate(), BaseSurrogate)


def test_name():
    assert TPESurrogate().name == "tpe"


def test_invalid_gamma_raises():
    with pytest.raises(ValueError):
        TPESurrogate(gamma=0.0)
    with pytest.raises(ValueError):
        TPESurrogate(gamma=1.0)


def test_fit_accepts_shared_history_shapes():
    X, y = _linear_training_data()
    surrogate = TPESurrogate()

    surrogate.fit(X, y)  # should not raise

    assert surrogate._is_fitted


def test_fit_raises_on_insufficient_history():
    surrogate = TPESurrogate()

    with pytest.raises(ValueError):
        surrogate.fit(np.array([[1.0]]), np.array([1.0]))


def test_fit_minimal_history_of_two_succeeds():
    surrogate = TPESurrogate()
    X = np.array([[0.0], [10.0]])
    y = np.array([0.0, 1.0])

    surrogate.fit(X, y)  # should not raise

    assert surrogate._is_fitted
    assert surrogate._good_X.shape[0] >= 1
    assert surrogate._bad_X.shape[0] >= 1


def test_propose_before_fit_raises():
    space = SearchSpace(bounds=[(0.0, 1.0)])
    surrogate = TPESurrogate()

    with pytest.raises(RuntimeError):
        surrogate.propose(space, n_candidates=3, rng=np.random.default_rng(0))


def test_propose_output_shape_and_bounds():
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = TPESurrogate(pool_size=50)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=4, rng=np.random.default_rng(0))

    assert candidates.shape == (4, 1)
    assert np.all(candidates >= 0.0) and np.all(candidates <= 10.0)


def test_propose_contains_no_nan_or_inf():
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = TPESurrogate(pool_size=50)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=10, rng=np.random.default_rng(0))

    assert np.all(np.isfinite(candidates))


def test_propose_prefers_high_value_region():
    # Higher-is-better: the "good" group is the high-x/high-y tail, so
    # sampled-and-ranked candidates should cluster toward the high end.
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = TPESurrogate(pool_size=200)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=5, rng=np.random.default_rng(0))

    assert candidates.mean() > 5.0


def test_propose_ranks_candidates_by_descending_log_density_ratio():
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = TPESurrogate(pool_size=50)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=10, rng=np.random.default_rng(0))
    log_good = surrogate._log_density(candidates, surrogate._good_X, surrogate._good_bandwidth)
    log_bad = surrogate._log_density(candidates, surrogate._bad_X, surrogate._bad_bandwidth)
    score = log_good - log_bad

    assert np.all(np.diff(score) <= 1e-9)  # non-increasing: best-first
    assert np.all(np.isfinite(score))


def test_deterministic_given_same_seed_end_to_end():
    X, y = _linear_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])

    surrogate_a = TPESurrogate(pool_size=30)
    surrogate_b = TPESurrogate(pool_size=30)
    surrogate_a.fit(X, y)
    surrogate_b.fit(X, y)

    candidates_a = surrogate_a.propose(space, 6, np.random.default_rng(123))
    candidates_b = surrogate_b.propose(space, 6, np.random.default_rng(123))

    np.testing.assert_array_equal(candidates_a, candidates_b)


def test_higher_y_defines_the_good_group():
    # Flip the sign of y and confirm the good group flips to the other end.
    X = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]])
    y_increasing = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    y_decreasing = -y_increasing

    surrogate_inc = TPESurrogate(gamma=0.5)
    surrogate_inc.fit(X, y_increasing)
    surrogate_dec = TPESurrogate(gamma=0.5)
    surrogate_dec.fit(X, y_decreasing)

    assert surrogate_inc._good_X.mean() > surrogate_inc._bad_X.mean()
    assert surrogate_dec._good_X.mean() < surrogate_dec._bad_X.mean()
