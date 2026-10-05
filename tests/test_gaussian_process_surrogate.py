import numpy as np
import pytest

from search_space import SearchSpace
from surrogates import BaseSurrogate, GaussianProcessSurrogate


def _small_training_data():
    X = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]])
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    return X, y


def test_is_a_base_surrogate():
    surrogate = GaussianProcessSurrogate()
    assert isinstance(surrogate, BaseSurrogate)


def test_name():
    assert GaussianProcessSurrogate().name == "gaussian_process"


def test_propose_before_fit_raises():
    space = SearchSpace(bounds=[(0.0, 1.0)])
    surrogate = GaussianProcessSurrogate()

    with pytest.raises(RuntimeError):
        surrogate.propose(space, n_candidates=3, rng=np.random.default_rng(0))


def test_fit_succeeds_on_small_dataset():
    X, y = _small_training_data()
    surrogate = GaussianProcessSurrogate(random_state=0)

    surrogate.fit(X, y)  # should not raise

    assert surrogate._is_fitted


def test_propose_output_shape():
    X, y = _small_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = GaussianProcessSurrogate(pool_size=50, random_state=0)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=4, rng=np.random.default_rng(0))

    assert candidates.shape == (4, 1)


def test_proposed_points_remain_within_bounds():
    X, y = _small_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = GaussianProcessSurrogate(pool_size=50, random_state=0)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=8, rng=np.random.default_rng(0))

    assert np.all(candidates >= 0.0) and np.all(candidates <= 10.0)


def test_deterministic_given_same_seed_and_data():
    X, y = _small_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])

    surrogate_a = GaussianProcessSurrogate(pool_size=30, random_state=42)
    surrogate_b = GaussianProcessSurrogate(pool_size=30, random_state=42)
    surrogate_a.fit(X, y)
    surrogate_b.fit(X, y)

    candidates_a = surrogate_a.propose(space, 6, np.random.default_rng(123))
    candidates_b = surrogate_b.propose(space, 6, np.random.default_rng(123))

    np.testing.assert_array_equal(candidates_a, candidates_b)


def test_propose_ranks_candidates_by_descending_predicted_score():
    X, y = _small_training_data()
    space = SearchSpace(bounds=[(0.0, 10.0)])
    surrogate = GaussianProcessSurrogate(pool_size=50, random_state=0)
    surrogate.fit(X, y)

    candidates = surrogate.propose(space, n_candidates=10, rng=np.random.default_rng(0))
    predicted = surrogate._model.predict(candidates)

    assert np.all(np.diff(predicted) <= 1e-9)  # non-increasing: best-first
