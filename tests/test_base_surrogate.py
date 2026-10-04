import numpy as np
import pytest

from search_space import SearchSpace
from surrogates import BaseSurrogate


class _DummySurrogate(BaseSurrogate):
    """Minimal concrete surrogate used only to exercise the BaseSurrogate
    contract in tests. Ignores the fitted history entirely and proposes
    plain random samples from the search space -- not a real model."""

    def __init__(self):
        self.last_X = None
        self.last_y = None

    @property
    def name(self) -> str:
        return "dummy_random"

    def fit(self, X, y):
        self.last_X = X
        self.last_y = y

    def propose(self, search_space, n_candidates, rng):
        return search_space.sample(n_candidates, rng)


def test_base_surrogate_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseSurrogate()


def test_contract_is_exactly_name_fit_propose():
    # Locks in that `predict` is NOT required by the framework contract.
    assert BaseSurrogate.__abstractmethods__ == frozenset({"name", "fit", "propose"})


def test_incomplete_subclass_cannot_be_instantiated():
    class _MissingPropose(BaseSurrogate):
        @property
        def name(self):
            return "incomplete"

        def fit(self, X, y):
            pass

    with pytest.raises(TypeError):
        _MissingPropose()


def test_dummy_surrogate_name():
    surrogate = _DummySurrogate()
    assert surrogate.name == "dummy_random"


def test_dummy_surrogate_fit_accepts_history():
    surrogate = _DummySurrogate()
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.1, 0.2])

    surrogate.fit(X, y)  # should not raise

    np.testing.assert_array_equal(surrogate.last_X, X)
    np.testing.assert_array_equal(surrogate.last_y, y)


def test_dummy_surrogate_propose_shape_and_bounds():
    space = SearchSpace(bounds=[(0.0, 1.0), (-5.0, 5.0)])
    surrogate = _DummySurrogate()
    rng = np.random.default_rng(0)

    candidates = surrogate.propose(space, n_candidates=5, rng=rng)

    assert candidates.shape == (5, 2)
    assert np.all(candidates[:, 0] >= 0.0) and np.all(candidates[:, 0] <= 1.0)
    assert np.all(candidates[:, 1] >= -5.0) and np.all(candidates[:, 1] <= 5.0)


def test_dummy_surrogate_propose_is_reproducible_given_same_rng_seed():
    space = SearchSpace(bounds=[(0.0, 1.0)])
    surrogate = _DummySurrogate()

    candidates_a = surrogate.propose(space, 4, np.random.default_rng(7))
    candidates_b = surrogate.propose(space, 4, np.random.default_rng(7))

    np.testing.assert_array_equal(candidates_a, candidates_b)
