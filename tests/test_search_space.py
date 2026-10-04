import numpy as np
import pytest

from search_space import SearchSpace


def test_sample_shape():
    space = SearchSpace(bounds=[(0.0, 1.0), (-5.0, 5.0), (10.0, 20.0)])
    rng = np.random.default_rng(0)

    samples = space.sample(7, rng)

    assert samples.shape == (7, 3)
    assert space.dim == 3


def test_samples_within_bounds():
    bounds = [(-3.0, 3.0), (0.0, 100.0)]
    space = SearchSpace(bounds=bounds)
    rng = np.random.default_rng(1)

    samples = space.sample(500, rng)

    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    assert np.all(samples >= low)
    assert np.all(samples <= high)


def test_identical_seed_produces_identical_samples():
    space = SearchSpace(bounds=[(0.0, 1.0), (0.0, 1.0)])

    samples_a = space.sample(10, np.random.default_rng(42))
    samples_b = space.sample(10, np.random.default_rng(42))

    np.testing.assert_array_equal(samples_a, samples_b)


def test_different_seeds_produce_different_samples():
    space = SearchSpace(bounds=[(0.0, 1.0), (0.0, 1.0)])

    samples_a = space.sample(10, np.random.default_rng(1))
    samples_b = space.sample(10, np.random.default_rng(2))

    assert not np.array_equal(samples_a, samples_b)


@pytest.mark.parametrize(
    "bad_bounds",
    [
        [],
        [(1.0, 1.0)],
        [(1.0, 0.0)],
        [(0.0, 1.0), (2.0,)],
    ],
)
def test_invalid_bounds_raise_value_error(bad_bounds):
    with pytest.raises(ValueError):
        SearchSpace(bounds=bad_bounds)
