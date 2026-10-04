import math

import numpy as np
import pytest

from objectives import branin, hartmann6, rastrigin

# (objective_fn, a valid x, expected dim or None if dimension-generic)
BENCHMARKS = [
    (branin, np.array([-3.0, 12.0]), 2),
    (hartmann6, np.array([0.2, 0.15, 0.47, 0.27, 0.31, 0.65]), 6),
    (rastrigin, np.array([1.0, -2.0, 3.0]), None),
]


@pytest.mark.parametrize("fn, x, _dim", BENCHMARKS)
def test_deterministic_output(fn, x, _dim):
    y1 = fn(x)
    y2 = fn(x)
    assert y1 == y2


@pytest.mark.parametrize("fn, x, _dim", BENCHMARKS)
def test_output_is_numeric(fn, x, _dim):
    rng = np.random.default_rng(0)
    for _ in range(20):
        sample = x + rng.normal(scale=0.1, size=x.shape)
        y = fn(sample)
        assert isinstance(y, float)
        assert math.isfinite(y)


def test_branin_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        branin(np.array([1.0, 2.0, 3.0]))


def test_hartmann6_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        hartmann6(np.array([0.1, 0.2, 0.3]))


def test_rastrigin_rejects_empty_input():
    with pytest.raises(ValueError):
        rastrigin(np.array([]))


def test_rastrigin_accepts_any_positive_dimension():
    rng = np.random.default_rng(0)
    for d in (1, 3, 6, 10):
        y = rastrigin(rng.uniform(-5.12, 5.12, size=d))
        assert math.isfinite(y)


def test_branin_known_global_minimum():
    # Global minimum of the native (minimize) Branin surface is ~0.397887,
    # so under the maximize convention y should be ~ -0.397887, and this
    # point should be at or near the maximum over the domain.
    x_star = np.array([-math.pi, 12.275])
    y = branin(x_star)
    assert y == pytest.approx(-0.397887, abs=1e-3)


def test_hartmann6_known_global_minimum():
    # Global minimum of the native (minimize) Hartmann6 surface is
    # ~ -3.32237, so under the maximize convention y should be ~3.32237.
    x_star = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    y = hartmann6(x_star)
    assert y == pytest.approx(3.32237, abs=1e-3)


def test_rastrigin_known_global_minimum():
    # Global minimum of the native (minimize) Rastrigin surface is exactly 0
    # at the origin, for any dimensionality.
    y = rastrigin(np.zeros(5))
    assert y == pytest.approx(0.0, abs=1e-9)
