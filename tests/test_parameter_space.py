import numpy as np
import pytest

from parameter_space import ChoiceDimension, ParameterSpace


def test_choice_dimension_decode_maps_to_correct_bucket():
    dim = ChoiceDimension("x", [10, 20, 30, 40])
    assert dim.decode(0.0) == 10
    assert dim.decode(0.24) == 10
    assert dim.decode(0.25) == 20
    assert dim.decode(0.49) == 20
    assert dim.decode(0.50) == 30
    assert dim.decode(0.99) == 40


def test_choice_dimension_decode_clamps_at_exact_upper_boundary():
    # raw_value == 1.0 computes index == n (out of range) before clamping.
    dim = ChoiceDimension("x", ["a", "b"])
    assert dim.decode(1.0) == "b"
    assert dim.decode(0.999999) == "b"


def test_choice_dimension_encode_decode_round_trip_for_every_choice():
    dim = ChoiceDimension("penalty", ["l1", "l2"])
    for choice in dim.choices:
        assert dim.decode(dim.encode(choice)) == choice


def test_choice_dimension_encode_rejects_invalid_choice():
    dim = ChoiceDimension("penalty", ["l1", "l2"])
    with pytest.raises(ValueError):
        dim.encode("elasticnet")


def test_choice_dimension_rejects_empty_choices():
    with pytest.raises(ValueError):
        ChoiceDimension("x", [])


def test_parameter_space_bounds_are_unit_interval_per_dimension():
    space = ParameterSpace([ChoiceDimension("a", [1, 2, 3]), ChoiceDimension("b", ["x", "y"])])
    assert space.bounds == [(0.0, 1.0), (0.0, 1.0)]
    assert space.dim == 2


def test_parameter_space_decode_maps_full_vector():
    space = ParameterSpace(
        [ChoiceDimension("C", [0.01, 0.1, 1, 10]), ChoiceDimension("penalty", ["l1", "l2"])]
    )
    assert space.decode(np.array([0.1, 0.9])) == {"C": 0.01, "penalty": "l2"}


def test_parameter_space_encode_decode_round_trip():
    space = ParameterSpace(
        [
            ChoiceDimension("n_estimators", [50, 100, 200]),
            ChoiceDimension("max_depth", [5, 10, 20]),
        ]
    )
    params = {"n_estimators": 100, "max_depth": 20}
    assert space.decode(space.encode(params)) == params


def test_parameter_space_decode_rejects_wrong_length_vector():
    space = ParameterSpace([ChoiceDimension("a", [1, 2])])
    with pytest.raises(ValueError):
        space.decode(np.array([0.1, 0.2]))


def test_parameter_space_encode_rejects_missing_parameter():
    space = ParameterSpace([ChoiceDimension("a", [1, 2]), ChoiceDimension("b", [3, 4])])
    with pytest.raises(ValueError):
        space.encode({"a": 1})


def test_parameter_space_rejects_duplicate_dimension_names():
    with pytest.raises(ValueError):
        ParameterSpace([ChoiceDimension("a", [1, 2]), ChoiceDimension("a", [3, 4])])


def test_parameter_space_rejects_empty_dimension_list():
    with pytest.raises(ValueError):
        ParameterSpace([])
