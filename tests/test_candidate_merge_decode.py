import numpy as np
import pytest

from candidate_merge import merge_candidates
from parameter_space import ChoiceDimension, ParameterSpace

SPACE = ParameterSpace(
    [ChoiceDimension("C", [0.01, 0.1, 1, 10]), ChoiceDimension("penalty", ["l1", "l2"])]
)


def test_decode_fn_none_preserves_raw_numeric_dedup_behavior():
    # Same as a plain continuous dedup scenario: decode_fn is irrelevant
    # when None, and no decoded_params key is ever added.
    proposals = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "B": np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),  # rank 0 duplicates A's rank 1
    }
    batch = merge_candidates(proposals, {"A": 2, "B": 2})
    values = [tuple(e["candidate"]) for e in batch]
    assert len(set(values)) == len(values)
    assert "decoded_params" not in batch[0]


def test_decode_fn_dedups_numerically_different_vectors_that_decode_the_same():
    # [0.05, 0.1] and [0.2, 0.3] are numerically distinct but both decode
    # to {"C": 0.01, "penalty": "l1"} -- the exact scenario that raw
    # numeric dedup would miss.
    proposals = {
        "A": np.array([[0.05, 0.1], [0.9, 0.9]]),  # -> (C=0.01,l1), (C=10,l2)
        "B": np.array([[0.2, 0.3], [0.6, 0.1], [0.7, 0.6]]),  # -> (C=0.01,l1)[dup], (C=1,l1), (C=1,l2)
    }
    allocation = {"A": 2, "B": 2}

    batch = merge_candidates(proposals, allocation, decode_fn=SPACE.decode)

    decoded_configs = [tuple(sorted(e["decoded_params"].items())) for e in batch]
    assert len(set(decoded_configs)) == len(decoded_configs), "duplicate decoded config slipped through"

    b_entries = [e for e in batch if e["surrogate_source"] == "B"]
    assert len(b_entries) == 2
    # B's rank-0 candidate was a decoded duplicate of A's rank-0 and was
    # skipped -- the first placed B entry must be rank 1, not rank 0.
    assert b_entries[0]["candidate_rank_within_source"] == 1


def test_decode_fn_batch_entries_carry_decoded_params():
    proposals = {"A": np.array([[0.05, 0.1]]), "B": np.array([[0.6, 0.1]])}
    batch = merge_candidates(proposals, {"A": 1, "B": 1}, decode_fn=SPACE.decode)

    a_entry = next(e for e in batch if e["surrogate_source"] == "A")
    assert a_entry["decoded_params"] == {"C": 0.01, "penalty": "l1"}


def test_decode_fn_insufficient_unique_configs_raises():
    # Both of A's candidates decode to the SAME configuration, so A
    # cannot fill an allocation of 2 unique decoded configs.
    proposals = {
        "A": np.array([[0.05, 0.1], [0.06, 0.05]]),
        "B": np.array([[0.6, 0.1], [0.7, 0.2]]),
    }
    with pytest.raises(ValueError):
        merge_candidates(proposals, {"A": 2, "B": 2}, decode_fn=SPACE.decode)
