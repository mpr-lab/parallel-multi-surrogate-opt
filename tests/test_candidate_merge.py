import numpy as np
import pytest

from candidate_merge import merge_candidates


def _as_rows(points):
    return np.array(points, dtype=float)


def test_no_duplicate_merge():
    proposals = {
        "A": _as_rows([[1.0], [2.0], [3.0]]),
        "B": _as_rows([[4.0], [5.0], [6.0]]),
    }
    allocation = {"A": 2, "B": 2}

    batch = merge_candidates(proposals, allocation)

    assert len(batch) == 4
    assert [entry["candidate"][0] for entry in batch] == [1.0, 2.0, 4.0, 5.0]


def test_duplicates_across_surrogates_are_removed_and_refilled():
    # Mirrors the worked example: B's rank-1 candidate duplicates A's.
    proposals = {
        "A": _as_rows([[1.0], [2.0], [3.0]]),
        "B": _as_rows([[10.0], [2.0], [30.0]]),
    }
    allocation = {"A": 2, "B": 2}

    batch = merge_candidates(proposals, allocation)

    candidate_values = [entry["candidate"][0] for entry in batch]
    assert candidate_values == [1.0, 2.0, 10.0, 30.0]
    assert len(batch) == 4
    # No value appears twice in the final batch.
    assert len(set(candidate_values)) == len(candidate_values)


def test_source_labels_are_preserved():
    proposals = {
        "surrogate_a": _as_rows([[1.0], [2.0]]),
        "surrogate_b": _as_rows([[3.0], [4.0]]),
    }
    allocation = {"surrogate_a": 1, "surrogate_b": 1}

    batch = merge_candidates(proposals, allocation)

    sources = {entry["surrogate_source"] for entry in batch}
    assert sources == {"surrogate_a", "surrogate_b"}


def test_within_source_rank_is_preserved_even_after_a_skip():
    proposals = {
        "A": _as_rows([[1.0], [2.0], [3.0]]),
        "B": _as_rows([[10.0], [2.0], [30.0]]),  # rank-1 duplicates A's output
    }
    allocation = {"A": 2, "B": 2}

    batch = merge_candidates(proposals, allocation)

    b_entries = [e for e in batch if e["surrogate_source"] == "B"]
    ranks = [e["candidate_rank_within_source"] for e in b_entries]
    # B's second placed candidate is rank 2 (its duplicate rank-1 was
    # skipped), not rank 1 -- rank reflects original position, not the
    # order it was placed into the batch.
    assert ranks == [0, 2]


def test_refill_behavior_is_deterministic():
    proposals = {
        "A": _as_rows([[1.0], [2.0], [3.0]]),
        "B": _as_rows([[10.0], [2.0], [30.0]]),
    }
    allocation = {"A": 2, "B": 2}

    batch_1 = merge_candidates(proposals, allocation)
    batch_2 = merge_candidates(proposals, allocation)

    simplify = lambda batch: [
        (e["surrogate_source"], e["candidate_rank_within_source"], tuple(e["candidate"]))
        for e in batch
    ]
    assert simplify(batch_1) == simplify(batch_2)


def test_insufficient_unique_candidates_raises_value_error():
    # A's list collapses to a single unique value once duplicates within
    # its own list are removed, so its allocation of 2 cannot be filled.
    proposals = {
        "A": _as_rows([[1.0], [1.0], [1.0]]),
        "B": _as_rows([[2.0], [3.0]]),
    }
    allocation = {"A": 2, "B": 2}

    with pytest.raises(ValueError):
        merge_candidates(proposals, allocation)


def test_mismatched_source_keys_raise_value_error():
    proposals = {"A": _as_rows([[1.0]])}
    allocation = {"B": 1}

    with pytest.raises(ValueError):
        merge_candidates(proposals, allocation)
