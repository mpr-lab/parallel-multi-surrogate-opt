import numpy as np
import pytest

from datastore import DataStore


def make_record(x, y, round_id=0, candidate_id=0, surrogate_source="surrogate_a",
                 rank=0, eval_duration=0.1, improved_global_best=False, **extra):
    record = {
        "x": x,
        "y": y,
        "round_id": round_id,
        "candidate_id": candidate_id,
        "surrogate_source": surrogate_source,
        "candidate_rank_within_source": rank,
        "eval_duration": eval_duration,
        "improved_global_best": improved_global_best,
    }
    record.update(extra)
    return record


def test_empty_store_behavior():
    store = DataStore()

    assert len(store) == 0
    assert store.best_y is None

    X, y = store.get_X_y()
    assert X.shape == (0, 0)
    assert y.shape == (0,)


def test_single_append():
    store = DataStore()
    record = make_record(x=[1.0, 2.0], y=0.5, improved_global_best=True)

    store.append(record)

    assert len(store) == 1
    X, y = store.get_X_y()
    assert X.shape == (1, 2)
    assert y.shape == (1,)
    np.testing.assert_array_equal(X[0], [1.0, 2.0])
    assert y[0] == 0.5
    assert store.best_y == 0.5


def test_batch_append_preserves_order():
    store = DataStore()
    records = [
        make_record(x=[0.0, 0.0], y=0.1, candidate_id=i)
        for i in range(3)
    ]
    records[1]["x"] = [1.0, 1.0]
    records[2]["x"] = [2.0, 2.0]

    store.append_batch(records)

    assert len(store) == 3
    X, y = store.get_X_y()
    np.testing.assert_array_equal(X, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])


def test_get_all_returns_copies_of_stored_records():
    store = DataStore()
    store.append(make_record(x=[1.0], y=2.0, improved_global_best=True))

    all_records = store.get_all()
    all_records[0]["y"] = 999.0  # mutating the copy must not affect the store

    assert len(all_records) == 1
    assert all_records[0]["improved_global_best"] is True
    assert store.best_y == 2.0


def test_get_X_y_shape_and_content():
    store = DataStore()
    store.append_batch(
        [
            make_record(x=np.array([1.0, -1.0, 3.0]), y=1.0),
            make_record(x=np.array([4.0, -2.0, 6.0]), y=2.0),
        ]
    )

    X, y = store.get_X_y()

    assert X.shape == (2, 3)
    assert X.dtype == np.float64
    assert y.shape == (2,)
    assert y.dtype == np.float64
    np.testing.assert_array_equal(X, [[1.0, -1.0, 3.0], [4.0, -2.0, 6.0]])
    np.testing.assert_array_equal(y, [1.0, 2.0])


def test_best_y_follows_maximize_convention():
    store = DataStore()

    store.append(make_record(x=[0.0], y=-5.0))
    assert store.best_y == -5.0

    store.append(make_record(x=[0.0], y=10.0))
    assert store.best_y == 10.0

    # A later, worse value must not lower best_y.
    store.append(make_record(x=[0.0], y=-100.0))
    assert store.best_y == 10.0


def test_extra_metadata_does_not_break_storage():
    store = DataStore()
    record = make_record(
        x=[1.0, 2.0],
        y=3.0,
        seed=42,
        eval_start_time=100.0,
        eval_end_time=100.1,
        note="some experiment-specific annotation",
    )

    store.append(record)

    assert len(store) == 1
    X, y = store.get_X_y()
    assert X.shape == (1, 2)
    assert store.best_y == 3.0


def test_append_rejects_record_missing_required_fields():
    store = DataStore()
    incomplete = {"x": [1.0], "y": 1.0}

    with pytest.raises(ValueError):
        store.append(incomplete)


def test_append_rejects_mismatched_dimension():
    store = DataStore()
    store.append(make_record(x=[1.0, 2.0], y=1.0))

    with pytest.raises(ValueError):
        store.append(make_record(x=[1.0, 2.0, 3.0], y=2.0))
