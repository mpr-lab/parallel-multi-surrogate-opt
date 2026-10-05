import time

import numpy as np
import pytest

from worker_pool import WorkerPool


def _batch(values):
    return [
        {
            "candidate": np.array([v]),
            "surrogate_source": "A" if i % 2 == 0 else "B",
            "candidate_rank_within_source": i,
        }
        for i, v in enumerate(values)
    ]


def test_worker_count_one_is_correct():
    pool = WorkerPool(n_workers=1)
    results = pool.evaluate(_batch([1.0, 2.0, 3.0]), lambda x: float(x[0]) * 10, round_id=0)

    assert [r["y"] for r in results] == [10.0, 20.0, 30.0]


def test_multiple_workers_are_correct_and_run_concurrently():
    def slow_objective(x):
        time.sleep(0.05)
        return float(x[0])

    batch = _batch([1.0, 2.0, 3.0, 4.0])
    pool = WorkerPool(n_workers=4)

    start = time.perf_counter()
    results = pool.evaluate(batch, slow_objective, round_id=0)
    wall_time = time.perf_counter() - start

    assert [r["y"] for r in results] == [1.0, 2.0, 3.0, 4.0]
    # Sequential would take ~0.2s; with 4 workers it should be close to 0.05s.
    assert wall_time < 0.15


def test_result_ordering_matches_input_even_when_completion_order_differs():
    # Earlier-indexed candidates sleep longer, so later ones finish first.
    def variable_delay_objective(x):
        delay = 0.06 - 0.01 * x[0]
        time.sleep(max(delay, 0.0))
        return float(x[0])

    batch = _batch([0.0, 1.0, 2.0, 3.0])
    pool = WorkerPool(n_workers=4)

    results = pool.evaluate(batch, variable_delay_objective, round_id=0)

    assert [r["x"][0] for r in results] == [0.0, 1.0, 2.0, 3.0]


def test_result_metadata_is_complete_and_consistent():
    batch = _batch([5.0])
    batch[0]["surrogate_source"] = "surrogate_a"
    batch[0]["candidate_rank_within_source"] = 3
    pool = WorkerPool(n_workers=1)

    [result] = pool.evaluate(batch, lambda x: float(x[0]), round_id=7)

    assert result["x"][0] == 5.0
    assert result["y"] == 5.0
    assert result["round_id"] == 7
    assert result["candidate_id"] == "r7_c0"
    assert result["surrogate_source"] == "surrogate_a"
    assert result["candidate_rank_within_source"] == 3
    assert isinstance(result["eval_start_time"], float)
    assert isinstance(result["eval_end_time"], float)
    assert result["eval_end_time"] >= result["eval_start_time"]
    assert result["eval_duration"] == pytest.approx(
        result["eval_end_time"] - result["eval_start_time"]
    )


def test_objective_values_are_correct():
    batch = _batch([1.0, -2.0, 3.5])
    pool = WorkerPool(n_workers=2)

    results = pool.evaluate(batch, lambda x: float(x[0]) ** 2, round_id=0)

    assert [r["y"] for r in results] == [1.0, 4.0, 12.25]


def test_seeds_are_none_when_no_experiment_seed_given():
    seen = []

    def objective(x, seed=None):
        seen.append(seed)
        return 0.0

    pool = WorkerPool(n_workers=1)
    pool.evaluate(_batch([1.0, 2.0]), objective, round_id=0)

    assert seen == [None, None]


def test_seeds_are_deterministic_given_same_experiment_seed_round_and_index():
    seen_a, seen_b = [], []

    def make_objective(sink):
        def objective(x, seed=None):
            sink.append(seed)
            return 0.0
        return objective

    batch = _batch([1.0, 2.0, 3.0])
    WorkerPool(n_workers=1, experiment_seed=42).evaluate(batch, make_objective(seen_a), round_id=5)
    WorkerPool(n_workers=1, experiment_seed=42).evaluate(batch, make_objective(seen_b), round_id=5)

    assert seen_a == seen_b
    assert all(s is not None for s in seen_a)
    # Different candidates in the same round must get different seeds.
    assert len(set(seen_a)) == len(seen_a)

    # A result record's own "seed" field matches what the objective received.
    results = WorkerPool(n_workers=1, experiment_seed=42).evaluate(
        batch, make_objective([]), round_id=5
    )
    assert [r["seed"] for r in results] == seen_a


def test_different_round_id_changes_seed():
    seen = {}

    def make_objective(round_id):
        def objective(x, seed=None):
            seen[round_id] = seed
            return 0.0
        return objective

    batch = _batch([1.0])
    WorkerPool(n_workers=1, experiment_seed=42).evaluate(batch, make_objective(0), round_id=0)
    WorkerPool(n_workers=1, experiment_seed=42).evaluate(batch, make_objective(1), round_id=1)

    assert seen[0] != seen[1]


def test_n_workers_must_be_at_least_one():
    with pytest.raises(ValueError):
        WorkerPool(n_workers=0)
