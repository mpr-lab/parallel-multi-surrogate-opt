import numpy as np
import pytest

from datastore import DataStore
from orchestrator import Orchestrator
from search_space import SearchSpace
from surrogates.base import BaseSurrogate
from worker_pool import WorkerPool


class _CountingDummySurrogate(BaseSurrogate):
    """Proposes random samples from the search space; records every call
    so tests can inspect exactly how the Orchestrator used it."""

    def __init__(self, label: str):
        self._label = label
        self.fit_calls = 0
        self.fit_history: list[tuple[np.ndarray, np.ndarray]] = []
        self.propose_calls: list[int] = []

    @property
    def name(self) -> str:
        return self._label

    def fit(self, X, y):
        self.fit_calls += 1
        self.fit_history.append((X.copy(), y.copy()))

    def propose(self, search_space, n_candidates, rng):
        self.propose_calls.append(n_candidates)
        return search_space.sample(n_candidates, rng)


class _LinspaceDummySurrogate(BaseSurrogate):
    """A deliberately different proposal strategy (no randomness, ignores
    fit history entirely) -- used to prove the Orchestrator doesn't care
    how a surrogate implements `propose`."""

    def __init__(self, label: str, low: float, high: float):
        self._label = label
        self._low = low
        self._high = high
        self.fit_calls = 0

    @property
    def name(self) -> str:
        return self._label

    def fit(self, X, y):
        self.fit_calls += 1

    def propose(self, search_space, n_candidates, rng):
        return np.linspace(self._low, self._high, n_candidates).reshape(-1, 1)


class _SequencedObjective:
    """Returns a predetermined sequence of y values regardless of x, so
    tests can control `improved_global_best` outcomes exactly."""

    def __init__(self, values: list[float]):
        self._values = iter(values)

    def __call__(self, x, seed=None):
        return next(self._values)


def _cheap_objective(x, seed=None):
    return float(np.sum(x))


def _build_orchestrator(
    datastore=None,
    n_initial=4,
    n_proposals_per_surrogate=3,
    n_rounds=2,
    objective_fn=_cheap_objective,
    surrogate_a=None,
    surrogate_b=None,
    experiment_seed=0,
):
    return Orchestrator(
        search_space=SearchSpace(bounds=[(0.0, 10.0)]),
        objective_fn=objective_fn,
        surrogate_a=surrogate_a or _CountingDummySurrogate("surrogate_a"),
        surrogate_b=surrogate_b or _CountingDummySurrogate("surrogate_b"),
        datastore=datastore if datastore is not None else DataStore(),
        worker_pool=WorkerPool(n_workers=1),
        n_initial=n_initial,
        n_proposals_per_surrogate=n_proposals_per_surrogate,
        n_rounds=n_rounds,
        experiment_seed=experiment_seed,
    )


def test_warm_start_creates_expected_number_of_records():
    store = DataStore()
    orch = _build_orchestrator(datastore=store, n_initial=5, n_rounds=0)

    orch.run()

    assert len(store) == 5
    assert all(r["surrogate_source"] == "random" for r in store.get_all())


def test_both_surrogates_see_the_same_history_each_round():
    surrogate_a = _CountingDummySurrogate("surrogate_a")
    surrogate_b = _CountingDummySurrogate("surrogate_b")
    orch = _build_orchestrator(
        n_initial=4, n_proposals_per_surrogate=2, n_rounds=3,
        surrogate_a=surrogate_a, surrogate_b=surrogate_b,
    )

    orch.run()

    assert len(surrogate_a.fit_history) == len(surrogate_b.fit_history) == 3
    for (X_a, y_a), (X_b, y_b) in zip(surrogate_a.fit_history, surrogate_b.fit_history):
        np.testing.assert_array_equal(X_a, X_b)
        np.testing.assert_array_equal(y_a, y_b)


def test_both_surrogates_are_fit_exactly_once_per_round():
    surrogate_a = _CountingDummySurrogate("surrogate_a")
    surrogate_b = _CountingDummySurrogate("surrogate_b")
    orch = _build_orchestrator(
        n_rounds=4, surrogate_a=surrogate_a, surrogate_b=surrogate_b,
    )

    orch.run()

    assert surrogate_a.fit_calls == 4
    assert surrogate_b.fit_calls == 4


def test_requested_proposal_counts_are_correct():
    surrogate_a = _CountingDummySurrogate("surrogate_a")
    surrogate_b = _CountingDummySurrogate("surrogate_b")
    orch = _build_orchestrator(
        n_proposals_per_surrogate=7, n_rounds=3,
        surrogate_a=surrogate_a, surrogate_b=surrogate_b,
    )

    orch.run()

    assert surrogate_a.propose_calls == [7, 7, 7]
    assert surrogate_b.propose_calls == [7, 7, 7]


def test_merged_batch_and_total_evaluation_count_are_correct():
    store = DataStore()
    orch = _build_orchestrator(
        datastore=store, n_initial=4, n_proposals_per_surrogate=3, n_rounds=2,
    )

    orch.run()

    assert len(store) == 4 + 2 * (2 * 3)


@pytest.mark.parametrize("n_rounds", [0, 1, 2, 3])
def test_datastore_grows_correctly_as_rounds_increase(n_rounds):
    store = DataStore()
    orch = _build_orchestrator(
        datastore=store, n_initial=4, n_proposals_per_surrogate=3, n_rounds=n_rounds,
    )

    orch.run()

    assert len(store) == 4 + n_rounds * 2 * 3


def test_improved_global_best_follows_maximize_convention():
    store = DataStore()
    # Warm start (3 evals): 1.0 -> True, 5.0 -> True, 3.0 -> False.
    # Round 1, batch = [surrogate_a's 1 candidate, surrogate_b's 1 candidate]:
    # 10.0 -> True, 2.0 -> False.
    objective = _SequencedObjective([1.0, 5.0, 3.0, 10.0, 2.0])
    orch = _build_orchestrator(
        datastore=store,
        n_initial=3,
        n_proposals_per_surrogate=1,
        n_rounds=1,
        objective_fn=objective,
    )

    orch.run()

    records = store.get_all()
    assert [r["y"] for r in records] == [1.0, 5.0, 3.0, 10.0, 2.0]
    assert [r["improved_global_best"] for r in records] == [
        True, True, False, True, False,
    ]
    assert store.best_y == 10.0


def test_orchestrator_is_agnostic_to_surrogate_implementation():
    surrogate_a = _CountingDummySurrogate("surrogate_a")
    surrogate_b = _LinspaceDummySurrogate("surrogate_b", low=0.0, high=10.0)
    store = DataStore()
    orch = _build_orchestrator(
        datastore=store,
        n_initial=2,
        n_proposals_per_surrogate=3,
        n_rounds=2,
        surrogate_a=surrogate_a,
        surrogate_b=surrogate_b,
    )

    orch.run()  # must not raise, despite the two surrogates working nothing alike

    assert len(store) == 2 + 2 * (2 * 3)
    assert surrogate_a.fit_calls == 2
    assert surrogate_b.fit_calls == 2


def test_rejects_surrogates_with_duplicate_names():
    with pytest.raises(ValueError):
        _build_orchestrator(
            surrogate_a=_CountingDummySurrogate("same_name"),
            surrogate_b=_CountingDummySurrogate("same_name"),
        )
