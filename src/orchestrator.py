import time
from typing import Callable

import numpy as np

from candidate_merge import merge_candidates
from datastore import DataStore
from search_space import SearchSpace
from surrogates.base import BaseSurrogate
from worker_pool import WorkerPool

_ROLE_WARM_START = 0
_ROLE_SURROGATE_A = 1
_ROLE_SURROGATE_B = 2


class Orchestrator:
    """Coordinates the synchronous optimization loop.

    Model-agnostic: holds `BaseSurrogate` instances and calls only
    `name`/`fit`/`propose` on them -- it never branches on which concrete
    surrogate class it is holding. Candidate allocation between the two
    surrogates is fixed and equal every round; there is no adaptive
    allocation, no fast/slow surrogate assumption, and no per-model
    special-casing anywhere in this class.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable,
        surrogate_a: BaseSurrogate,
        surrogate_b: BaseSurrogate,
        datastore: DataStore,
        worker_pool: WorkerPool,
        n_initial: int,
        n_proposals_per_surrogate: int,
        n_rounds: int,
        experiment_seed: int | None = None,
    ):
        if surrogate_a.name == surrogate_b.name:
            raise ValueError(
                f"surrogate_a and surrogate_b must have distinct names, "
                f"both are {surrogate_a.name!r}"
            )

        self.search_space = search_space
        self.objective_fn = objective_fn
        self.surrogate_a = surrogate_a
        self.surrogate_b = surrogate_b
        self.datastore = datastore
        self.worker_pool = worker_pool
        self.n_initial = n_initial
        self.n_proposals_per_surrogate = n_proposals_per_surrogate
        self.n_rounds = n_rounds
        self.experiment_seed = experiment_seed

        self._current_best: float | None = None
        self.round_metrics: list[dict] = []

    def run(self) -> None:
        self._current_best = self.datastore.best_y
        self._run_warm_start()
        for round_id in range(1, self.n_rounds + 1):
            self._run_round(round_id)

    # -- warm start ---------------------------------------------------------

    def _run_warm_start(self) -> None:
        round_start = time.perf_counter()

        rng = self._make_rng(_ROLE_WARM_START, round_id=0)
        eval_start = time.perf_counter()
        X_warm = self.search_space.sample(self.n_initial, rng)
        warm_batch = [
            {
                "candidate": x,
                "surrogate_source": "random",
                "candidate_rank_within_source": i,
            }
            for i, x in enumerate(X_warm)
        ]
        results = self.worker_pool.evaluate(warm_batch, self.objective_fn, round_id=0)
        eval_time = time.perf_counter() - eval_start

        self._append_results(results)

        self.round_metrics.append(
            {
                "round_id": 0,
                "fit_time": {},
                "propose_time": {},
                "merge_time": 0.0,
                "eval_time": eval_time,
                "round_total_time": time.perf_counter() - round_start,
            }
        )

    # -- one optimization round ----------------------------------------------

    def _run_round(self, round_id: int) -> None:
        round_start = time.perf_counter()

        X, y = self.datastore.get_X_y()

        fit_start = time.perf_counter()
        self.surrogate_a.fit(X, y)
        fit_a_time = time.perf_counter() - fit_start

        fit_start = time.perf_counter()
        self.surrogate_b.fit(X, y)
        fit_b_time = time.perf_counter() - fit_start

        rng_a = self._make_rng(_ROLE_SURROGATE_A, round_id)
        rng_b = self._make_rng(_ROLE_SURROGATE_B, round_id)

        propose_start = time.perf_counter()
        proposals_a = self.surrogate_a.propose(
            self.search_space, self.n_proposals_per_surrogate, rng_a
        )
        propose_a_time = time.perf_counter() - propose_start

        propose_start = time.perf_counter()
        proposals_b = self.surrogate_b.propose(
            self.search_space, self.n_proposals_per_surrogate, rng_b
        )
        propose_b_time = time.perf_counter() - propose_start

        merge_start = time.perf_counter()
        batch = merge_candidates(
            {
                self.surrogate_a.name: proposals_a,
                self.surrogate_b.name: proposals_b,
            },
            {
                self.surrogate_a.name: self.n_proposals_per_surrogate,
                self.surrogate_b.name: self.n_proposals_per_surrogate,
            },
        )
        merge_time = time.perf_counter() - merge_start

        eval_start = time.perf_counter()
        results = self.worker_pool.evaluate(batch, self.objective_fn, round_id)
        eval_time = time.perf_counter() - eval_start

        self._append_results(results)

        self.round_metrics.append(
            {
                "round_id": round_id,
                "fit_time": {
                    self.surrogate_a.name: fit_a_time,
                    self.surrogate_b.name: fit_b_time,
                },
                "propose_time": {
                    self.surrogate_a.name: propose_a_time,
                    self.surrogate_b.name: propose_b_time,
                },
                "merge_time": merge_time,
                "eval_time": eval_time,
                "round_total_time": time.perf_counter() - round_start,
            }
        )

    # -- shared helpers -------------------------------------------------------

    def _append_results(self, results: list[dict]) -> None:
        for result in results:
            improved = self._current_best is None or result["y"] > self._current_best
            if improved:
                self._current_best = result["y"]
            record = dict(result)
            record["improved_global_best"] = improved
            self.datastore.append(record)

    def _make_rng(self, role: int, round_id: int) -> np.random.Generator:
        if self.experiment_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(
            np.random.SeedSequence([self.experiment_seed, role, round_id])
        )
