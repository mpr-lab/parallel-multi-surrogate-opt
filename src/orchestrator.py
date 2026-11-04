import time
from typing import Callable

import numpy as np

from candidate_merge import merge_candidates
from datastore import DataStore
from metrics import MetricsLog
from parameter_space import ParameterSpace
from search_space import SearchSpace
from surrogates.base import BaseSurrogate
from worker_pool import WorkerPool

_ROLE_WARM_START = 0
_ROLE_SURROGATE_A = 1
_ROLE_SURROGATE_B = 2

# Upper bound on how many candidates to request from propose() when a
# finite ParameterSpace is active, so decode-aware refill (skipping
# in-batch AND historical duplicates) has enough ranked headroom without
# asking for an unbounded pool on a hypothetical huge discrete space.
_MAX_PROPOSAL_OVERSAMPLE = 500


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
        parameter_space: ParameterSpace | None = None,
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
        # Optional encode/decode scheme (see ParameterSpace). None
        # (default) means every candidate stays a raw numeric vector end
        # to end, identical to the framework's original behavior -- no
        # decoding, no evaluated-config tracking, no oversampling.
        self.parameter_space = parameter_space

        self._current_best: float | None = None
        # Canonical decoded-config keys already evaluated (warm start +
        # every round so far). Seeded into CandidateMerge's dedup set so
        # a candidate decoding to an already-evaluated configuration is
        # skipped, not silently re-evaluated as if it were new.
        self._evaluated_configs: set[tuple] = set()
        self.metrics = MetricsLog()

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
        warm_batch = []
        for i, x in enumerate(X_warm):
            entry = {
                "candidate": x,
                "surrogate_source": "random",
                "candidate_rank_within_source": i,
            }
            if self.parameter_space is not None:
                entry["decoded_params"] = self.parameter_space.decode(x)
            warm_batch.append(entry)
        results = self.worker_pool.evaluate(warm_batch, self.objective_fn, round_id=0)
        eval_time = time.perf_counter() - eval_start

        self._append_results(results)

        self.metrics.record_warm_start(
            n_initial=self.n_initial,
            evaluation_wall_time=eval_time,
            total_round_wall_time=time.perf_counter() - round_start,
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

        # With a finite ParameterSpace, ask each surrogate for a much
        # longer ranked list than we actually need, so CandidateMerge has
        # enough fallback room to skip past in-batch AND historical
        # decoded duplicates without starving the batch. Continuous
        # (parameter_space is None) benchmarks are unaffected.
        propose_n = self._propose_count()

        propose_start = time.perf_counter()
        proposals_a = self.surrogate_a.propose(self.search_space, propose_n, rng_a)
        propose_a_time = time.perf_counter() - propose_start

        propose_start = time.perf_counter()
        proposals_b = self.surrogate_b.propose(self.search_space, propose_n, rng_b)
        propose_b_time = time.perf_counter() - propose_start

        if self.parameter_space is not None:
            decode_fn = self.parameter_space.decode
            evaluated_configs = self._evaluated_configs
            total_configurations = self.parameter_space.total_configurations
        else:
            decode_fn = None
            evaluated_configs = None
            total_configurations = None

        batch = merge_candidates(
            {
                self.surrogate_a.name: proposals_a,
                self.surrogate_b.name: proposals_b,
            },
            {
                self.surrogate_a.name: self.n_proposals_per_surrogate,
                self.surrogate_b.name: self.n_proposals_per_surrogate,
            },
            decode_fn=decode_fn,
            evaluated_configs=evaluated_configs,
            total_configurations=total_configurations,
        )

        eval_start = time.perf_counter()
        results = self.worker_pool.evaluate(batch, self.objective_fn, round_id)
        eval_time = time.perf_counter() - eval_start

        self._append_results(results)

        candidates_from_a = sum(
            1 for entry in batch if entry["surrogate_source"] == self.surrogate_a.name
        )
        candidates_from_b = sum(
            1 for entry in batch if entry["surrogate_source"] == self.surrogate_b.name
        )

        self.metrics.record_round(
            round_id=round_id,
            best_y_so_far=self._current_best,
            surrogate_a_name=self.surrogate_a.name,
            surrogate_b_name=self.surrogate_b.name,
            surrogate_a_fit_time=fit_a_time,
            surrogate_b_fit_time=fit_b_time,
            surrogate_a_proposal_time=propose_a_time,
            surrogate_b_proposal_time=propose_b_time,
            surrogate_a_candidates_contributed=candidates_from_a,
            surrogate_b_candidates_contributed=candidates_from_b,
            evaluation_wall_time=eval_time,
            total_round_wall_time=time.perf_counter() - round_start,
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

            decoded_params = result.get("decoded_params")
            if decoded_params is not None:
                self._evaluated_configs.add(tuple(sorted(decoded_params.items())))

    def _propose_count(self) -> int:
        if self.parameter_space is None:
            return self.n_proposals_per_surrogate
        # propose() ranks a pool and returns only its top-N -- requesting
        # exactly total_configurations is NOT enough to guarantee every
        # bucket is represented, since the top-N by predicted mean can
        # still cluster into a few buckets even when N == the number of
        # possible buckets (the discarded remainder of the pool may have
        # covered the rest). Request the full oversample cap regardless,
        # which also grows the underlying pool for every current
        # concrete surrogate (pool_size = max(configured pool_size, N)),
        # improving decode-coverage, not just list length.
        return _MAX_PROPOSAL_OVERSAMPLE

    def _make_rng(self, role: int, round_id: int) -> np.random.Generator:
        if self.experiment_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(
            np.random.SeedSequence([self.experiment_seed, role, round_id])
        )
