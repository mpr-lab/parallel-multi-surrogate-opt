import numpy as np
import pytest

from candidate_merge import SearchSpaceExhaustedError, merge_candidates
from datastore import DataStore
from objectives.classical_ml import (
    LOGISTIC_REGRESSION_PARAMETER_SPACE,
    make_logistic_regression_objective,
)
from orchestrator import Orchestrator
from parameter_space import ChoiceDimension, ParameterSpace
from search_space import SearchSpace
from surrogates import GaussianProcessSurrogate, RandomForestSurrogate
from worker_pool import WorkerPool

SPACE = ParameterSpace(
    [ChoiceDimension("C", [0.01, 0.1, 1, 10]), ChoiceDimension("penalty", ["l1", "l2"])]
)
TINY_SPACE = ParameterSpace([ChoiceDimension("x", ["a", "b"])])  # total_configurations == 2


# -- CandidateMerge unit tests --------------------------------------------


def test_two_raw_candidates_decoding_to_same_configuration_are_deduped():
    # [0.05, 0.1] and [0.2, 0.3] are numerically different but both decode
    # to {"C": 0.01, "penalty": "l1"}.
    proposals = {"A": np.array([[0.05, 0.1], [0.9, 0.9]])}
    batch = merge_candidates(proposals, {"A": 1}, decode_fn=SPACE.decode)
    assert batch[0]["decoded_params"] == {"C": 0.01, "penalty": "l1"}

    proposals_with_dup_first = {"A": np.array([[0.2, 0.3], [0.9, 0.9]])}
    # Same decoded config as above at rank 0 -- on its own this isn't a
    # dup (nothing else in the batch yet), included for contrast with the
    # refill test below, which puts the duplicate at rank 0 deliberately.
    batch2 = merge_candidates(proposals_with_dup_first, {"A": 1}, decode_fn=SPACE.decode)
    assert batch2[0]["decoded_params"] == {"C": 0.01, "penalty": "l1"}


def test_refill_chooses_next_ranked_decoded_unique_candidate():
    proposals = {
        "A": np.array([[0.05, 0.1]]),  # -> {"C": 0.01, "penalty": "l1"}
        "B": np.array([[0.2, 0.3], [0.6, 0.1], [0.7, 0.6]]),  # rank0 dup of A; rank1,2 distinct
    }
    batch = merge_candidates(proposals, {"A": 1, "B": 1}, decode_fn=SPACE.decode)

    b_entry = next(e for e in batch if e["surrogate_source"] == "B")
    assert b_entry["candidate_rank_within_source"] == 1  # rank 0 was skipped as a decoded dup
    assert b_entry["decoded_params"] == {"C": 1, "penalty": "l1"}


def test_candidate_rejected_if_decoded_config_already_evaluated_in_history():
    # {"C": 0.01, "penalty": "l1"} is already in evaluated_configs (prior
    # round / warm start), even though it never appears in THIS batch.
    already_evaluated = {tuple(sorted({"C": 0.01, "penalty": "l1"}.items()))}
    proposals = {"A": np.array([[0.05, 0.1], [0.9, 0.9]])}  # rank0 is the historical dup

    batch = merge_candidates(
        proposals, {"A": 1}, decode_fn=SPACE.decode, evaluated_configs=already_evaluated
    )

    assert len(batch) == 1
    assert batch[0]["candidate_rank_within_source"] == 1  # rank 0 skipped (seen in history)
    assert batch[0]["decoded_params"] == {"C": 10, "penalty": "l2"}


def test_exact_finite_space_exhaustion_raises_specific_error():
    # TINY_SPACE has exactly 2 possible configurations: {"x": "a"}, {"x": "b"}.
    already_evaluated = {tuple(sorted({"x": "a"}.items()))}
    proposals = {
        "A": np.array([[0.9]]),  # decodes to "b" -- the one remaining unused config
        "B": np.array([[0.1], [0.9]]),  # decodes to "a" (used), "b" (now also used by A)
    }

    with pytest.raises(SearchSpaceExhaustedError):
        merge_candidates(
            proposals,
            {"A": 1, "B": 1},
            decode_fn=TINY_SPACE.decode,
            evaluated_configs=already_evaluated,
            total_configurations=TINY_SPACE.total_configurations,
        )


def test_non_exhausted_insufficient_candidates_raises_plain_value_error_not_exhaustion():
    # Only 1 of TINY_SPACE's 2 configs has ever been used -- genuinely NOT
    # exhausted -- but B's own ranked list happens to only ever decode to
    # the already-used one, so B specifically can't fill its allocation.
    already_evaluated = {tuple(sorted({"x": "a"}.items()))}
    proposals = {"B": np.array([[0.1], [0.2]])}  # both decode to "a"

    with pytest.raises(ValueError) as exc_info:
        merge_candidates(
            proposals,
            {"B": 1},
            decode_fn=TINY_SPACE.decode,
            evaluated_configs=already_evaluated,
            total_configurations=TINY_SPACE.total_configurations,
        )
    assert not isinstance(exc_info.value, SearchSpaceExhaustedError)


# -- Orchestrator-level integration tests ---------------------------------


def _build_ml_orchestrator(seed, n_initial, n_proposals_per_surrogate, n_rounds):
    parameter_space = LOGISTIC_REGRESSION_PARAMETER_SPACE
    objective_fn = make_logistic_regression_objective(dataset="synthetic_classification", seed=seed)
    search_space = SearchSpace(bounds=parameter_space.bounds)
    return Orchestrator(
        search_space=search_space,
        objective_fn=objective_fn,
        surrogate_a=GaussianProcessSurrogate(random_state=seed),
        surrogate_b=RandomForestSurrogate(random_state=seed),
        datastore=DataStore(),
        worker_pool=WorkerPool(n_workers=1, experiment_seed=seed),
        n_initial=n_initial,
        n_proposals_per_surrogate=n_proposals_per_surrogate,
        n_rounds=n_rounds,
        experiment_seed=seed,
        parameter_space=parameter_space,
    )


@pytest.mark.parametrize("seed", range(10))
def test_no_spurious_collisions_across_many_seeds_within_space_budget(seed):
    # LOGISTIC_REGRESSION_PARAMETER_SPACE has 8 total configurations;
    # 5 + 1*2 = 7 stays safely under that ceiling. Before the oversampling
    # + history-aware refill fix, this failed for roughly half of seeds
    # 0-19 under these exact (CLI-default) surrogate settings.
    orchestrator = _build_ml_orchestrator(
        seed, n_initial=5, n_proposals_per_surrogate=1, n_rounds=1
    )
    orchestrator.run()  # must not raise
    assert len(orchestrator.datastore) == 7


def test_exhaustion_is_raised_cleanly_when_budget_exceeds_space_size():
    orchestrator = _build_ml_orchestrator(
        seed=0, n_initial=5, n_proposals_per_surrogate=1, n_rounds=3  # 5 + 3*2 = 11 > 8
    )
    with pytest.raises(SearchSpaceExhaustedError):
        orchestrator.run()


def test_same_decoded_configuration_is_never_evaluated_twice_across_rounds():
    orchestrator = _build_ml_orchestrator(
        seed=2, n_initial=5, n_proposals_per_surrogate=1, n_rounds=1
    )
    orchestrator.run()

    decoded_keys = [
        tuple(sorted(r["decoded_params"].items())) for r in orchestrator.datastore.get_all()
    ]
    assert len(decoded_keys) == len(set(decoded_keys)), "a decoded configuration was evaluated twice"


def test_deterministic_given_fixed_experiment_seed():
    orchestrator_a = _build_ml_orchestrator(
        seed=5, n_initial=5, n_proposals_per_surrogate=1, n_rounds=1
    )
    orchestrator_b = _build_ml_orchestrator(
        seed=5, n_initial=5, n_proposals_per_surrogate=1, n_rounds=1
    )
    orchestrator_a.run()
    orchestrator_b.run()

    records_a = orchestrator_a.datastore.get_all()
    records_b = orchestrator_b.datastore.get_all()

    assert [r["x"] for r in records_a] == [r["x"] for r in records_b]
    assert [r["y"] for r in records_a] == [r["y"] for r in records_b]
    assert [r["decoded_params"] for r in records_a] == [r["decoded_params"] for r in records_b]
