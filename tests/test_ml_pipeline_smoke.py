import numpy as np

from datastore import DataStore
from objectives.classical_ml import (
    LOGISTIC_REGRESSION_PARAMETER_SPACE,
    make_logistic_regression_objective,
)
from orchestrator import Orchestrator
from search_space import SearchSpace
from surrogates import GaussianProcessSurrogate, RandomForestSurrogate
from worker_pool import WorkerPool


def test_ml_objective_runs_through_full_pipeline():
    """warm start -> surrogate fit -> candidate proposal -> CandidateMerge
    -> evaluation -> DataStore -> metrics, using a real ML objective
    wired in through the exact same Orchestrator synthetic benchmarks use.
    """
    parameter_space = LOGISTIC_REGRESSION_PARAMETER_SPACE
    objective_fn = make_logistic_regression_objective(dataset="synthetic_classification", seed=0)

    search_space = SearchSpace(bounds=parameter_space.bounds)
    surrogate_a = GaussianProcessSurrogate(pool_size=50, random_state=0)
    surrogate_b = RandomForestSurrogate(n_estimators=5, pool_size=50, random_state=0)
    datastore = DataStore()
    worker_pool = WorkerPool(n_workers=1, experiment_seed=0)

    # This ParameterSpace has only 8 possible decoded configurations.
    # Orchestrator now over-requests proposals (up to total_configurations)
    # and skips both in-batch AND historical decoded duplicates, so normal
    # operation no longer depends on a lucky seed (see
    # test_candidate_merge_decode.py / test_collision_and_exhaustion.py for
    # that fix in isolation). This smoke test's own budget (7 evaluations)
    # is kept safely under the 8-config ceiling on purpose -- going over it
    # deterministically exhausts the space, which is correct behavior
    # tested separately, not a "flaky" failure to work around here.
    n_initial = 3
    n_proposals_per_surrogate = 1
    n_rounds = 2
    orchestrator = Orchestrator(
        search_space=search_space,
        objective_fn=objective_fn,
        surrogate_a=surrogate_a,
        surrogate_b=surrogate_b,
        datastore=datastore,
        worker_pool=worker_pool,
        n_initial=n_initial,
        n_proposals_per_surrogate=n_proposals_per_surrogate,
        n_rounds=n_rounds,
        experiment_seed=0,
        parameter_space=parameter_space,
    )

    orchestrator.run()

    assert len(datastore) == n_initial + n_rounds * (2 * n_proposals_per_surrogate)

    allowed_C = LOGISTIC_REGRESSION_PARAMETER_SPACE.dimensions[0].choices
    allowed_penalty = LOGISTIC_REGRESSION_PARAMETER_SPACE.dimensions[1].choices
    for record in datastore.get_all():
        assert 0.0 <= record["y"] <= 1.0
        assert set(record["decoded_params"]) == {"C", "penalty"}
        assert record["decoded_params"]["C"] in allowed_C
        assert record["decoded_params"]["penalty"] in allowed_penalty

    # Surrogates actually fit on the shared raw-numeric X,y history.
    X, y = datastore.get_X_y()
    assert X.shape == (n_initial + n_rounds * 2 * n_proposals_per_surrogate, 2)
    assert np.all((X >= 0.0) & (X <= 1.0))

    assert orchestrator.metrics.warm_start is not None
    assert len(orchestrator.metrics.rounds) == n_rounds
    for round_record in orchestrator.metrics.rounds:
        assert round_record["surrogate_a_name"] == "gaussian_process"
        assert round_record["surrogate_b_name"] == "random_forest"
