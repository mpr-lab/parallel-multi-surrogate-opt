import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np


def _accepts_seed(objective_fn: Callable) -> bool:
    try:
        params = inspect.signature(objective_fn).parameters
    except (TypeError, ValueError):
        return False
    return "seed" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


class WorkerPool:
    """Evaluates a merged batch of candidates against the real objective.

    Execution inside a batch may be parallel (threads), but `evaluate()`
    always blocks until the entire batch is complete -- the framework-level
    loop stays synchronous even though in-batch execution may not be.
    """

    def __init__(self, n_workers: int = 1, experiment_seed: int | None = None):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        self.experiment_seed = experiment_seed

    def evaluate(
        self,
        batch: list[dict],
        objective_fn: Callable,
        round_id: int,
    ) -> list[dict]:
        seed_supported = _accepts_seed(objective_fn)

        def run_one(indexed_entry: tuple[int, dict]) -> dict:
            index, entry = indexed_entry
            x = entry["candidate"]
            candidate_id = f"r{round_id}_c{index}"
            seed = self._derive_seed(round_id, index)

            eval_start_time = time.perf_counter()
            y = objective_fn(x, seed=seed) if seed_supported else objective_fn(x)
            eval_end_time = time.perf_counter()

            return {
                "x": x,
                "y": y,
                "round_id": round_id,
                "candidate_id": candidate_id,
                "surrogate_source": entry["surrogate_source"],
                "candidate_rank_within_source": entry["candidate_rank_within_source"],
                "eval_start_time": eval_start_time,
                "eval_end_time": eval_end_time,
                "eval_duration": eval_end_time - eval_start_time,
                "seed": seed,
            }

        tasks = list(enumerate(batch))

        if self.n_workers == 1:
            return [run_one(task) for task in tasks]

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            return list(executor.map(run_one, tasks))

    def _derive_seed(self, round_id: int, index: int) -> int | None:
        if self.experiment_seed is None:
            return None
        entropy = [self.experiment_seed, round_id, index]
        return int(np.random.SeedSequence(entropy).generate_state(1)[0])
