import numpy as np

REQUIRED_FIELDS = (
    "x",
    "y",
    "round_id",
    "candidate_id",
    "surrogate_source",
    "candidate_rank_within_source",
    "eval_duration",
    "improved_global_best",
)


class DataStore:
    """The single shared history of real evaluations.

    Pure storage: makes no optimization decisions. `best_y` is a read-only
    summary over records already stored, not a decision about what to
    evaluate next — callers (e.g. the WorkerPool/Orchestrator) are
    responsible for computing fields like `improved_global_best` before
    appending, not the store itself.
    """

    def __init__(self):
        self._records: list[dict] = []

    def append(self, record: dict) -> None:
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        if missing:
            raise ValueError(f"record missing required field(s): {missing}")

        x = np.asarray(record["x"], dtype=float).reshape(-1)
        if self._records:
            expected_dim = len(self._records[0]["x"])
            if x.shape[0] != expected_dim:
                raise ValueError(
                    f"x has dim {x.shape[0]}, expected {expected_dim} to "
                    f"match existing records"
                )

        # Normalize to a plain tuple of floats: a consistent, serializable
        # representation that doesn't alias a caller-owned numpy array.
        # Extra metadata keys are preserved as-is.
        normalized = dict(record)
        normalized["x"] = tuple(x.tolist())
        normalized["y"] = float(record["y"])
        self._records.append(normalized)

    def append_batch(self, records: list[dict]) -> None:
        for record in records:
            self.append(record)

    def get_all(self) -> list[dict]:
        """Full records, including metadata, for metrics/analysis. Returns
        shallow copies so callers can't mutate internal state."""
        return [dict(r) for r in self._records]

    def get_X_y(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._records:
            return np.empty((0, 0)), np.empty((0,))
        X = np.array([r["x"] for r in self._records], dtype=float)
        y = np.array([r["y"] for r in self._records], dtype=float)
        return X, y

    @property
    def best_y(self) -> float | None:
        """Highest y observed so far, or None if the store is empty."""
        if not self._records:
            return None
        return max(r["y"] for r in self._records)

    def __len__(self) -> int:
        return len(self._records)
