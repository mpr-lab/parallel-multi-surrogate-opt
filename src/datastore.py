"""
Shared optimization history for the Parallel Multi-Surrogate BO prototype.

DataStore is the single place where every evaluated hyperparameter
configuration and its metadata are collected.  Both surrogate models
(RF and GP) read from it; the orchestrator writes to it after each batch
of evaluations.

Design philosophy
-----------------
- Primary storage is a plain Python list of dicts: ``self.records``.
  This is easy to inspect in a debugger, easy to serialise, and requires
  no external dependencies.
- No locking, no multiprocessing managers, no async logic.  The
  orchestrator runs a single-process loop; parallelism happens only
  during worker evaluation (worker_pool.py), and results are collected
  back into the main process before being appended here.
- The new primary API (``append`` / ``get_X_y`` / ``to_dataframe`` / ...)
  is what you should use for new code.
- The older API (``add_observations`` / ``.X`` / ``.y`` / ...) is kept as
  a thin compatibility layer so that orchestrator.py and the experiment
  runner scripts continue to work without modification.

Typical record fields
---------------------
Required (validated on insert):
    ``x``               -- array-like, the evaluated hyperparameter config
    ``y``               -- float, the observed objective value
    ``round_id``        -- int, which BO round this came from (-1 = warm start)

Optional (set by orchestrator / worker pool, all are just stored as-is):
    ``candidate_id``        -- position of this candidate in the round's batch
    ``surrogate_source``    -- "rf", "gp", or "random"
    ``eval_start_time``     -- float (time.perf_counter)
    ``eval_end_time``       -- float
    ``eval_duration``       -- float, seconds
    ``was_gp_stale``        -- bool, was GP model stale when this was proposed?
    ``gp_model_version``    -- int, how many times the GP has been retrained
    ``rf_model_version``    -- int, same for RF
    ``accepted_into_batch`` -- bool, was this candidate actually evaluated?
"""

import os
import json
import numpy as np


# Fields that every record must contain.  Validation happens in append().
_REQUIRED_FIELDS = ("x", "y", "round_id")


class DataStore:
    """
    Lightweight in-memory store for BO observations, backed by a list of dicts.

    New-API example
    ---------------
    >>> ds = DataStore()
    >>> ds.append({"x": [0.5, 1.2], "y": -3.14, "round_id": 0})
    >>> ds.append({"x": [0.1, 0.9], "y": -1.01, "round_id": 0})
    >>> X, y = ds.get_X_y()    # numpy arrays
    >>> ds.save_csv("results/run.csv")

    Orchestrator-API example (backward compat)
    -------------------------------------------
    >>> ds = DataStore(dim=2, output_dir="results/branin", run_name="msbo_seed42")
    >>> ds.add_observations(X_batch, y_batch, round_id=3, source="rf")
    >>> rf.fit(ds.X, ds.y)
    """

    def __init__(
        self,
        dim: int | None = None,
        output_dir: str | None = None,
        run_name: str = "run",
    ):
        """
        Args:
            dim:        Number of hyperparameter dimensions.  Optional -- only
                        needed so that ``ds.X`` returns an (0, dim) empty array
                        before any data has been inserted.  Inferred from the
                        first record if omitted.
            output_dir: Where to write CSV / JSON output files.  The directory
                        is created on first write if it doesn't exist yet.
                        Pass ``None`` if you only need in-memory storage.
            run_name:   A human-readable label for this run (e.g.
                        ``"parallel_msbo_branin_seed42"``).  Used as a filename
                        prefix and embedded in CSV rows.  Can be reassigned
                        after construction.
        """
        # Primary storage: every evaluated configuration lives here as a dict.
        self.records: list[dict] = []

        # Kept for backward compatibility and for naming output files.
        self.dim        = dim
        self.output_dir = output_dir or "results"
        self.run_name   = run_name

    # ---------------------------------------------------------------------- #
    #  Core write API                                                          #
    # ---------------------------------------------------------------------- #

    def append(self, record: dict) -> None:
        """
        Append a single record to the store.

        Validates that the three required fields are present before inserting.
        Stores ``x`` as a plain Python list so records are JSON-serialisable.

        Args:
            record: A dict that must contain at least ``"x"``, ``"y"``, and
                    ``"round_id"``.  Any extra fields (eval_duration, source,
                    etc.) are stored as-is.

        Raises:
            ValueError: if any required field is missing.

        Example:
            ds.append({
                "x":               [0.5, 1.2],
                "y":               -3.14,
                "round_id":        2,
                "surrogate_source": "rf",
                "eval_duration":   0.003,
            })
        """
        # Minimal validation -- keep it readable, not exhaustive.
        missing = [f for f in _REQUIRED_FIELDS if f not in record]
        if missing:
            raise ValueError(
                f"Record is missing required field(s): {missing}.  "
                f"Record keys present: {list(record.keys())}"
            )

        # Normalise x to a plain Python list so the record is JSON-safe and
        # the dict stays lightweight (no numpy arrays hiding inside).
        normalised = dict(record)
        normalised["x"] = np.asarray(record["x"], dtype=float).tolist()
        normalised["y"] = float(record["y"])

        self.records.append(normalised)

    def append_batch(self, records: list[dict]) -> None:
        """
        Append multiple records at once.

        Delegates to ``append()`` so validation runs for every record.

        Args:
            records: List of record dicts.  Each must satisfy the same
                     requirements as a single ``append()`` call.
        """
        for record in records:
            self.append(record)

    # ---------------------------------------------------------------------- #
    #  Core read API                                                           #
    # ---------------------------------------------------------------------- #

    def get_X_y(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return all evaluated configs and objective values as NumPy arrays.

        Returns:
            X: shape ``(n_obs, dim)`` -- hyperparameter configurations.
            y: shape ``(n_obs,)``     -- objective values (higher = better).

        Returns empty arrays with the right shape when the store is empty.

        Example:
            X, y = ds.get_X_y()
            surrogate.fit(X, y)
        """
        if not self.records:
            # Return typed empties so callers never have to guard against None.
            cols = self.dim if self.dim is not None else 1
            return np.empty((0, cols), dtype=float), np.empty((0,), dtype=float)

        X = np.array([r["x"] for r in self.records], dtype=float)
        y = np.array([r["y"] for r in self.records], dtype=float)
        return X, y

    def get_all(self) -> list[dict]:
        """
        Return a copy of all records in insertion order.

        Returns a shallow copy so the caller cannot accidentally mutate the
        store's internal list.
        """
        return list(self.records)

    def get_latest(self, n: int | None = None) -> list[dict]:
        """
        Return the most recent ``n`` records, or all records if ``n`` is None.

        Args:
            n: How many records to return from the end of the store.
               ``None`` means "return everything" (same as ``get_all()``).

        Returns:
            A list of dicts, newest last.
        """
        if n is None:
            return list(self.records)
        # Clamp so we don't error on n > len(records)
        return list(self.records[-n:]) if n > 0 else []

    def current_round(self) -> int:
        """
        Return the highest ``round_id`` currently stored.

        Returns ``0`` if the store is empty (so callers can always do
        arithmetic on the result without a special-case check).
        """
        if not self.records:
            return 0
        return max(r["round_id"] for r in self.records)

    def is_empty(self) -> bool:
        """Return True if no records have been stored yet."""
        return len(self.records) == 0

    def __len__(self) -> int:
        """Support ``len(datastore)``."""
        return len(self.records)

    # ---------------------------------------------------------------------- #
    #  Export                                                                  #
    # ---------------------------------------------------------------------- #

    def to_dataframe(self):
        """
        Convert all records to a pandas DataFrame.

        The ``x`` array in each record is expanded into individual columns
        ``x0``, ``x1``, ... so the DataFrame is flat and easy to use with
        matplotlib or seaborn.

        Returns:
            A ``pandas.DataFrame`` in insertion order.  Returns an empty
            DataFrame (no rows, no columns) if the store is empty.

        Example:
            df = ds.to_dataframe()
            df.to_csv("results.csv", index=False)
            df.groupby("surrogate_source")["y"].mean()
        """
        import pandas as pd

        if not self.records:
            return pd.DataFrame()

        rows = []
        for record in self.records:
            # Flatten x[0], x[1], ... into separate columns, then keep
            # everything else as-is.
            flat: dict = {}
            for key, value in record.items():
                if key == "x":
                    for i, xi in enumerate(value):
                        flat[f"x{i}"] = float(xi)
                else:
                    flat[key] = value
            rows.append(flat)

        return pd.DataFrame(rows)

    def save_csv(self, path: str) -> None:
        """
        Export all records to a CSV file via pandas.

        Creates any missing parent directories automatically.

        Args:
            path: Full path to the output CSV file, e.g. ``"results/run.csv"``.

        Note:
            The CSV will have columns ``x0``, ``x1``, ..., ``y``, ``round_id``,
            plus any optional fields present in the records.
        """
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        self.to_dataframe().to_csv(abs_path, index=False)

    # ---------------------------------------------------------------------- #
    #  Backward-compatible API                                                 #
    #  (used by orchestrator.py, main.py, run_parallel_msbo.py, etc.)         #
    #  These methods delegate to the new API above; do not use them in new    #
    #  code -- prefer append() / get_X_y() / save_csv() instead.              #
    # ---------------------------------------------------------------------- #

    @property
    def X(self) -> np.ndarray:
        """All tried configurations as a NumPy array, shape (n_obs, dim)."""
        X, _ = self.get_X_y()
        return X

    @property
    def y(self) -> np.ndarray:
        """All objective values as a NumPy array, shape (n_obs,)."""
        _, y = self.get_X_y()
        return y

    @property
    def best_y(self) -> float:
        """Best (highest) objective value observed so far."""
        if not self.records:
            return float("-inf")
        return float(np.max(self.y))

    @property
    def n_obs(self) -> int:
        """Total number of observations stored."""
        return len(self.records)

    def add_observations(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        round_id: int,
        source: str = "unknown",
    ) -> None:
        """
        Insert one or more observations (backward-compat wrapper).

        Converts the NumPy batch into individual records and delegates to
        ``append()``.  Prefer ``append()`` / ``append_batch()`` in new code.

        Args:
            X_new:    Configs to store, shape (n_new, dim) or (dim,).
            y_new:    Objective values,  shape (n_new,) or scalar.
            round_id: Which BO round these came from (-1 = warm start).
            source:   Which surrogate proposed them: "rf", "gp", or "random".
        """
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        y_new = np.atleast_1d(np.asarray(y_new, dtype=float))

        for i in range(len(y_new)):
            self.append({
                "x":               X_new[i].tolist(),
                "y":               float(y_new[i]),
                "round_id":        round_id,
                "surrogate_source": source,
            })

    def save_results_csv(self) -> str:
        """
        Write all observations to a CSV file named after this run.

        The CSV includes a ``run_name`` column and a ``seed`` column (parsed
        from the run name if it contains ``_seed<N>``) so result files are
        self-identifying when you aggregate across many runs.

        Returns:
            Absolute path to the saved file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{self.run_name}_results.csv")

        # Build a DataFrame, then add run-level annotation columns.
        import pandas as pd
        df = self.to_dataframe()

        if not df.empty:
            df["run_name"] = self.run_name
            # Attempt to parse the seed from the run name (format: *_seed<N>)
            seed_str = (
                self.run_name.split("_seed")[-1]
                if "_seed" in self.run_name
                else ""
            )
            df["seed"] = seed_str

        abs_path = os.path.abspath(path)
        df.to_csv(abs_path, index=False)
        return abs_path

    def save_metadata_json(self, metadata: dict) -> str:
        """
        Write run-level metadata to a JSON file alongside the CSV.

        Adds ``final_best_y`` and ``n_observations`` summary fields so the
        JSON is self-contained without needing to read the CSV.

        Args:
            metadata: Dict produced by ``build_run_metadata()`` in
                      ``src/utils/serialization.py``.

        Returns:
            Absolute path to the saved file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{self.run_name}_metadata.json")
        payload = {
            **metadata,
            "final_best_y":   self.best_y,
            "n_observations": self.n_obs,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        return os.path.abspath(path)

    def get_convergence_curve(self) -> list[dict]:
        """
        Return per-round best-y values for plotting a convergence curve.

        Returns:
            List of dicts ``[{"round": r, "best_y": v, "n_obs": n}, ...]``,
            one entry per unique ``round_id``, in ascending round order.
        """
        if not self.records:
            return []

        summaries = []
        seen_rounds = sorted(set(r["round_id"] for r in self.records))
        for rid in seen_rounds:
            ys_so_far = [
                float(r["y"])
                for r in self.records
                if r["round_id"] <= rid
            ]
            summaries.append({
                "round":  rid,
                "best_y": float(max(ys_so_far)),
                "n_obs":  len(ys_so_far),
            })
        return summaries


# --------------------------------------------------------------------------- #
#  Module-level helper                                                          #
# --------------------------------------------------------------------------- #

def _json_default(obj):
    """
    Fallback serialiser for json.dump when it encounters a numpy type.
    Registered via the ``default=`` argument, not monkey-patching.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
