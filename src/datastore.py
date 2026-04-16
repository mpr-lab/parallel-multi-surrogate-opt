"""
Shared dataset and result logger.

The DataStore holds ALL (X, y) observations collected during an optimization
run.  It also handles writing results to disk.

Design choices:
  - We write to CSV after every update so partial results are never lost.
  - The run_name (which includes the seed) is embedded in every output
    filename and in every CSV row, so results files are self-identifying.
  - JSON metadata (saved separately) contains everything needed to rerun.

Reproducibility: the seed is propagated into both the CSV rows (via run_name)
and the JSON metadata, so every output file carries its own provenance.
"""

import os
import csv
import json
from typing import Optional
import numpy as np


class DataStore:
    """
    Storage for all (config, value) observations from a BO run.

    Attributes:
        X: All tried configurations, shape (n_obs, dim).
        y: All observed objective values, shape (n_obs,).
    """

    def __init__(self, dim: int, output_dir: str, run_name: str):
        """
        Args:
            dim:        Number of hyperparameter dimensions.
            output_dir: Directory for output files (created if needed).
            run_name:   Unique name for this run, e.g.
                        "parallel_msbo_branin_seed42".
                        Used in output filenames and CSV row labels.
        """
        self.dim        = dim
        self.output_dir = output_dir
        self.run_name   = run_name

        # Internal storage — append-only lists for efficiency
        self._X:        list[np.ndarray] = []
        self._y:        list[float]      = []
        self._round_ids:   list[int]     = []
        self._sources:     list[str]     = []   # "rf", "gp", or "random"

        os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------------- #
    #  Read-only properties                                                    #
    # ---------------------------------------------------------------------- #

    @property
    def X(self) -> np.ndarray:
        """All tried configurations, shape (n_obs, dim)."""
        if not self._X:
            return np.empty((0, self.dim))
        return np.array(self._X)

    @property
    def y(self) -> np.ndarray:
        """All observed objective values, shape (n_obs,)."""
        return np.array(self._y, dtype=float)

    @property
    def best_y(self) -> float:
        """Best (highest) objective value seen so far."""
        if not self._y:
            return float("-inf")
        return float(np.max(self._y))

    @property
    def n_obs(self) -> int:
        """Number of observations collected so far."""
        return len(self._y)

    # ---------------------------------------------------------------------- #
    #  Insertion                                                               #
    # ---------------------------------------------------------------------- #

    def add_observations(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        round_id: int,
        source: str = "unknown",
    ) -> None:
        """
        Add new (config, value) pairs to the dataset.

        Args:
            X_new:    New configurations, shape (n_new, dim) or (dim,).
            y_new:    New objective values, shape (n_new,) or scalar.
            round_id: Which optimization round these came from (-1 = initial).
            source:   Which surrogate proposed them: "rf", "gp", or "random".
        """
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(np.asarray(y_new, dtype=float))

        for i in range(len(y_new)):
            self._X.append(X_new[i].copy())
            self._y.append(float(y_new[i]))
            self._round_ids.append(round_id)
            self._sources.append(source)

    # ---------------------------------------------------------------------- #
    #  Persistence                                                             #
    # ---------------------------------------------------------------------- #

    def save_results_csv(self) -> str:
        """
        Write all observations to a CSV file.

        Columns: x0, x1, ..., x(dim-1), y, round_id, source, run_name, seed.

        The seed is embedded in run_name AND as its own column so it is
        easy to filter by seed when aggregating results from multiple runs.

        Returns:
            Absolute path to the saved file.
        """
        path = os.path.join(self.output_dir, f"{self.run_name}_results.csv")

        # Extract seed from run_name if present (format: *_seed<N>)
        seed_str = ""
        if "_seed" in self.run_name:
            seed_str = self.run_name.split("_seed")[-1]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            x_cols = [f"x{i}" for i in range(self.dim)]
            writer.writerow(x_cols + ["y", "round_id", "source", "run_name", "seed"])

            # Data rows
            for i in range(self.n_obs):
                row = (
                    self._X[i].tolist()
                    + [self._y[i], self._round_ids[i],
                       self._sources[i], self.run_name, seed_str]
                )
                writer.writerow(row)

        return os.path.abspath(path)

    def save_metadata_json(self, metadata: dict) -> str:
        """
        Write run-level metadata (the 'rerun recipe') to a JSON file.

        Adds a few summary fields (best_y, n_observations) to the metadata
        dict before writing, so the JSON is self-contained.

        Args:
            metadata: Dict from build_run_metadata() in serialization.py.

        Returns:
            Absolute path to the saved file.
        """
        path = os.path.join(self.output_dir, f"{self.run_name}_metadata.json")
        metadata = {
            **metadata,
            "final_best_y":    self.best_y,
            "n_observations":  self.n_obs,
        }
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)
        return os.path.abspath(path)

    def get_convergence_curve(self) -> list[dict]:
        """
        Return per-round best-y values for plotting a convergence curve.

        Returns a list of dicts: [{"round": r, "best_y": v, "n_obs": n}, ...]
        """
        if not self._y:
            return []

        summaries = []
        for r in sorted(set(self._round_ids)):
            ys_up_to_r = [
                self._y[i] for i in range(self.n_obs)
                if self._round_ids[i] <= r
            ]
            summaries.append({
                "round":  r,
                "best_y": float(max(ys_up_to_r)),
                "n_obs":  len(ys_up_to_r),
            })
        return summaries


def _json_default(obj):
    """Handle numpy types during JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")
