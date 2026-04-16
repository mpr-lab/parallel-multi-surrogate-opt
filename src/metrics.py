"""
Metrics collection for a complete optimization run.

Tracks the three key metrics from the prototype spec:
  - best_y vs number of evaluations  (convergence quality)
  - GP fit time per round             (surrogate overhead)
  - RF fit time per round             (surrogate overhead)
  - Evaluation time per round         (objective overhead)
  - Proposal contribution per surrogate

All metrics are stored as plain Python types (not numpy) so they can be
serialized directly to JSON with save_json().

Reproducibility: metrics include the seed so every JSON output is
self-contained and traceable.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class RoundMetrics:
    """All metrics recorded after a single optimization round."""
    round_id:             int
    best_y:               float    # best objective value seen so far
    n_obs:                int      # total observations collected so far
    rf_fit_time:          float    # seconds to retrain the RF
    gp_fit_time:          float    # seconds to retrain the GP (0.0 if not refreshed)
    gp_was_refreshed:     bool     # was the GP actually retrained this round?
    eval_times:           List[float]   # per-candidate evaluation times
    n_rf_candidates:      int      # how many candidates came from the RF
    n_gp_candidates:      int      # how many candidates came from the GP
    cumulative_wall_time: float    # seconds elapsed since the run started


@dataclass
class RunMetrics:
    """All metrics for one complete optimization run."""
    seed:      int
    method:    str
    benchmark: str
    rounds:    List[RoundMetrics] = field(default_factory=list)

    def add_round(self, m: RoundMetrics) -> None:
        self.rounds.append(m)

    # ---------------------------------------------------------------------- #
    #  Derived summaries                                                       #
    # ---------------------------------------------------------------------- #

    def best_y_per_round(self) -> list[float]:
        """Convergence curve: best_y after each round."""
        return [r.best_y for r in self.rounds]

    def cumulative_gp_fit_time(self) -> float:
        """Total seconds spent fitting the GP across all rounds."""
        return sum(r.gp_fit_time for r in self.rounds)

    def cumulative_rf_fit_time(self) -> float:
        """Total seconds spent fitting the RF across all rounds."""
        return sum(r.rf_fit_time for r in self.rounds)

    def cumulative_eval_time(self) -> float:
        """Total seconds spent evaluating candidates."""
        return sum(sum(r.eval_times) for r in self.rounds)

    def n_gp_refreshes(self) -> int:
        """How many times the GP was actually retrained."""
        return sum(1 for r in self.rounds if r.gp_was_refreshed)

    def to_dict(self) -> dict:
        """
        Serialize to a plain dict for JSON output.

        The seed is included at the top level so every metrics file is
        self-contained and can be traced back to a specific run.
        """
        return {
            # Run-level identifiers (reproducibility metadata)
            "seed":      self.seed,
            "method":    self.method,
            "benchmark": self.benchmark,

            # Summary statistics
            "best_y_final":            self.rounds[-1].best_y if self.rounds else None,
            "best_y_per_round":        self.best_y_per_round(),
            "cumulative_gp_fit_time":  self.cumulative_gp_fit_time(),
            "cumulative_rf_fit_time":  self.cumulative_rf_fit_time(),
            "cumulative_eval_time":    self.cumulative_eval_time(),
            "n_gp_refreshes":          self.n_gp_refreshes(),

            # Per-round detail
            "rounds": [
                {
                    "round_id":             r.round_id,
                    "best_y":               r.best_y,
                    "n_obs":                r.n_obs,
                    "rf_fit_time":          r.rf_fit_time,
                    "gp_fit_time":          r.gp_fit_time,
                    "gp_was_refreshed":     r.gp_was_refreshed,
                    "mean_eval_time":       float(np.mean(r.eval_times))
                                            if r.eval_times else 0.0,
                    "n_rf_candidates":      r.n_rf_candidates,
                    "n_gp_candidates":      r.n_gp_candidates,
                    "cumulative_wall_time": r.cumulative_wall_time,
                }
                for r in self.rounds
            ],
        }
