import csv
import json


class MetricsLog:
    """Round-level optimization metrics, separate from DataStore's
    per-candidate records.

    Surrogate-agnostic: fields use fixed positional roles
    (`surrogate_a_*` / `surrogate_b_*`), never a model-family-specific key
    like `rf_fit_time` or `gp_fit_time`. The actual model identity is
    carried separately in `surrogate_a_name` / `surrogate_b_name`.
    """

    def __init__(self):
        self.warm_start: dict | None = None
        self.rounds: list[dict] = []

    def record_warm_start(
        self,
        n_initial: int,
        evaluation_wall_time: float,
        total_round_wall_time: float,
    ) -> None:
        self.warm_start = {
            "round_id": 0,
            "n_initial": n_initial,
            "evaluation_wall_time": evaluation_wall_time,
            "total_round_wall_time": total_round_wall_time,
        }

    def record_round(
        self,
        round_id: int,
        best_y_so_far: float,
        surrogate_a_name: str,
        surrogate_b_name: str,
        surrogate_a_fit_time: float,
        surrogate_b_fit_time: float,
        surrogate_a_proposal_time: float,
        surrogate_b_proposal_time: float,
        surrogate_a_candidates_contributed: int,
        surrogate_b_candidates_contributed: int,
        evaluation_wall_time: float,
        total_round_wall_time: float,
    ) -> None:
        self.rounds.append(
            {
                "round_id": round_id,
                "best_y_so_far": best_y_so_far,
                "surrogate_a_name": surrogate_a_name,
                "surrogate_b_name": surrogate_b_name,
                "surrogate_a_fit_time": surrogate_a_fit_time,
                "surrogate_b_fit_time": surrogate_b_fit_time,
                "surrogate_a_proposal_time": surrogate_a_proposal_time,
                "surrogate_b_proposal_time": surrogate_b_proposal_time,
                "surrogate_a_candidates_contributed": surrogate_a_candidates_contributed,
                "surrogate_b_candidates_contributed": surrogate_b_candidates_contributed,
                "evaluation_wall_time": evaluation_wall_time,
                "total_round_wall_time": total_round_wall_time,
            }
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"warm_start": self.warm_start, "rounds": self.rounds}, f, indent=2)

    def to_csv(self, path: str) -> None:
        if not self.rounds:
            raise ValueError("no round metrics recorded yet")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rounds[0].keys()))
            writer.writeheader()
            writer.writerows(self.rounds)
