import csv
import json
from pathlib import Path

REQUIRED_CANDIDATE_FIELDS = (
    "y",
    "round_id",
    "candidate_id",
    "surrogate_source",
    "candidate_rank_within_source",
    "eval_start_time",
    "eval_end_time",
    "eval_duration",
    "improved_global_best",
)

REQUIRED_METADATA_FIELDS = (
    "run_name",
    "benchmark",
    "surrogate_a_name",
    "surrogate_b_name",
)


def _parse_bool(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"expected 'True'/'False' for a boolean field, got {value!r}")
    return value == "True"


def _parse_optional_int(value: str) -> int | None:
    return None if value in ("", "None") else int(value)


def load_candidates(run_dir: Path) -> list[dict]:
    """Parse candidates.csv into typed records. Raises clearly if the file
    is missing or a required field/column is absent -- never fabricates
    values for missing data."""
    path = Path(run_dir) / "candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing candidates.csv in {run_dir}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [field for field in REQUIRED_CANDIDATE_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(f"candidates.csv is missing required field(s): {missing}")

        x_columns = sorted(
            (name for name in fieldnames if name.startswith("x_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if not x_columns:
            raise ValueError("candidates.csv has no x_* columns")

        records = []
        for row in reader:
            records.append(
                {
                    "x": tuple(float(row[col]) for col in x_columns),
                    "y": float(row["y"]),
                    "round_id": int(row["round_id"]),
                    "candidate_id": row["candidate_id"],
                    "surrogate_source": row["surrogate_source"],
                    "candidate_rank_within_source": int(row["candidate_rank_within_source"]),
                    "eval_start_time": float(row["eval_start_time"]),
                    "eval_end_time": float(row["eval_end_time"]),
                    "eval_duration": float(row["eval_duration"]),
                    "seed": _parse_optional_int(row.get("seed", "")),
                    "improved_global_best": _parse_bool(row["improved_global_best"]),
                }
            )
    return records


def load_round_metrics(run_dir: Path) -> dict:
    """Parse round_metrics.json. Raises clearly if the file is missing or
    malformed."""
    path = Path(run_dir) / "round_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing round_metrics.json in {run_dir}")
    with open(path) as f:
        data = json.load(f)
    if "rounds" not in data:
        raise ValueError("round_metrics.json missing required key: 'rounds'")
    return data


def load_metadata(run_dir: Path) -> dict:
    """Parse metadata.json. Raises clearly if the file is missing or a
    required field is absent."""
    path = Path(run_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing metadata.json in {run_dir}")
    with open(path) as f:
        data = json.load(f)
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in data]
    if missing:
        raise ValueError(f"metadata.json is missing required field(s): {missing}")
    return data
