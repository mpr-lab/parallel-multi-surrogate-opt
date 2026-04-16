"""
Lightweight helpers for saving experiment configs and results.

Keeps all I/O logic in one place so the rest of the code just calls
save_json / load_yaml without worrying about file formats.

The most important function here is build_run_metadata(), which captures
everything needed to rerun an experiment: seed, method, benchmark,
timestamp, git hash, Python version, and the full config contents.
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
#  Config loading                                                               #
# --------------------------------------------------------------------------- #

def load_yaml(path: str) -> dict:
    """
    Load a YAML config file and return it as a plain dict.

    Args:
        path: Path to the .yaml file.

    Returns:
        Dict of config values.  Returns {} if the file is empty.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- #
#  JSON persistence                                                             #
# --------------------------------------------------------------------------- #

def save_json(data: dict, path: str) -> None:
    """
    Save a dict as a pretty-printed JSON file.  Creates parent dirs if needed.

    Args:
        data: The dict to save.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: str) -> dict:
    """Load a JSON file and return as dict."""
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Reproducibility metadata                                                     #
# --------------------------------------------------------------------------- #

def get_git_commit() -> str:
    """
    Return the current git commit hash (first 8 characters), or 'unknown'.

    This is included in every run's metadata so you can always go back to
    the exact code version that produced a given set of results.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_run_metadata(cfg: dict, config_path: str = "") -> dict:
    """
    Build a metadata dict that captures everything needed to rerun an experiment.

    This dict is saved alongside every result file.  Given this metadata,
    you can always reproduce a run exactly:
      1. Check out the git commit listed here.
      2. Use the config contents listed here (or re-point to config_path).
      3. Use the seed listed here.

    Args:
        cfg:         The full experiment config dict (already loaded from YAML).
        config_path: Path to the config file, for reference.

    Returns:
        A dict with seed, method, benchmark, timestamp, git hash, Python version,
        and the full config contents embedded.
    """
    return {
        # The three most important fields for reproducing the run:
        "seed":       cfg.get("seed", "NOT_SET"),
        "method":     cfg.get("method", "unknown"),
        "benchmark":  cfg.get("benchmark", "unknown"),

        # When was this run?
        "timestamp":  datetime.now().isoformat(),

        # Exact code version (check out this commit to reproduce)
        "git_commit": get_git_commit(),

        # Python environment
        "python_version": sys.version,

        # Where is the config file? (for reference)
        "config_path": os.path.abspath(config_path) if config_path else "",

        # The full config contents — so even if the file changes later,
        # we still know what settings were used for this run.
        "config": cfg,
    }


# --------------------------------------------------------------------------- #
#  Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _json_default(obj: Any) -> Any:
    """
    Handle types that json.dump doesn't know about natively.
    Called automatically by json.dump when it encounters an unknown type.
    """
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
