"""
Merge candidate lists from multiple surrogates into one evaluation batch.

The prototype uses two surrogates: RF (fast) and GP (slow).  Each proposes
some candidates independently.  This module combines them into a single list
to be sent to the worker pool.

We also track the "source" label ("rf" or "gp") for each candidate so we
can measure surrogate contribution in the final metrics.
"""

import numpy as np


def merge_candidates(
    rf_candidates: np.ndarray,
    gp_candidates: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """
    Combine RF and GP candidate lists into a single batch.

    If gp_candidates is None (e.g., the GP has not been trained yet in the
    first few rounds), only RF candidates are returned.

    Args:
        rf_candidates: Candidates from the RF,  shape (n_rf, dim).
        gp_candidates: Candidates from the GP,  shape (n_gp, dim), or None.

    Returns:
        Tuple of:
            candidates: Combined array, shape (n_rf + n_gp, dim).
            sources:    List of strings ("rf" or "gp") for each candidate.
                        Same length as the first axis of candidates.
    """
    if gp_candidates is None or len(gp_candidates) == 0:
        # GP not ready yet — use only RF candidates
        sources = ["rf"] * len(rf_candidates)
        return rf_candidates, sources

    # Stack: RF candidates first, then GP candidates
    candidates = np.vstack([rf_candidates, gp_candidates])
    sources = (
        ["rf"] * len(rf_candidates)
        + ["gp"] * len(gp_candidates)
    )
    return candidates, sources
