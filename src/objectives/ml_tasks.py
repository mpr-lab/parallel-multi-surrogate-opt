"""
Real ML task objective: XGBoost hyperparameter tuning on breast cancer data.

The objective function trains an XGBoost classifier with a given set of
hyperparameters and returns cross-validated AUC-ROC (higher = better).

We use sklearn's built-in breast cancer dataset (no download needed) so the
benchmark is fully self-contained.

Reproducibility:
    Both the XGBoost model AND the cross-validation split receive the worker
    seed (derived from base_seed + round_id + candidate_id).  This means:
    - The same candidate always gets the same train/val split.
    - The same candidate always trains the same model.
    - Passing a different seed intentionally produces a different evaluation
      (testing on different data folds).
"""

import numpy as np

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Load the dataset once at import time.
# The breast cancer dataset is part of sklearn — no internet needed.
_DATA = load_breast_cancer()
_X_DATA, _y_DATA = _DATA.data, _DATA.target


def xgboost_task(x: np.ndarray, seed: int = 42) -> float:
    """
    Train XGBoost on the breast cancer dataset and return mean CV AUC-ROC.

    Hyperparameter encoding (what each dimension of x means):
        x[0] → learning_rate  in [0.01, 0.30]
        x[1] → max_depth      in [2, 10]  (rounded to int)
        x[2] → n_estimators   in [50, 500] (rounded to int)
        x[3] → subsample      in [0.50, 1.0]

    Args:
        x:    1D array of length 4 with values in the ranges above.
        seed: Worker seed derived from derive_worker_seed().
              Controls both the XGBoost model internals and the CV splits.
              This is passed automatically by the worker pool.

    Returns:
        Mean 3-fold cross-validated AUC-ROC score (float in [0, 1]).
        Higher is better.
    """
    if not _XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not installed.  Run: pip install xgboost"
        )

    # Decode hyperparameters from the float vector
    learning_rate = float(np.clip(x[0], 0.01, 0.30))
    max_depth     = int(round(np.clip(x[1], 2, 10)))
    n_estimators  = int(round(np.clip(x[2], 50, 500)))
    subsample     = float(np.clip(x[3], 0.5, 1.0))

    model = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        random_state=seed,          # explicit seed — NOT relying on global state
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )

    # Use a seeded StratifiedKFold so the exact same train/val splits are
    # used for every evaluation with the same seed.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    scores = cross_val_score(
        model, _X_DATA, _y_DATA,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,   # no nested parallelism — the outer loop handles parallelism
    )
    return float(scores.mean())


# Search space bounds for the XGBoost task
XGBOOST_BOUNDS = [
    (0.01, 0.30),    # learning_rate
    (2.0,  10.0),    # max_depth
    (50.0, 500.0),   # n_estimators
    (0.5,  1.0),     # subsample
]

XGBOOST_DIM = 4
