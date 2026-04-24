from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings


def logistic_regression_objective(
    x: np.ndarray | list[float],
    seed: int = 42,
    return_metadata: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """
    Logistic regression objective for hyperparameter optimization.

    Parameters
    ----------
    x :
        A 3-dimensional vector containing:
        [log10_C, l1_ratio, log10_tol]

    seed :
        Random seed for reproducible train/validation split and model behavior.

    return_metadata :
        If True, also return a metadata dictionary with decoded parameters.

    Returns
    -------
    float
        Negative validation accuracy (so lower is better for minimization).

    OR

    (float, dict)
        Objective value plus metadata if return_metadata=True.
    """
    x = np.asarray(x, dtype=float)

    if x.shape[0] != 3:
        raise ValueError(
            f"logistic_regression_objective expects 3 parameters, got shape {x.shape}"
        )

    # Unpack search variables
    log10_C, l1_ratio, log10_tol = x

    # Convert to actual logistic regression hyperparameters
    C = 10.0 ** log10_C
    tol = 10.0 ** log10_tol

    # Keep l1_ratio safely within [0, 1]
    l1_ratio = float(np.clip(l1_ratio, 0.0, 1.0))

    # Load a small, standard classification dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Reproducible train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    # Use a pipeline so scaling happens correctly
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=C,
                    l1_ratio=l1_ratio,
                    tol=tol,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )

    # Suppress convergence warnings during optimization runs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)

    # Minimization objective: lower is better
    objective_value = -float(val_acc)

    if return_metadata:
        metadata = {
            "C": C,
            "l1_ratio": l1_ratio,
            "tol": tol,
            "val_accuracy": float(val_acc),
        }
        return objective_value, metadata

    return objective_value