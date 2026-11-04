from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from parameter_space import ChoiceDimension, ParameterSpace

# Distinct from Orchestrator's role codes (0, 1, 2) -- this is a separate
# seed namespace scoped to dataset-split derivation only, so dataset-split
# determinism never collides with warm-start/propose sub-seeds.
_ROLE_DATASET_SPLIT = 1000

LOGISTIC_REGRESSION_PARAMETER_SPACE = ParameterSpace(
    [
        ChoiceDimension("C", [0.01, 0.1, 1, 10]),
        ChoiceDimension("penalty", ["l1", "l2"]),
    ]
)

RANDOM_FOREST_PARAMETER_SPACE = ParameterSpace(
    [
        ChoiceDimension("n_estimators", [50, 100, 200]),
        ChoiceDimension("max_depth", [5, 10, 20]),
    ]
)

XGBOOST_PARAMETER_SPACE = ParameterSpace(
    [
        ChoiceDimension("eta", [0.01, 0.1, 0.2]),
        ChoiceDimension("max_depth", [3, 6, 9]),
    ]
)


def _load_breast_cancer(seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    """SMOKE-TEST dataset only. Bundled with scikit-learn: no download,
    deterministic by nature. NOT the frozen dataset for real classical-ML
    experiments -- that choice is still pending a separate review of the
    historical profiling artifacts."""
    return load_breast_cancer(return_X_y=True)


def _load_synthetic_classification(seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    """SMOKE-TEST dataset only -- see _load_breast_cancer docstring."""
    return make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=seed if seed is not None else 0,
    )


# Explicit, named dataset choices -- no dataset is silently assumed.
#
# IMPORTANT: both of these are SMOKE-TEST datasets only (bundled/synthetic,
# no download, fast). Neither is "the" classical-ML experiment dataset --
# the prior profiling work never froze one, and this module does not
# invent one either. The actual dataset for the real experiment sweep is
# a separate decision, to be made after reviewing the historical profiling
# artifacts, and will be added as its own explicit, named choice here when
# that decision is made.
DATASET_LOADERS: dict[str, Callable[[int | None], tuple[np.ndarray, np.ndarray]]] = {
    "breast_cancer": _load_breast_cancer,
    "synthetic_classification": _load_synthetic_classification,
}

# Smoke-test default ONLY -- always pass --dataset explicitly for any run
# that isn't just exercising the pipeline mechanics.
DEFAULT_SMOKE_TEST_DATASET = "breast_cancer"


@dataclass(frozen=True)
class ClassicalMLData:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


def _derive_dataset_split_seed(experiment_seed: int | None) -> int | None:
    if experiment_seed is None:
        return None
    return int(
        np.random.SeedSequence([experiment_seed, _ROLE_DATASET_SPLIT]).generate_state(1)[0]
    )


def prepare_classical_ml_data(
    dataset: str = DEFAULT_SMOKE_TEST_DATASET,
    seed: int | None = None,
    data_dir: Path | None = None,
    validation_fraction: float = 0.25,
) -> ClassicalMLData:
    """Load and deterministically split a dataset ONCE. Called by each
    make_*_objective factory at experiment-setup time -- never from inside
    a per-candidate objective call, so this cost is never counted as part
    of any single candidate's objective-evaluation time.

    `data_dir` is accepted for interface symmetry with the Hugging Face
    objective's cache_dir, even though the currently bundled/synthetic
    datasets don't need a path -- kept explicit rather than silently
    ignored so a future file-backed dataset has an obvious place to plug
    in.
    """
    if dataset not in DATASET_LOADERS:
        raise ValueError(
            f"unknown dataset {dataset!r}, expected one of {sorted(DATASET_LOADERS)}"
        )

    split_seed = _derive_dataset_split_seed(seed)
    X, y = DATASET_LOADERS[dataset](split_seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_fraction, random_state=split_seed, stratify=y
    )
    return ClassicalMLData(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)


def make_logistic_regression_objective(
    dataset: str = DEFAULT_SMOKE_TEST_DATASET,
    data_dir: Path | None = None,
    seed: int | None = None,
) -> Callable:
    data = prepare_classical_ml_data(dataset, seed, data_dir)
    # Fixed across every candidate in this run, so the ONLY thing that
    # varies between candidates is the hyperparameters being searched,
    # not incidental model-internal randomness. Deliberately NOT the
    # per-call `seed` WorkerPool derives per candidate.
    model_random_state = seed if seed is not None else 0

    def objective(params: dict, seed: int | None = None) -> float:
        model = LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver="liblinear",
            max_iter=1000,
            random_state=model_random_state,
        )
        model.fit(data.X_train, data.y_train)
        return float(model.score(data.X_val, data.y_val))

    return objective


def make_random_forest_objective(
    dataset: str = DEFAULT_SMOKE_TEST_DATASET,
    data_dir: Path | None = None,
    seed: int | None = None,
) -> Callable:
    data = prepare_classical_ml_data(dataset, seed, data_dir)
    model_random_state = seed if seed is not None else 0

    def objective(params: dict, seed: int | None = None) -> float:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=model_random_state,
        )
        model.fit(data.X_train, data.y_train)
        return float(model.score(data.X_val, data.y_val))

    return objective


def make_xgboost_objective(
    dataset: str = DEFAULT_SMOKE_TEST_DATASET,
    data_dir: Path | None = None,
    seed: int | None = None,
) -> Callable:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        # The common, specific case: the package itself isn't installed.
        raise ImportError(
            "The xgboost benchmark requires the optional 'xgboost' package. "
            "Install it with: pip install -r requirements-ml.txt"
        ) from exc
    except Exception as exc:
        # xgboost's native library can fail to load even when the package
        # IS pip-installed (e.g. xgboost.core.XGBoostError from a missing
        # libomp runtime on macOS). That error type lives inside xgboost
        # itself, so we can't name it directly without importing xgboost --
        # the very thing that's failing -- making ImportError alone
        # insufficient here. This broader catch is deliberately scoped to
        # only the import statement above (nothing else in this function),
        # and the original exception is chained so the real cause is never
        # silently relabeled as a missing dependency.
        raise ImportError(
            "The xgboost benchmark's optional 'xgboost' package is "
            "installed but failed to load (often a missing native "
            "runtime, e.g. `brew install libomp` on macOS). See the "
            "chained exception below for the underlying error."
        ) from exc

    data = prepare_classical_ml_data(dataset, seed, data_dir)
    model_random_state = seed if seed is not None else 0

    def objective(params: dict, seed: int | None = None) -> float:
        model = XGBClassifier(
            learning_rate=params["eta"],
            max_depth=params["max_depth"],
            random_state=model_random_state,
            eval_metric="logloss",
        )
        model.fit(data.X_train, data.y_train)
        return float(model.score(data.X_val, data.y_val))

    return objective
