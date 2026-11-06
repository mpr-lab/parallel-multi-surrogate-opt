from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from parameter_space import ChoiceDimension, ParameterSpace

# Historical recipe recovered from hpc-bayes-opt's BoTorch/Optuna Logistic
# Regression notebooks and the Optuna XGBoost notebook (all three loaded
# HIGGS identically and produced real result CSVs). Random Forest was
# never historically run against HIGGS in that repo -- a DeepHyper
# template notebook's comments state the shared-pipeline intent ("should
# be fully reusable for xgboost and rf") but it was never completed for
# RF. Treat this recipe's reuse for RF as this task's explicit decision,
# not a rediscovered historical fact.
HIGGS_FILENAME = "HIGGS.csv"
HIGGS_KAGGLE_DATASET = "erikbiswas/higgs-uci-dataset"  # provenance only -- not auto-downloaded
HIGGS_SUBSAMPLE_SIZE = 50_000
HIGGS_VALIDATION_FRACTION = 0.2  # historical 80/20 split, recovered as-is

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

# "Expanded" search spaces for the matched 15-evaluation experiment
# protocol (n_initial=5, n_rounds=5, batch_size=2). These are SEPARATE
# ParameterSpace constants, not a change to the historical-grid spaces
# above -- logistic_regression/random_forest/xgboost keep their original
# coarse grids unchanged. Each expanded space reuses the *same* decoded
# parameter names ("C"/"penalty", "n_estimators"/"max_depth",
# "eta"/"max_depth") as its historical counterpart, so the existing
# make_*_objective factories below are reused completely unchanged --
# only the ParameterSpace paired with them (in run_experiment.py's
# registry) differs.

# 40 log-spaced values from 1e-4 to 1e3 (inclusive), 2 penalties -> 80.
LOGISTIC_REGRESSION_EXPANDED_C_VALUES = np.logspace(-4, 3, 40).tolist()
LOGISTIC_REGRESSION_EXPANDED_PARAMETER_SPACE = ParameterSpace(
    [
        ChoiceDimension("C", LOGISTIC_REGRESSION_EXPANDED_C_VALUES),
        ChoiceDimension("penalty", ["l1", "l2"]),
    ]
)

# 19 n_estimators values x 8 max_depth values -> 152.
RANDOM_FOREST_EXPANDED_PARAMETER_SPACE = ParameterSpace(
    [
        ChoiceDimension(
            "n_estimators",
            [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500],
        ),
        ChoiceDimension("max_depth", [3, 5, 7, 10, 15, 20, 25, 30]),
    ]
)

# 12 log-spaced values from 0.01 to 0.3 (inclusive), 8 max_depth values -> 96.
XGBOOST_EXPANDED_ETA_VALUES = np.logspace(np.log10(0.01), np.log10(0.3), 12).tolist()
XGBOOST_EXPANDED_PARAMETER_SPACE = ParameterSpace(
    [
        ChoiceDimension("eta", XGBOOST_EXPANDED_ETA_VALUES),
        ChoiceDimension("max_depth", [3, 4, 5, 6, 7, 8, 9, 10]),
    ]
)


def _load_breast_cancer(seed: int | None, data_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """SMOKE-TEST dataset only. Bundled with scikit-learn: no download,
    deterministic by nature. NOT the frozen dataset for real classical-ML
    experiments -- HIGGS is (see _load_higgs)."""
    return load_breast_cancer(return_X_y=True)


def _load_synthetic_classification(
    seed: int | None, data_dir: Path | None
) -> tuple[np.ndarray, np.ndarray]:
    """SMOKE-TEST dataset only -- see _load_breast_cancer docstring."""
    return make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=seed if seed is not None else 0,
    )


def _load_higgs(seed: int | None, data_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """The frozen real-experiment dataset for Logistic Regression, Random
    Forest, and XGBoost (recovered recipe -- see module-level comment and
    HIGGS_* constants).

    Does NOT auto-download: HIGGS.csv is a multi-GB file and the
    historical download path (kagglehub) needs Kaggle API credentials,
    neither of which belongs inside a library import or a test. Download
    it once yourself (e.g. `kagglehub.dataset_download(HIGGS_KAGGLE_DATASET)`
    or the Kaggle CLI) and point `--data-dir` at the directory containing
    HIGGS.csv -- e.g. a path under /work on Unity.
    """
    if data_dir is None:
        raise ValueError(
            "the 'higgs' dataset requires --data-dir to point at a directory "
            f"containing {HIGGS_FILENAME} (not downloaded automatically)"
        )

    higgs_path = Path(data_dir) / HIGGS_FILENAME
    if not higgs_path.exists():
        raise FileNotFoundError(
            f"{higgs_path} not found. Download it once (e.g. "
            f"kagglehub.dataset_download({HIGGS_KAGGLE_DATASET!r}) or the "
            f"Kaggle CLI) and place {HIGGS_FILENAME} at --data-dir."
        )

    import pandas as pd  # lazy: only the higgs path needs pandas

    data = pd.read_csv(higgs_path, header=None)
    X_full = data.iloc[:, 1:].values
    y_full = data.iloc[:, 0].values

    subsample_seed = seed if seed is not None else 0
    X, _, y, _ = train_test_split(
        X_full,
        y_full,
        train_size=min(HIGGS_SUBSAMPLE_SIZE, len(y_full)),
        random_state=subsample_seed,
        stratify=y_full,
    )
    return X, y


# Explicit, named dataset choices -- no dataset is silently assumed.
#
# "breast_cancer"/"synthetic_classification" are SMOKE-TEST datasets only
# (bundled/synthetic, no download, fast). "higgs" is the frozen dataset
# for the real classical-ML experiment sweep (Logistic Regression, Random
# Forest, XGBoost), recovered from the historical profiling artifacts.
DATASET_LOADERS: dict[str, Callable[[int | None, Path | None], tuple[np.ndarray, np.ndarray]]] = {
    "breast_cancer": _load_breast_cancer,
    "synthetic_classification": _load_synthetic_classification,
    "higgs": _load_higgs,
}

# Per-dataset validation fraction. HIGGS uses the historically-recovered
# 80/20 split; datasets with no historical split convention keep this
# framework's own smoke-test default.
DATASET_VALIDATION_FRACTIONS: dict[str, float] = {
    "higgs": HIGGS_VALIDATION_FRACTION,
}
DEFAULT_VALIDATION_FRACTION = 0.25

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
    validation_fraction: float | None = None,
) -> ClassicalMLData:
    """Load and deterministically split a dataset ONCE -- the one shared
    pipeline used identically by Logistic Regression, Random Forest, and
    XGBoost. Called by each make_*_objective factory at experiment-setup
    time -- never from inside a per-candidate objective call, so this
    cost is never counted as part of any single candidate's
    objective-evaluation time.

    `data_dir` is where a file-backed dataset (currently only "higgs")
    is read from -- never a hard-coded path, so it can point at e.g.
    /work on Unity. Ignored by the bundled/synthetic smoke-test datasets.

    `validation_fraction`, if not given, is resolved per-dataset via
    DATASET_VALIDATION_FRACTIONS (HIGGS uses the historically-recovered
    0.2; everything else keeps this framework's 0.25 smoke-test default)
    -- explicitly passing a value always overrides that lookup.
    """
    if dataset not in DATASET_LOADERS:
        raise ValueError(
            f"unknown dataset {dataset!r}, expected one of {sorted(DATASET_LOADERS)}"
        )
    if validation_fraction is None:
        validation_fraction = DATASET_VALIDATION_FRACTIONS.get(
            dataset, DEFAULT_VALIDATION_FRACTION
        )

    split_seed = _derive_dataset_split_seed(seed)
    X, y = DATASET_LOADERS[dataset](split_seed, data_dir)
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
    # Logistic Regression is scale-sensitive (regularized linear model);
    # Random Forest/XGBoost are not and are deliberately left unscaled
    # (see make_random_forest_objective / make_xgboost_objective) -- this
    # matches the historical HIGGS notebooks, which only ever scaled for
    # Logistic Regression. Scaling happens once here (factory/setup time),
    # not inside the per-candidate closure below.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(data.X_train)
    X_val_scaled = scaler.transform(data.X_val)
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
        model.fit(X_train_scaled, data.y_train)
        return float(model.score(X_val_scaled, data.y_val))

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
            # Explicit, not sklearn's default: sklearn's n_jobs=None means
            # 1 core, so this is a no-op on the actual parallelism (kept
            # for symmetry with the XGBoost fix below and to make the
            # single-core intent explicit rather than incidental).
            n_jobs=1,
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
            # Explicit, not XGBoost's default: XGBoost's n_jobs=None
            # resolves to using all available CPU threads (nthread=-1
            # internally), which otherwise confounds wall-clock timing
            # comparisons with the WorkerPool's own worker count.
            n_jobs=1,
        )
        model.fit(data.X_train, data.y_train)
        return float(model.score(data.X_val, data.y_val))

    return objective
