"""Controlled Phase 2C baselines evaluated on Train and Validation only."""

from dataclasses import asdict, dataclass
from typing import Any, Callable

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.preparation import (
    EXCLUDED_MODEL_COLUMNS,
    PreparedPartition,
    TrainMedianPreprocessor,
    chronological_split,
)

BASELINE_RANDOM_SEED = 20260814
BASELINE_THRESHOLD = 0.50


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class BaselineResult:
    model_name: str
    configuration: dict[str, object]
    train: ClassificationMetrics
    validation: ClassificationMetrics

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselinePartitions:
    """Deliberately excludes Test from the baseline comparison interface."""

    train: PreparedPartition
    validation: PreparedPartition
    std_imputation_median: float


def prepare_baseline_partitions(observations: pd.DataFrame) -> BaselinePartitions:
    """Fit preprocessing on Train and transform Train/Validation only."""
    split = chronological_split(observations)
    preprocessor = TrainMedianPreprocessor().fit(split.train.loc[:, FEATURE_COLUMNS])

    def prepare(frame: pd.DataFrame) -> PreparedPartition:
        return PreparedPartition(
            features=preprocessor.transform(frame.loc[:, FEATURE_COLUMNS]),
            target=frame[TARGET_COLUMN].astype("int8").copy(),
            audit=frame.loc[:, EXCLUDED_MODEL_COLUMNS].copy(),
        )

    return BaselinePartitions(
        train=prepare(split.train),
        validation=prepare(split.validation),
        std_imputation_median=preprocessor.std_median,
    )


def baseline_model_factories(
    seed: int = BASELINE_RANDOM_SEED,
) -> dict[str, tuple[Callable[[], Any], dict[str, object]]]:
    """Return fixed, untuned model factories and reportable configurations."""
    configurations: dict[str, tuple[Callable[[], Any], dict[str, object]]] = {
        "Logistic Regression": (
            lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    l1_ratio=0.0, C=1.0, solver="lbfgs", max_iter=2000,
                    random_state=seed,
                )),
            ]),
            {
                "scaler": "StandardScaler",
                "penalty": "l2 (l1_ratio=0.0)",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "class_weight": None,
                "random_state": seed,
            },
        ),
        "Random Forest": (
            lambda: RandomForestClassifier(
                n_estimators=300, random_state=seed, n_jobs=1, class_weight=None
            ),
            {
                "n_estimators": 300,
                "class_weight": None,
                "random_state": seed,
                "n_jobs": 1,
            },
        ),
        "XGBoost": (
            lambda: XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=1.0,
                colsample_bytree=1.0,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
            ),
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "eval_metric": "logloss",
                "random_state": seed,
                "n_jobs": 1,
                "tree_method": "hist",
                "class_weight": None,
            },
        ),
        "LightGBM": (
            lambda: LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                random_state=seed,
                n_jobs=1,
                class_weight=None,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            ),
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "random_state": seed,
                "n_jobs": 1,
                "class_weight": None,
                "deterministic": True,
                "force_col_wise": True,
            },
        ),
    }
    return configurations


def evaluate_classifier(model: Any, partition: PreparedPartition) -> ClassificationMetrics:
    probabilities = model.predict_proba(partition.features)[:, 1]
    predictions = (probabilities >= BASELINE_THRESHOLD).astype("int8")
    matrix = confusion_matrix(partition.target, predictions, labels=[0, 1])
    return ClassificationMetrics(
        accuracy=float(accuracy_score(partition.target, predictions)),
        precision=float(precision_score(partition.target, predictions)),
        recall=float(recall_score(partition.target, predictions)),
        f1=float(f1_score(partition.target, predictions)),
        pr_auc=float(average_precision_score(partition.target, probabilities)),
        roc_auc=float(roc_auc_score(partition.target, probabilities)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
    )


def compare_baselines(partitions: BaselinePartitions) -> tuple[BaselineResult, ...]:
    """Fit on Train and evaluate Train/Validation at the fixed 0.50 threshold."""
    results = []
    for model_name, (factory, configuration) in baseline_model_factories().items():
        model = factory()
        model.fit(partitions.train.features, partitions.train.target)
        results.append(BaselineResult(
            model_name=model_name,
            configuration=configuration,
            train=evaluate_classifier(model, partitions.train),
            validation=evaluate_classifier(model, partitions.validation),
        ))
    return tuple(results)
