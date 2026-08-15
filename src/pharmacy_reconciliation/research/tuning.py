"""Train-only time-aware hyperparameter searches for Phase 2D."""

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pharmacy_reconciliation.research.baselines import (
    BASELINE_RANDOM_SEED,
    evaluate_classifier,
)
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.preparation import (
    EXCLUDED_MODEL_COLUMNS,
    MISSING_STD_COLUMN,
    MISSING_STD_INDICATOR,
    MODEL_FEATURE_COLUMNS,
    PreparedPartition,
    chronological_split,
)

TUNING_SCORING = "average_precision"
TUNING_CV_SPLITS = 5
TUNING_TRIALS = {
    "Logistic Regression": 10,
    "Random Forest": 30,
    "XGBoost": 40,
    "LightGBM": 40,
}


class FoldLocalRefillPreprocessor(BaseEstimator, TransformerMixin):
    """Learn the refill-interval median inside each CV training fold."""

    def fit(self, features: pd.DataFrame, target: Any = None) -> "FoldLocalRefillPreprocessor":
        del target
        median = features[MISSING_STD_COLUMN].median(skipna=True)
        if pd.isna(median):
            raise ValueError("Training fold has no available refill-interval standard deviation.")
        self.std_median_ = float(median)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "std_median_"):
            raise RuntimeError("Preprocessor must be fitted before transformation.")
        output = features.loc[:, FEATURE_COLUMNS].copy()
        output[MISSING_STD_INDICATOR] = output[MISSING_STD_COLUMN].notna().astype("int8")
        output[MISSING_STD_COLUMN] = output[MISSING_STD_COLUMN].fillna(self.std_median_)
        return output.loc[:, MODEL_FEATURE_COLUMNS]


@dataclass(frozen=True)
class RawTuningPartitions:
    """Raw Train/Validation only; Test is deliberately absent."""

    train: PreparedPartition
    validation: PreparedPartition


@dataclass(frozen=True)
class TunedModelResult:
    model_name: str
    trials: int
    best_cv_pr_auc: float
    best_parameters: dict[str, object]
    train: Any
    validation: Any

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def prepare_tuning_partitions(observations: pd.DataFrame) -> RawTuningPartitions:
    """Return chronologically sorted raw Train/Validation without accessing Test values."""
    split = chronological_split(observations)

    def partition(frame: pd.DataFrame) -> PreparedPartition:
        return PreparedPartition(
            features=frame.loc[:, FEATURE_COLUMNS].copy(),
            target=frame[TARGET_COLUMN].astype("int8").copy(),
            audit=frame.loc[:, EXCLUDED_MODEL_COLUMNS].copy(),
        )

    return RawTuningPartitions(partition(split.train), partition(split.validation))


def tuning_searches(
    seed: int = BASELINE_RANDOM_SEED,
) -> dict[str, RandomizedSearchCV]:
    """Build the four specified deterministic randomized searches."""
    cv = TimeSeriesSplit(n_splits=TUNING_CV_SPLITS)

    def pipeline(classifier: Any, scale: bool = False) -> Pipeline:
        steps: list[tuple[str, Any]] = [("preprocessor", FoldLocalRefillPreprocessor())]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        return Pipeline(steps)

    specifications: dict[str, tuple[Pipeline, dict[str, list[object]]]] = {
        "Logistic Regression": (
            pipeline(LogisticRegression(
                solver="saga", max_iter=5000, random_state=seed, class_weight=None,
                l1_ratio=0.0,
            ), scale=True),
            {
                "classifier__C": [0.01, 0.1, 0.5, 1, 2, 5, 10],
                # scikit-learn >=1.8 maps 1.0 to L1 and 0.0 to L2.
                "classifier__l1_ratio": [1.0, 0.0],
            },
        ),
        "Random Forest": (
            pipeline(RandomForestClassifier(
                random_state=seed, n_jobs=1, class_weight=None
            )),
            {
                "classifier__n_estimators": [200, 400, 600],
                "classifier__max_depth": [3, 5, 8, 12, None],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 5, 10],
                "classifier__max_features": ["sqrt", "log2", 0.5, 1.0],
            },
        ),
        "XGBoost": (
            pipeline(XGBClassifier(
                eval_metric="logloss", random_state=seed, n_jobs=1, tree_method="hist"
            )),
            {
                "classifier__n_estimators": [150, 250, 400, 600],
                "classifier__learning_rate": [0.02, 0.05, 0.1],
                "classifier__max_depth": [2, 3, 4, 5],
                "classifier__min_child_weight": [1, 3, 5, 10],
                "classifier__subsample": [0.7, 0.85, 1.0],
                "classifier__colsample_bytree": [0.7, 0.85, 1.0],
                "classifier__reg_alpha": [0, 0.01, 0.1, 0.5],
                "classifier__reg_lambda": [1, 2, 5, 10],
            },
        ),
        "LightGBM": (
            pipeline(LGBMClassifier(
                random_state=seed,
                n_jobs=1,
                class_weight=None,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )),
            {
                "classifier__n_estimators": [150, 250, 400, 600],
                "classifier__learning_rate": [0.02, 0.05, 0.1],
                "classifier__num_leaves": [7, 15, 31, 63],
                "classifier__max_depth": [3, 5, 8, -1],
                "classifier__min_child_samples": [10, 20, 40, 80],
                "classifier__subsample": [0.7, 0.85, 1.0],
                "classifier__colsample_bytree": [0.7, 0.85, 1.0],
                "classifier__reg_alpha": [0, 0.01, 0.1, 0.5],
                "classifier__reg_lambda": [0, 1, 5, 10],
            },
        ),
    }
    return {
        name: RandomizedSearchCV(
            estimator=estimator,
            param_distributions=parameters,
            n_iter=TUNING_TRIALS[name],
            scoring=TUNING_SCORING,
            cv=cv,
            refit=True,
            random_state=seed,
            n_jobs=1,
            return_train_score=False,
        )
        for name, (estimator, parameters) in specifications.items()
    }


def _clean_parameters(parameters: dict[str, object]) -> dict[str, object]:
    cleaned = {
        key.removeprefix("classifier__"): value for key, value in parameters.items()
    }
    l1_ratio = cleaned.pop("l1_ratio", None)
    if l1_ratio is not None:
        cleaned["penalty"] = "l1" if l1_ratio == 1.0 else "l2"
    return cleaned


def tune_models(partitions: RawTuningPartitions) -> tuple[TunedModelResult, ...]:
    """Tune on Train only, refit on full Train, and compare on Validation."""
    results = []
    for model_name, search in tuning_searches().items():
        search.fit(partitions.train.features, partitions.train.target)
        best_model = search.best_estimator_
        results.append(TunedModelResult(
            model_name=model_name,
            trials=TUNING_TRIALS[model_name],
            best_cv_pr_auc=float(search.best_score_),
            best_parameters=_clean_parameters(search.best_params_),
            train=evaluate_classifier(best_model, partitions.train),
            validation=evaluate_classifier(best_model, partitions.validation),
        ))
    return tuple(results)
