"""Train-only stability and Validation operating-point comparison for two candidates."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from pharmacy_reconciliation.research.feature_analysis import tuned_logistic_pipeline
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.preparation import chronological_split
from pharmacy_reconciliation.research.threshold_analysis import THRESHOLD_GRID
from pharmacy_reconciliation.research.tree_feature_importance import tuned_tree_pipelines

STABILITY_CV_SPLITS = 5
STABILITY_THRESHOLD = 0.50
VALIDATION_COMPARISON_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55)
CANDIDATE_NAMES = ("Logistic Regression", "XGBoost")


@dataclass(frozen=True)
class CandidateStabilityResult:
    model_name: str
    folds: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class CandidateValidationResult:
    model_name: str
    pr_auc: float
    roc_auc: float
    thresholds: pd.DataFrame


def fixed_candidate_pipelines() -> dict[str, Pipeline]:
    """Return only the two unchanged tuned full-feature candidate pipelines."""
    return {
        "Logistic Regression": tuned_logistic_pipeline(),
        "XGBoost": tuned_tree_pipelines()["XGBoost"],
    }


def _classification_row(
    target: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(target, predicted)),
        "recall": float(recall_score(target, predicted)),
        "f1": float(f1_score(target, predicted)),
        "accuracy": float(accuracy_score(target, predicted)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def analyze_train_cv_stability(
    observations: pd.DataFrame,
) -> tuple[CandidateStabilityResult, ...]:
    """Evaluate fixed candidates on five chronological expanding folds within Train."""
    train = chronological_split(observations).train
    features = train.loc[:, FEATURE_COLUMNS]
    target = train[TARGET_COLUMN].astype("int8")
    splitter = TimeSeriesSplit(n_splits=STABILITY_CV_SPLITS)
    results = []
    for model_name, model in fixed_candidate_pipelines().items():
        rows = []
        for fold, (fit_indices, validation_indices) in enumerate(
            splitter.split(features), start=1
        ):
            fold_target = target.iloc[validation_indices]
            model.fit(features.iloc[fit_indices], target.iloc[fit_indices])
            probabilities = model.predict_proba(features.iloc[validation_indices])[:, 1]
            row: dict[str, float | int] = {
                "fold": fold,
                "train_observations": len(fit_indices),
                "validation_observations": len(validation_indices),
                "positive_rate": float(fold_target.mean()),
                "pr_auc": float(average_precision_score(fold_target, probabilities)),
                "roc_auc": float(roc_auc_score(fold_target, probabilities)),
            }
            row.update(_classification_row(
                fold_target, probabilities, STABILITY_THRESHOLD
            ))
            rows.append(row)
        folds = pd.DataFrame(rows)
        summary_columns = (
            "positive_rate", "pr_auc", "roc_auc", "precision", "recall",
            "f1", "accuracy", "false_positives", "false_negatives",
        )
        summary = folds.loc[:, summary_columns].agg(
            ["mean", "std", "min", "max"]
        ).transpose().reset_index(names="metric")
        results.append(CandidateStabilityResult(model_name, folds, summary))
    return tuple(results)


def analyze_candidate_validation(
    observations: pd.DataFrame,
) -> tuple[CandidateValidationResult, ...]:
    """Fit on Train and compare fixed-grid operating points on Validation only."""
    split = chronological_split(observations)
    train_features = split.train.loc[:, FEATURE_COLUMNS]
    train_target = split.train[TARGET_COLUMN].astype("int8")
    validation_features = split.validation.loc[:, FEATURE_COLUMNS]
    validation_target = split.validation[TARGET_COLUMN].astype("int8")
    validation_count = len(validation_target)
    actual_positives = int(validation_target.sum())
    results = []
    for model_name, model in fixed_candidate_pipelines().items():
        model.fit(train_features, train_target)
        probabilities = model.predict_proba(validation_features)[:, 1]
        rows = []
        for threshold in THRESHOLD_GRID:
            row = {"threshold": threshold}
            row.update(_classification_row(validation_target, probabilities, threshold))
            flagged = row["true_positives"] + row["false_positives"]
            row.update({
                "flagged_observations": flagged,
                "flagged_percentage": flagged / validation_count * 100,
                "renewals_missed_per_100_actual": (
                    row["false_negatives"] / actual_positives * 100
                ),
                "unnecessary_followups_per_100_observations": (
                    row["false_positives"] / validation_count * 100
                ),
            })
            rows.append(row)
        results.append(CandidateValidationResult(
            model_name=model_name,
            pr_auc=float(average_precision_score(validation_target, probabilities)),
            roc_auc=float(roc_auc_score(validation_target, probabilities)),
            thresholds=pd.DataFrame(rows),
        ))
    return tuple(results)


def matched_recall_comparison(
    validation_results: tuple[CandidateValidationResult, ...],
) -> pd.DataFrame:
    """Pair specified Logistic points with the closest fixed-grid XGBoost recall."""
    by_name = {result.model_name: result.thresholds for result in validation_results}
    logistic = by_name["Logistic Regression"]
    xgboost = by_name["XGBoost"]
    rows = []
    for threshold in VALIDATION_COMPARISON_THRESHOLDS:
        logistic_row = logistic.loc[logistic["threshold"] == threshold].iloc[0]
        distance = (xgboost["recall"] - logistic_row["recall"]).abs()
        xgboost_row = xgboost.loc[distance.idxmin()]
        for model_name, row in (
            ("Logistic Regression", logistic_row), ("XGBoost", xgboost_row)
        ):
            rows.append({
                "logistic_threshold": threshold,
                "model_name": model_name,
                "threshold": row["threshold"],
                "recall": row["recall"],
                "precision": row["precision"],
                "f1": row["f1"],
                "false_positives": int(row["false_positives"]),
                "false_negatives": int(row["false_negatives"]),
                "flagged_observations": int(row["flagged_observations"]),
                "flagged_percentage": row["flagged_percentage"],
            })
    return pd.DataFrame(rows)
