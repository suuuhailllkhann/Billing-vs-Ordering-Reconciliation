"""Validation-only threshold analysis for tuned full-feature models."""

from dataclasses import dataclass

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
from sklearn.pipeline import Pipeline

from pharmacy_reconciliation.research.feature_analysis import tuned_logistic_pipeline
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.preparation import chronological_split
from pharmacy_reconciliation.research.tree_feature_importance import tuned_tree_pipelines

THRESHOLD_GRID = (
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
)


@dataclass(frozen=True)
class ThresholdAnalysisResult:
    model_name: str
    pr_auc: float
    roc_auc: float
    thresholds: pd.DataFrame


def tuned_full_feature_pipelines() -> dict[str, Pipeline]:
    """Return the four unchanged Phase 2D tuned full-feature pipelines."""
    return {
        "Logistic Regression": tuned_logistic_pipeline(),
        **tuned_tree_pipelines(),
    }


def analyze_validation_thresholds(
    observations: pd.DataFrame,
) -> tuple[ThresholdAnalysisResult, ...]:
    """Fit on Train and analyze probabilities on Validation; Test is not evaluated."""
    split = chronological_split(observations)
    train_features = split.train.loc[:, FEATURE_COLUMNS]
    train_target = split.train[TARGET_COLUMN].astype("int8")
    validation_features = split.validation.loc[:, FEATURE_COLUMNS]
    validation_target = split.validation[TARGET_COLUMN].astype("int8")
    validation_patients = split.validation["patient_id"]
    actual_positives = int(validation_target.sum())
    observation_count = len(validation_target)

    results = []
    for model_name, model in tuned_full_feature_pipelines().items():
        model.fit(train_features, train_target)
        probabilities = model.predict_proba(validation_features)[:, 1]
        rows = []
        for threshold in THRESHOLD_GRID:
            predicted = probabilities >= threshold
            tn, fp, fn, tp = confusion_matrix(
                validation_target, predicted, labels=[0, 1]
            ).ravel()
            flagged = int(predicted.sum())
            rows.append({
                "threshold": threshold,
                "precision": precision_score(validation_target, predicted),
                "recall": recall_score(validation_target, predicted),
                "f1": f1_score(validation_target, predicted),
                "accuracy": accuracy_score(validation_target, predicted),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "flagged_observations": flagged,
                "flagged_unique_patients": int(validation_patients.loc[predicted].nunique()),
                "flagged_percentage": flagged / observation_count * 100,
                "renewals_missed_per_100_actual": int(fn) / actual_positives * 100,
                "unnecessary_followups_per_100_observations": int(fp) / observation_count * 100,
            })
        results.append(ThresholdAnalysisResult(
            model_name=model_name,
            pr_auc=float(average_precision_score(validation_target, probabilities)),
            roc_auc=float(roc_auc_score(validation_target, probabilities)),
            thresholds=pd.DataFrame(rows),
        ))
    return tuple(results)
