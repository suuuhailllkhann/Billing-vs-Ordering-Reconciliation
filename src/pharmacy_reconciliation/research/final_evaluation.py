"""One-time locked Test evaluation for the final Logistic Regression candidate."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from pharmacy_reconciliation.research.feature_analysis import tuned_logistic_pipeline
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.preparation import chronological_split

LOCKED_THRESHOLD = 0.50
FINAL_FITTING_POLICY = "Train only"


@dataclass(frozen=True)
class FinalEvaluationResult:
    overall: dict[str, float | int]
    seen_patients: dict[str, float | int]
    unseen_patients: dict[str, float | int]
    validation: dict[str, float | int]


def _metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float | int]:
    target = frame[TARGET_COLUMN].astype("int8")
    predicted = probabilities >= LOCKED_THRESHOLD
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    observations = len(frame)
    positives = int(target.sum())
    flagged = int(predicted.sum())
    precision = int(tp) / flagged if flagged else 0.0
    recall = int(tp) / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "observations": observations,
        "unique_patients": int(frame["patient_id"].nunique()),
        "positives": positives,
        "negatives": observations - positives,
        "positive_rate": float(target.mean()),
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(average_precision_score(target, probabilities)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "flagged_observations": flagged,
        "flagged_percentage": flagged / observations * 100,
        "renewals_missed_per_100_actual": int(fn) / positives * 100,
        "unnecessary_followups_per_100_observations": int(fp) / observations * 100,
    }


def evaluate_locked_test(observations: pd.DataFrame) -> FinalEvaluationResult:
    """Fit locked preprocessing/model on Train and evaluate Validation and Test once."""
    split = chronological_split(observations)
    model = tuned_logistic_pipeline()
    model.fit(split.train.loc[:, FEATURE_COLUMNS], split.train[TARGET_COLUMN].astype("int8"))
    validation_probabilities = model.predict_proba(
        split.validation.loc[:, FEATURE_COLUMNS]
    )[:, 1]
    test_probabilities = model.predict_proba(split.test.loc[:, FEATURE_COLUMNS])[:, 1]

    seen_ids = set(split.train["patient_id"])
    seen_mask = split.test["patient_id"].isin(seen_ids).to_numpy()
    return FinalEvaluationResult(
        overall=_metrics(split.test, test_probabilities),
        seen_patients=_metrics(split.test.loc[seen_mask], test_probabilities[seen_mask]),
        unseen_patients=_metrics(split.test.loc[~seen_mask], test_probabilities[~seen_mask]),
        validation=_metrics(split.validation, validation_probabilities),
    )
