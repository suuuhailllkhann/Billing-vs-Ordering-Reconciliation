"""Aggregate post-Test diagnostics for the locked Logistic Regression model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pharmacy_reconciliation.research.feature_analysis import tuned_logistic_pipeline
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.final_evaluation import LOCKED_THRESHOLD
from pharmacy_reconciliation.research.preparation import chronological_split

ERROR_ANALYSIS_FEATURES = (
    "previous_on_time_fill_rate",
    "latest_refill_timing_gap_days",
    "average_previous_timing_gap_days",
    "previous_fill_count",
    "std_previous_refill_interval_days",
    "refill_interval_std_available",
    "average_previous_refill_interval_days",
    "medication_prior_fill_count",
    "prescription_age_days",
)
EXPECTED_CONFUSION = {"TN": 42, "FP": 116, "FN": 20, "TP": 198}
NEAR_THRESHOLD_MARGIN = 0.05


@dataclass(frozen=True)
class LockedErrorAnalysis:
    confusion: dict[str, int]
    feature_statistics: pd.DataFrame
    patient_history: pd.DataFrame
    outcome_probabilities: pd.DataFrame
    error_confidence: pd.DataFrame


def _prediction_frame(observations: pd.DataFrame) -> pd.DataFrame:
    split = chronological_split(observations)
    model = tuned_logistic_pipeline()
    model.fit(split.train.loc[:, FEATURE_COLUMNS], split.train[TARGET_COLUMN].astype("int8"))
    transformed = model.named_steps["preprocessor"].transform(
        split.test.loc[:, FEATURE_COLUMNS]
    )
    probabilities = model.predict_proba(split.test.loc[:, FEATURE_COLUMNS])[:, 1]
    target = split.test[TARGET_COLUMN].astype("int8").to_numpy()
    predicted = probabilities >= LOCKED_THRESHOLD
    outcomes = np.select(
        [predicted & (target == 1), ~predicted & (target == 1),
         predicted & (target == 0), ~predicted & (target == 0)],
        ["TP", "FN", "FP", "TN"],
        default="",
    )
    output = transformed.copy()
    output["patient_id"] = split.test["patient_id"].to_numpy()
    output["seen_in_train"] = split.test["patient_id"].isin(set(split.train["patient_id"])).to_numpy()
    output["probability"] = probabilities
    output["outcome"] = outcomes
    return output


def analyze_locked_errors(observations: pd.DataFrame) -> LockedErrorAnalysis:
    """Return aggregate diagnostics; never persist row-level prediction records."""
    predictions = _prediction_frame(observations)
    counts = predictions["outcome"].value_counts().reindex(
        ["TN", "FP", "FN", "TP"], fill_value=0
    ).astype(int)
    confusion = {name: int(counts[name]) for name in ("TN", "FP", "FN", "TP")}
    if confusion != EXPECTED_CONFUSION:
        raise RuntimeError(f"Locked confusion matrix mismatch: {confusion}")

    feature_statistics = (
        predictions.groupby("outcome")[list(ERROR_ANALYSIS_FEATURES)]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "feature"})
        .loc[:, ["outcome", "feature", "count", "mean", "50%", "std", "25%", "75%"]]
        .rename(columns={"50%": "median", "25%": "q25", "75%": "q75"})
    )

    history_rows = []
    for label, mask in (
        ("seen", predictions["seen_in_train"]),
        ("unseen", ~predictions["seen_in_train"]),
    ):
        group = predictions.loc[mask]
        counts = group["outcome"].value_counts()
        tp, fp = int(counts.get("TP", 0)), int(counts.get("FP", 0))
        tn, fn = int(counts.get("TN", 0)), int(counts.get("FN", 0))
        history_rows.append({
            "patient_group": label, "observations": len(group),
            "unique_patients": int(group["patient_id"].nunique()),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "error_rate": (fp + fn) / len(group),
            "false_negative_rate": fn / (tp + fn),
            "false_positive_rate": fp / (tn + fp),
        })

    probability_rows = []
    confidence_rows = []
    for outcome in ("TP", "FN", "FP", "TN"):
        values = predictions.loc[predictions["outcome"] == outcome, "probability"]
        probability_rows.append({
            "outcome": outcome, "count": len(values), "mean": values.mean(),
            "q10": values.quantile(0.10), "q25": values.quantile(0.25),
            "median": values.median(), "q75": values.quantile(0.75),
            "q90": values.quantile(0.90), "min": values.min(), "max": values.max(),
        })
        if outcome in {"FN", "FP"}:
            near = (values - LOCKED_THRESHOLD).abs() <= NEAR_THRESHOLD_MARGIN
            confidence_rows.append({
                "outcome": outcome, "errors": len(values),
                "near_threshold": int(near.sum()),
                "near_threshold_percentage": float(near.mean() * 100),
                "higher_confidence": int((~near).sum()),
                "higher_confidence_percentage": float((~near).mean() * 100),
            })
    return LockedErrorAnalysis(
        confusion=confusion,
        feature_statistics=feature_statistics,
        patient_history=pd.DataFrame(history_rows),
        outcome_probabilities=pd.DataFrame(probability_rows),
        error_confidence=pd.DataFrame(confidence_rows),
    )
