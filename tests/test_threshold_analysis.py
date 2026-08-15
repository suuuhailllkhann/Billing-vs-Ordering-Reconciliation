from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from pharmacy_reconciliation.research.threshold_analysis import (
    THRESHOLD_GRID,
    analyze_validation_thresholds,
    tuned_full_feature_pipelines,
)

OBSERVATIONS_PATH = Path("data/synthetic/longitudinal/refill_observations.csv")


def test_exact_threshold_grid_and_full_feature_models() -> None:
    assert THRESHOLD_GRID == (
        0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    )
    assert tuple(tuned_full_feature_pipelines()) == (
        "Logistic Regression", "Random Forest", "XGBoost", "LightGBM"
    )


def test_threshold_counts_and_test_isolation_are_deterministic() -> None:
    observations = pd.read_csv(OBSERVATIONS_PATH)
    first = analyze_validation_thresholds(observations)
    second = analyze_validation_thresholds(observations)
    for left, right in zip(first, second, strict=True):
        assert left.pr_auc == right.pr_auc
        assert left.roc_auc == right.roc_auc
        assert_frame_equal(left.thresholds, right.thresholds)
        table = left.thresholds
        assert (table["true_positives"] + table["false_negatives"] == 241).all()
        assert (table["true_negatives"] + table["false_positives"] == 143).all()
        assert (
            table["flagged_observations"]
            == table["true_positives"] + table["false_positives"]
        ).all()
        assert (table["flagged_unique_patients"] <= table["flagged_observations"]).all()

    changed_test = observations.copy()
    test_mask = pd.to_datetime(changed_test["observation_date"]) >= "2026-04-01"
    changed_test.loc[test_mask, "previous_on_time_fill_rate"] = 999.0
    changed_test.loc[test_mask, "prescription_renewal_within_window"] = 0
    changed = analyze_validation_thresholds(changed_test)
    for original, modified in zip(first, changed, strict=True):
        assert original.pr_auc == modified.pr_auc
        assert original.roc_auc == modified.roc_auc
        assert_frame_equal(original.thresholds, modified.thresholds)
