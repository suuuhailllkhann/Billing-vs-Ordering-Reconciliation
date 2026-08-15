from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.model_selection import TimeSeriesSplit

from pharmacy_reconciliation.research.candidate_stability import (
    CANDIDATE_NAMES,
    STABILITY_CV_SPLITS,
    VALIDATION_COMPARISON_THRESHOLDS,
    analyze_candidate_validation,
    analyze_train_cv_stability,
    fixed_candidate_pipelines,
    matched_recall_comparison,
)
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS

OBSERVATIONS_PATH = Path("data/synthetic/longitudinal/refill_observations.csv")


def test_fixed_candidates_and_five_chronological_folds() -> None:
    pipelines = fixed_candidate_pipelines()
    assert tuple(pipelines) == CANDIDATE_NAMES
    assert pipelines["Logistic Regression"].named_steps["classifier"].C == 0.01
    assert pipelines["XGBoost"].named_steps["classifier"].n_estimators == 400
    assert STABILITY_CV_SPLITS == 5
    splitter = TimeSeriesSplit(n_splits=STABILITY_CV_SPLITS)
    assert splitter.shuffle is False
    assert splitter.random_state is None

    raw = pd.read_csv(OBSERVATIONS_PATH)
    train = raw.loc[pd.to_datetime(raw["observation_date"]) < "2026-02-01"]
    transformed = pipelines["Logistic Regression"].named_steps["preprocessor"].fit_transform(
        train
    )
    assert tuple(transformed.columns) == MODEL_FEATURE_COLUMNS


def test_stability_and_validation_are_deterministic_and_test_isolated() -> None:
    observations = pd.read_csv(OBSERVATIONS_PATH)
    stability = analyze_train_cv_stability(observations)
    assert all(len(result.folds) == 5 for result in stability)
    assert all((result.folds["train_observations"] < 1714).all() for result in stability)

    validation = analyze_candidate_validation(observations)
    matched = matched_recall_comparison(validation)
    assert tuple(sorted(matched["logistic_threshold"].unique())) == VALIDATION_COMPARISON_THRESHOLDS

    changed = observations.copy()
    test_mask = pd.to_datetime(changed["observation_date"]) >= "2026-04-01"
    changed.loc[test_mask, "previous_on_time_fill_rate"] = 999.0
    changed.loc[test_mask, "prescription_renewal_within_window"] = 0
    changed_stability = analyze_train_cv_stability(changed)
    changed_validation = analyze_candidate_validation(changed)
    for original, modified in zip(stability, changed_stability, strict=True):
        assert_frame_equal(original.folds, modified.folds)
        assert_frame_equal(original.summary, modified.summary)
    for original, modified in zip(validation, changed_validation, strict=True):
        assert original.pr_auc == modified.pr_auc
        assert original.roc_auc == modified.roc_auc
        assert_frame_equal(original.thresholds, modified.thresholds)
