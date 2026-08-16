import pandas as pd
from pandas.testing import assert_frame_equal

from pharmacy_reconciliation.research.error_analysis import (
    ERROR_ANALYSIS_FEATURES,
    EXPECTED_CONFUSION,
    analyze_locked_errors,
)
from pharmacy_reconciliation.research.final_evaluation import LOCKED_THRESHOLD
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS


def test_locked_error_contract_and_confusion_reproduction(
    longitudinal_observations: pd.DataFrame,
) -> None:
    result = analyze_locked_errors(longitudinal_observations)
    assert LOCKED_THRESHOLD == 0.50
    assert result.confusion == EXPECTED_CONFUSION
    assert set(ERROR_ANALYSIS_FEATURES) <= set(MODEL_FEATURE_COLUMNS)
    assert "patient_id" not in MODEL_FEATURE_COLUMNS
    assert set(result.feature_statistics["outcome"]) == {"TP", "FN", "FP", "TN"}


def test_error_analysis_is_aggregate_and_deterministic(
    longitudinal_observations: pd.DataFrame,
) -> None:
    frame = longitudinal_observations
    first = analyze_locked_errors(frame)
    second = analyze_locked_errors(frame)
    assert first.confusion == second.confusion
    assert_frame_equal(first.feature_statistics, second.feature_statistics)
    assert_frame_equal(first.patient_history, second.patient_history)
    assert_frame_equal(first.outcome_probabilities, second.outcome_probabilities)
    assert_frame_equal(first.error_confidence, second.error_confidence)
    assert "patient_id" not in first.feature_statistics.columns
