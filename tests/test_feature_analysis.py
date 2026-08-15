from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from pharmacy_reconciliation.research.feature_analysis import (
    TUNED_LOGISTIC_C,
    fit_tuned_logistic_coefficients,
    tuned_logistic_pipeline,
)
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS

OBSERVATIONS_PATH = Path("data/synthetic/longitudinal/refill_observations.csv")


def test_tuned_logistic_coefficients_cover_every_final_input() -> None:
    observations = pd.read_csv(OBSERVATIONS_PATH)
    coefficients = fit_tuned_logistic_coefficients(observations)

    assert set(coefficients["feature"]) == set(MODEL_FEATURE_COLUMNS)
    assert len(coefficients) == len(MODEL_FEATURE_COLUMNS)
    assert coefficients["absolute_coefficient"].is_monotonic_decreasing
    assert set(coefficients["direction"]) <= {"positive", "negative"}

    classifier = tuned_logistic_pipeline().named_steps["classifier"]
    assert classifier.C == TUNED_LOGISTIC_C
    assert classifier.solver == "saga"
    assert classifier.l1_ratio == 0.0


def test_coefficient_fit_is_invariant_to_test_period_values() -> None:
    observations = pd.read_csv(OBSERVATIONS_PATH)
    changed_test = observations.copy()
    test_mask = pd.to_datetime(changed_test["observation_date"]) >= "2026-04-01"
    changed_test.loc[test_mask, "previous_on_time_fill_rate"] = 999.0
    changed_test.loc[test_mask, "prescription_renewal_within_window"] = 0

    original = fit_tuned_logistic_coefficients(observations)
    changed = fit_tuned_logistic_coefficients(changed_test)
    assert_frame_equal(original, changed)
