from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from pharmacy_reconciliation.research.feature_analysis import tuned_logistic_pipeline
from pharmacy_reconciliation.research.permutation_analysis import (
    PERMUTATION_RANDOM_STATE,
    PERMUTATION_REPEATS,
    PERMUTATION_SCORING,
    locked_logistic_permutation_importance,
)
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS

OBSERVATIONS = Path("data/synthetic/longitudinal/refill_observations.csv")


def test_permutation_contract_and_locked_model() -> None:
    assert PERMUTATION_SCORING == "average_precision"
    assert PERMUTATION_REPEATS == 30
    assert PERMUTATION_RANDOM_STATE == 20260814
    classifier = tuned_logistic_pipeline().named_steps["classifier"]
    assert classifier.C == 0.01
    assert classifier.solver == "saga"
    assert classifier.max_iter == 5000

    result = locked_logistic_permutation_importance(pd.read_csv(OBSERVATIONS))
    assert result.validation_observations == 384
    assert set(result.importances["feature"]) == set(MODEL_FEATURE_COLUMNS)
    assert result.importances["rank"].tolist() == list(range(1, 17))


def test_permutation_results_are_deterministic_and_test_isolated() -> None:
    observations = pd.read_csv(OBSERVATIONS)
    first = locked_logistic_permutation_importance(observations)
    changed = observations.copy()
    test_mask = pd.to_datetime(changed["observation_date"]) >= "2026-04-01"
    changed.loc[test_mask, "previous_on_time_fill_rate"] = 999.0
    changed.loc[test_mask, "prescription_renewal_within_window"] = 0
    assert_frame_equal(
        first.importances,
        locked_logistic_permutation_importance(changed).importances,
    )
    assert_frame_equal(
        first.importances,
        locked_logistic_permutation_importance(observations).importances,
    )
