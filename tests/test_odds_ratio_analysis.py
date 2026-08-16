import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pharmacy_reconciliation.research.odds_ratio_analysis import (
    DOCUMENTED_LOCKED_COEFFICIENTS,
    locked_logistic_odds_ratios,
)
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS


def test_all_locked_coefficients_and_odds_ratios_reproduce(
    longitudinal_observations: pd.DataFrame,
) -> None:
    result = locked_logistic_odds_ratios(longitudinal_observations)
    assert tuple(result["feature"]) == tuple(DOCUMENTED_LOCKED_COEFFICIENTS)
    assert set(result["feature"]) == set(MODEL_FEATURE_COLUMNS)
    assert np.allclose(result["odds_ratio"], np.exp(result["coefficient"]))


def test_odds_ratios_are_deterministic_and_independent_of_later_partitions(
    longitudinal_observations: pd.DataFrame,
) -> None:
    observations = longitudinal_observations
    first = locked_logistic_odds_ratios(observations)
    changed = observations.copy()
    later = pd.to_datetime(changed["observation_date"]) >= "2026-02-01"
    changed.loc[later, "previous_on_time_fill_rate"] = 999.0
    changed.loc[later, "prescription_renewal_within_window"] = 0
    assert_frame_equal(first, locked_logistic_odds_ratios(changed))
    assert_frame_equal(first, locked_logistic_odds_ratios(observations))
