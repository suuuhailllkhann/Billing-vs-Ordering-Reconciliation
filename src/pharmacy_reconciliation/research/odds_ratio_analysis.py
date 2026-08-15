"""Train-only standardized odds ratios for the locked Logistic Regression."""

import numpy as np
import pandas as pd

from pharmacy_reconciliation.research.feature_analysis import fit_tuned_logistic_coefficients

NEUTRAL_COEFFICIENT_TOLERANCE = 0.01
DOCUMENTED_LOCKED_COEFFICIENTS = {
    "previous_on_time_fill_rate": 0.49936279,
    "average_previous_timing_gap_days": -0.13421066,
    "previous_fill_count": 0.11387149,
    "previous_early_fill_rate": -0.06458892,
    "latest_refill_timing_gap_days": -0.05895071,
    "refill_interval_std_available": -0.05106063,
    "medication_prior_fill_count": 0.04926908,
    "average_previous_refill_interval_days": -0.04845641,
    "medication_prior_average_days_supply": 0.04333492,
    "current_refill_number": -0.01875323,
    "current_quantity_billed": 0.01754541,
    "medication_prior_average_quantity": -0.01658121,
    "std_previous_refill_interval_days": -0.01228752,
    "current_days_supply": 0.00956247,
    "days_since_previous_fill": -0.00800432,
    "prescription_age_days": -0.00028218,
}


def locked_logistic_odds_ratios(observations: pd.DataFrame) -> pd.DataFrame:
    """Reproduce locked coefficients and convert them to standardized odds ratios."""
    coefficients = fit_tuned_logistic_coefficients(observations)
    documented = coefficients["feature"].map(DOCUMENTED_LOCKED_COEFFICIENTS)
    if documented.isna().any() or not np.allclose(
        coefficients["coefficient"], documented, atol=5e-8, rtol=0
    ):
        raise RuntimeError("Locked coefficients do not reproduce the documented analysis.")
    result = coefficients.loc[:, ["feature", "coefficient", "absolute_coefficient"]].copy()
    result["odds_ratio"] = np.exp(result["coefficient"])

    def direction(coefficient: float) -> str:
        if abs(coefficient) < NEUTRAL_COEFFICIENT_TOLERANCE:
            return "approximately neutral"
        if coefficient > 0:
            return "increases predicted renewal odds"
        return "decreases predicted renewal odds"

    result["direction"] = result["coefficient"].map(direction)
    return result.loc[:, ["feature", "coefficient", "odds_ratio", "direction"]]
