"""Experimental reduced-feature retuning without changing the full feature contract."""

from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from pharmacy_reconciliation.research.baselines import (
    BASELINE_RANDOM_SEED,
    evaluate_classifier,
)
from pharmacy_reconciliation.research.preparation import MISSING_STD_INDICATOR
from pharmacy_reconciliation.research.tuning import (
    FoldLocalRefillPreprocessor,
    RawTuningPartitions,
    TunedModelResult,
    tuning_searches,
)

REDUCED_FEATURE_COLUMNS = (
    "previous_on_time_fill_rate",
    "latest_refill_timing_gap_days",
    "average_previous_timing_gap_days",
    "std_previous_refill_interval_days",
    MISSING_STD_INDICATOR,
    "previous_fill_count",
    "average_previous_refill_interval_days",
    "current_refill_number",
    "medication_prior_average_quantity",
    "medication_prior_average_days_supply",
    "medication_prior_fill_count",
)

EXPERIMENTALLY_EXCLUDED_FEATURES = (
    "current_quantity_billed",
    "current_days_supply",
    "days_since_previous_fill",
    "previous_early_fill_rate",
    "prescription_age_days",
)


class ReducedFoldLocalRefillPreprocessor(FoldLocalRefillPreprocessor):
    """Apply the existing fold-local logic, then select the experimental columns."""

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = super().transform(features)
        return transformed.loc[:, REDUCED_FEATURE_COLUMNS]


def reduced_tuning_searches(
    seed: int = BASELINE_RANDOM_SEED,
) -> dict[str, RandomizedSearchCV]:
    """Use the full experiment's searches with only the feature selector replaced."""
    searches = tuning_searches(seed)
    for search in searches.values():
        search.estimator.set_params(preprocessor=ReducedFoldLocalRefillPreprocessor())
    return searches


def _clean_parameters(parameters: dict[str, object]) -> dict[str, object]:
    cleaned = {
        key.removeprefix("classifier__"): value for key, value in parameters.items()
    }
    l1_ratio = cleaned.pop("l1_ratio", None)
    if l1_ratio is not None:
        cleaned["penalty"] = "l1" if l1_ratio == 1.0 else "l2"
    return cleaned


def tune_reduced_models(
    partitions: RawTuningPartitions,
) -> tuple[TunedModelResult, ...]:
    """Tune reduced-feature models on Train and evaluate on Validation only."""
    results = []
    for model_name, search in reduced_tuning_searches().items():
        search.fit(partitions.train.features, partitions.train.target)
        best_model: Any = search.best_estimator_
        results.append(TunedModelResult(
            model_name=model_name,
            trials=search.n_iter,
            best_cv_pr_auc=float(search.best_score_),
            best_parameters=_clean_parameters(search.best_params_),
            train=evaluate_classifier(best_model, partitions.train),
            validation=evaluate_classifier(best_model, partitions.validation),
        ))
    return tuple(results)
