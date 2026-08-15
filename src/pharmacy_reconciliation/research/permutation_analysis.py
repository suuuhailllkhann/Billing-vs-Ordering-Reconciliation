"""Validation-only permutation importance for the locked Logistic Regression."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from pharmacy_reconciliation.research.baselines import BASELINE_RANDOM_SEED
from pharmacy_reconciliation.research.feature_analysis import tuned_logistic_pipeline
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS
from pharmacy_reconciliation.research.tuning import prepare_tuning_partitions

PERMUTATION_SCORING = "average_precision"
PERMUTATION_REPEATS = 30
PERMUTATION_RANDOM_STATE = BASELINE_RANDOM_SEED


@dataclass(frozen=True)
class PermutationAnalysisResult:
    validation_observations: int
    importances: pd.DataFrame


def locked_logistic_permutation_importance(
    observations: pd.DataFrame,
) -> PermutationAnalysisResult:
    """Fit on Train and permute final inputs on Validation only."""
    partitions = prepare_tuning_partitions(observations)
    model = tuned_logistic_pipeline()
    model.fit(partitions.train.features, partitions.train.target)
    validation_features = model.named_steps["preprocessor"].transform(
        partitions.validation.features
    )
    fitted_downstream = Pipeline([
        ("scaler", model.named_steps["scaler"]),
        ("classifier", model.named_steps["classifier"]),
    ])
    permutation: Any = permutation_importance(
        fitted_downstream,
        validation_features,
        partitions.validation.target,
        scoring=PERMUTATION_SCORING,
        n_repeats=PERMUTATION_REPEATS,
        random_state=PERMUTATION_RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": MODEL_FEATURE_COLUMNS,
        "mean_importance": permutation.importances_mean,
        "standard_deviation": permutation.importances_std,
    }).sort_values("mean_importance", ascending=False, kind="mergesort")
    table = table.reset_index(drop=True)
    table["rank"] = table.index + 1
    return PermutationAnalysisResult(
        validation_observations=len(partitions.validation.target),
        importances=table.loc[:, ["feature", "mean_importance", "standard_deviation", "rank"]],
    )
