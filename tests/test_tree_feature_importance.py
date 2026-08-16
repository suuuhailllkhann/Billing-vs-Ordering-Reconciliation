import pandas as pd
from pandas.testing import assert_frame_equal

from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS
from pharmacy_reconciliation.research.tree_feature_importance import (
    TREE_MODEL_NAMES,
    combined_tree_importance_ranks,
    fit_tuned_tree_importances,
    tuned_tree_pipelines,
)


def test_tree_importances_cover_and_rank_every_final_input(
    longitudinal_observations: pd.DataFrame,
) -> None:
    observations = longitudinal_observations
    results = fit_tuned_tree_importances(observations)

    assert tuple(results) == TREE_MODEL_NAMES
    for result in results.values():
        assert set(result["feature"]) == set(MODEL_FEATURE_COLUMNS)
        assert result["importance"].is_monotonic_decreasing
        assert result["rank"].tolist() == list(range(1, len(MODEL_FEATURE_COLUMNS) + 1))

    combined = combined_tree_importance_ranks(results)
    assert set(combined["feature"]) == set(MODEL_FEATURE_COLUMNS)
    assert list(combined.columns) == [
        "feature",
        "Random Forest rank",
        "XGBoost rank",
        "LightGBM rank",
    ]


def test_tuned_tree_configuration_and_test_period_isolation(
    longitudinal_observations: pd.DataFrame,
) -> None:
    pipelines = tuned_tree_pipelines()
    assert pipelines["Random Forest"].named_steps["classifier"].n_estimators == 200
    assert pipelines["XGBoost"].named_steps["classifier"].reg_lambda == 10
    assert pipelines["LightGBM"].named_steps["classifier"].num_leaves == 7

    observations = longitudinal_observations
    changed_test = observations.copy()
    test_mask = pd.to_datetime(changed_test["observation_date"]) >= "2026-04-01"
    changed_test.loc[test_mask, "previous_on_time_fill_rate"] = 999.0
    changed_test.loc[test_mask, "prescription_renewal_within_window"] = 0
    original = fit_tuned_tree_importances(observations)
    changed = fit_tuned_tree_importances(changed_test)
    for name in TREE_MODEL_NAMES:
        assert_frame_equal(original[name], changed[name])
