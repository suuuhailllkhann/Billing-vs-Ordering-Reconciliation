import pandas as pd
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.tuning import (
    TUNING_CV_SPLITS,
    TUNING_SCORING,
    TUNING_TRIALS,
    FoldLocalRefillPreprocessor,
    prepare_tuning_partitions,
    tuning_searches,
)


def _observations() -> pd.DataFrame:
    rows = []
    dates = ["2025-01-01"] * 20 + ["2026-02-01"] * 5 + ["2026-04-01"] * 5
    for index, date in enumerate(dates):
        row: dict[str, object] = {
            "observation_id": f"OBS-{index}",
            "observation_date": date,
            "expected_supply_end_date": date,
            "patient_id": f"PAT-{index}",
            "medication_id": "MED-1",
            "ndc": "90000000001",
            "prescription_id": f"RX-{index}",
            "current_refills_remaining": 0,
            TARGET_COLUMN: index % 2,
        }
        row.update(dict.fromkeys(FEATURE_COLUMNS, 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def test_searches_use_fixed_time_series_split_trials_and_scoring():
    searches = tuning_searches()
    assert set(searches) == set(TUNING_TRIALS)
    for name, search in searches.items():
        assert isinstance(search.cv, TimeSeriesSplit)
        assert search.cv.n_splits == TUNING_CV_SPLITS == 5
        assert search.scoring == TUNING_SCORING == "average_precision"
        assert search.n_iter == TUNING_TRIALS[name]


def test_search_candidate_sampling_is_deterministic():
    first = tuning_searches()
    second = tuning_searches()
    for name in first:
        first_candidates = list(ParameterSampler(
            first[name].param_distributions,
            n_iter=first[name].n_iter,
            random_state=first[name].random_state,
        ))
        second_candidates = list(ParameterSampler(
            second[name].param_distributions,
            n_iter=second[name].n_iter,
            random_state=second[name].random_state,
        ))
        assert first_candidates == second_candidates


def test_preprocessing_and_scaling_are_inside_cv_pipelines():
    searches = tuning_searches()
    for name, search in searches.items():
        assert isinstance(search.estimator.named_steps["preprocessor"], FoldLocalRefillPreprocessor)
        assert ("scaler" in search.estimator.named_steps) == (name == "Logistic Regression")


def test_tuning_partitions_exclude_test_by_construction():
    partitions = prepare_tuning_partitions(_observations())
    assert not hasattr(partitions, "test")
    assert len(partitions.train.features) == 20
    assert len(partitions.validation.features) == 5
