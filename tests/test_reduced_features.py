import pandas as pd
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.reduced_features import (
    EXPERIMENTALLY_EXCLUDED_FEATURES,
    REDUCED_FEATURE_COLUMNS,
    ReducedFoldLocalRefillPreprocessor,
    reduced_tuning_searches,
)
from pharmacy_reconciliation.research.tuning import (
    TUNING_CV_SPLITS,
    TUNING_SCORING,
    TUNING_TRIALS,
    prepare_tuning_partitions,
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
        row["std_previous_refill_interval_days"] = None if index == 0 else 2.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_exact_reduced_and_excluded_feature_contracts() -> None:
    assert REDUCED_FEATURE_COLUMNS == (
        "previous_on_time_fill_rate",
        "latest_refill_timing_gap_days",
        "average_previous_timing_gap_days",
        "std_previous_refill_interval_days",
        "refill_interval_std_available",
        "previous_fill_count",
        "average_previous_refill_interval_days",
        "current_refill_number",
        "medication_prior_average_quantity",
        "medication_prior_average_days_supply",
        "medication_prior_fill_count",
    )
    assert set(EXPERIMENTALLY_EXCLUDED_FEATURES).isdisjoint(REDUCED_FEATURE_COLUMNS)


def test_reduced_preprocessor_is_fold_local_and_excludes_five_features() -> None:
    raw = _observations().loc[:, FEATURE_COLUMNS]
    transformed = ReducedFoldLocalRefillPreprocessor().fit(raw).transform(raw)
    assert tuple(transformed.columns) == REDUCED_FEATURE_COLUMNS
    assert not set(EXPERIMENTALLY_EXCLUDED_FEATURES).intersection(transformed.columns)
    assert transformed.loc[0, "refill_interval_std_available"] == 0
    assert transformed.loc[0, "std_previous_refill_interval_days"] == 2.0


def test_reduced_search_protocol_is_identical_and_deterministic() -> None:
    first = reduced_tuning_searches()
    second = reduced_tuning_searches()
    for name, search in first.items():
        assert isinstance(search.cv, TimeSeriesSplit)
        assert search.cv.n_splits == TUNING_CV_SPLITS == 5
        assert search.scoring == TUNING_SCORING == "average_precision"
        assert search.n_iter == TUNING_TRIALS[name]
        assert isinstance(search.estimator.named_steps["preprocessor"], ReducedFoldLocalRefillPreprocessor)
        assert isinstance(search.estimator.named_steps.get("scaler"), StandardScaler) == (
            name == "Logistic Regression"
        )
        candidates = list(ParameterSampler(
            search.param_distributions, search.n_iter, random_state=search.random_state
        ))
        other = second[name]
        other_candidates = list(ParameterSampler(
            other.param_distributions, other.n_iter, random_state=other.random_state
        ))
        assert candidates == other_candidates


def test_reduced_tuning_partitions_preserve_split_and_exclude_test() -> None:
    partitions = prepare_tuning_partitions(_observations())
    assert len(partitions.train.features) == 20
    assert len(partitions.validation.features) == 5
    assert not hasattr(partitions, "test")
