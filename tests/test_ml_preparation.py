import pandas as pd

from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.preparation import (
    EXCLUDED_MODEL_COLUMNS,
    MISSING_STD_COLUMN,
    MISSING_STD_INDICATOR,
    MODEL_FEATURE_COLUMNS,
    PREPROCESSING_PLANS,
    TrainMedianPreprocessor,
    chronological_split,
    prepare_temporal_dataset,
)


def _observation(observation_id: str, observation_date: str, target: int) -> dict[str, object]:
    row: dict[str, object] = {
        "observation_id": observation_id,
        "observation_date": observation_date,
        "expected_supply_end_date": observation_date,
        "patient_id": "SYN-PAT-1",
        "medication_id": "SYN-MED-1",
        "ndc": "90000000001",
        "prescription_id": "SYN-RX-1",
        "current_refills_remaining": 0,
        TARGET_COLUMN: target,
    }
    row.update(dict.fromkeys(FEATURE_COLUMNS, 1.0))
    return row


def test_chronological_split_boundaries_are_exact_and_never_randomized():
    observations = pd.DataFrame([
        _observation("test", "2026-04-01", 0),
        _observation("train", "2026-01-31", 1),
        _observation("validation-start", "2026-02-01", 1),
        _observation("validation-end", "2026-03-31", 0),
    ])
    split = chronological_split(observations)
    assert split.train["observation_id"].tolist() == ["train"]
    assert split.validation["observation_id"].tolist() == [
        "validation-start", "validation-end"
    ]
    assert split.test["observation_id"].tolist() == ["test"]


def test_model_schema_excludes_identifiers_dates_and_eligibility_field():
    expected_exclusions = {
        "observation_id", "observation_date", "expected_supply_end_date",
        "patient_id", "medication_id", "ndc", "prescription_id",
        "current_refills_remaining",
    }
    assert set(EXCLUDED_MODEL_COLUMNS) == expected_exclusions
    assert expected_exclusions.isdisjoint(MODEL_FEATURE_COLUMNS)
    assert TARGET_COLUMN not in MODEL_FEATURE_COLUMNS
    assert set(FEATURE_COLUMNS) < set(MODEL_FEATURE_COLUMNS)


def test_missingness_indicator_and_train_median_are_applied():
    training = pd.DataFrame([
        dict.fromkeys(FEATURE_COLUMNS, 1.0),
        dict.fromkeys(FEATURE_COLUMNS, 1.0),
        dict.fromkeys(FEATURE_COLUMNS, 1.0),
    ])
    training[MISSING_STD_COLUMN] = [2.0, None, 6.0]
    preprocessor = TrainMedianPreprocessor().fit(training)
    transformed = preprocessor.transform(training)
    assert preprocessor.std_median == 4.0
    assert transformed[MISSING_STD_INDICATOR].tolist() == [1, 0, 1]
    assert transformed[MISSING_STD_COLUMN].tolist() == [2.0, 4.0, 6.0]


def test_validation_and_test_values_cannot_influence_fitted_imputation():
    rows = [
        _observation("train-1", "2026-01-01", 1),
        _observation("train-2", "2026-01-02", 0),
        _observation("train-3", "2026-01-03", 1),
        _observation("validation", "2026-02-01", 0),
        _observation("test", "2026-04-01", 0),
    ]
    observations = pd.DataFrame(rows)
    observations[MISSING_STD_COLUMN] = [2.0, None, 6.0, 1000.0, -1000.0]
    prepared = prepare_temporal_dataset(observations)
    assert prepared.std_imputation_median == 4.0

    observations.loc[
        observations["observation_id"].isin(["validation", "test"]),
        MISSING_STD_COLUMN,
    ] = None
    changed_future = prepare_temporal_dataset(observations)
    assert changed_future.std_imputation_median == 4.0
    assert changed_future.validation.features[MISSING_STD_COLUMN].iloc[0] == 4.0
    assert changed_future.test.features[MISSING_STD_COLUMN].iloc[0] == 4.0


def test_model_family_preprocessing_contracts_are_explicit():
    assert PREPROCESSING_PLANS["logistic_regression"].standardize
    for model in ("random_forest", "xgboost", "lightgbm"):
        assert not PREPROCESSING_PLANS[model].standardize
    assert all(plan.imputation == "train_median" for plan in PREPROCESSING_PLANS.values())
    assert all(plan.add_missingness_indicator for plan in PREPROCESSING_PLANS.values())
