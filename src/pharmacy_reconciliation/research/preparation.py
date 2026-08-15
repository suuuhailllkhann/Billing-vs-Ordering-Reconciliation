"""Leakage-safe ML dataset preparation without model training."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN

TRAIN_END_EXCLUSIVE = pd.Timestamp("2026-02-01")
VALIDATION_END_INCLUSIVE = pd.Timestamp("2026-03-31")
TEST_START = pd.Timestamp("2026-04-01")
MISSING_STD_COLUMN = "std_previous_refill_interval_days"
MISSING_STD_INDICATOR = "refill_interval_std_available"

EXCLUDED_MODEL_COLUMNS = (
    "observation_id",
    "observation_date",
    "expected_supply_end_date",
    "patient_id",
    "medication_id",
    "ndc",
    "prescription_id",
    "current_refills_remaining",
)
MODEL_FEATURE_COLUMNS = (*FEATURE_COLUMNS, MISSING_STD_INDICATOR)

ModelFamily = Literal["logistic_regression", "random_forest", "xgboost", "lightgbm"]


@dataclass(frozen=True)
class PreprocessingPlan:
    """Planned preprocessing contract for a future model family."""

    imputation: Literal["train_median"]
    add_missingness_indicator: bool
    standardize: bool


PREPROCESSING_PLANS: dict[ModelFamily, PreprocessingPlan] = {
    "logistic_regression": PreprocessingPlan("train_median", True, True),
    "random_forest": PreprocessingPlan("train_median", True, False),
    "xgboost": PreprocessingPlan("train_median", True, False),
    "lightgbm": PreprocessingPlan("train_median", True, False),
}


@dataclass(frozen=True)
class TemporalObservationSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class PreparedPartition:
    features: pd.DataFrame
    target: pd.Series
    audit: pd.DataFrame


@dataclass(frozen=True)
class PreparedTemporalDataset:
    train: PreparedPartition
    validation: PreparedPartition
    test: PreparedPartition
    std_imputation_median: float


def chronological_split(observations: pd.DataFrame) -> TemporalObservationSplit:
    """Apply the fixed production-like chronological boundaries without shuffling."""
    required = {"observation_date", TARGET_COLUMN, *FEATURE_COLUMNS, *EXCLUDED_MODEL_COLUMNS}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Observation dataset is missing required columns: {missing}")

    dated = observations.copy()
    dated["observation_date"] = pd.to_datetime(dated["observation_date"]).dt.normalize()
    dated = dated.sort_values(
        ["observation_date", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)
    train = dated.loc[dated["observation_date"] < TRAIN_END_EXCLUSIVE].copy()
    validation = dated.loc[
        dated["observation_date"].between(
            TRAIN_END_EXCLUSIVE, VALIDATION_END_INCLUSIVE, inclusive="both"
        )
    ].copy()
    test = dated.loc[dated["observation_date"] >= TEST_START].copy()
    if len(train) + len(validation) + len(test) != len(dated):
        raise ValueError("One or more observations did not map to a temporal split.")
    return TemporalObservationSplit(train, validation, test)


class TrainMedianPreprocessor:
    """Fit the one approved imputation value on training features only."""

    def __init__(self) -> None:
        self._std_median: float | None = None

    @property
    def std_median(self) -> float:
        if self._std_median is None:
            raise RuntimeError("Preprocessor has not been fitted on training data.")
        return self._std_median

    def fit(self, training_features: pd.DataFrame) -> "TrainMedianPreprocessor":
        self._validate_features(training_features)
        median = training_features[MISSING_STD_COLUMN].median(skipna=True)
        if pd.isna(median):
            raise ValueError("Training data has no available refill-interval standard deviation.")
        self._std_median = float(median)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        self._validate_features(features)
        output = features.loc[:, FEATURE_COLUMNS].copy()
        output[MISSING_STD_INDICATOR] = output[MISSING_STD_COLUMN].notna().astype("int8")
        output[MISSING_STD_COLUMN] = output[MISSING_STD_COLUMN].fillna(self.std_median)
        return output.loc[:, MODEL_FEATURE_COLUMNS]

    @staticmethod
    def _validate_features(features: pd.DataFrame) -> None:
        missing = sorted(set(FEATURE_COLUMNS).difference(features.columns))
        if missing:
            raise ValueError(f"Feature table is missing required columns: {missing}")


def _partition(
    observations: pd.DataFrame, preprocessor: TrainMedianPreprocessor
) -> PreparedPartition:
    raw_features = observations.loc[:, FEATURE_COLUMNS]
    return PreparedPartition(
        features=preprocessor.transform(raw_features),
        target=observations[TARGET_COLUMN].astype("int8").copy(),
        audit=observations.loc[:, EXCLUDED_MODEL_COLUMNS].copy(),
    )


def prepare_temporal_dataset(observations: pd.DataFrame) -> PreparedTemporalDataset:
    """Split chronologically, fit on Train, then transform all three periods."""
    split = chronological_split(observations)
    preprocessor = TrainMedianPreprocessor().fit(split.train.loc[:, FEATURE_COLUMNS])
    return PreparedTemporalDataset(
        train=_partition(split.train, preprocessor),
        validation=_partition(split.validation, preprocessor),
        test=_partition(split.test, preprocessor),
        std_imputation_median=preprocessor.std_median,
    )
