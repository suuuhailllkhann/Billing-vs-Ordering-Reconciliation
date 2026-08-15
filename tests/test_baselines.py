import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from pharmacy_reconciliation.research.baselines import (
    BASELINE_THRESHOLD,
    baseline_model_factories,
    compare_baselines,
    prepare_baseline_partitions,
)
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN


def _baseline_observations() -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    rows = []
    dates = ["2025-06-01"] * 80 + ["2026-02-15"] * 30 + ["2026-04-15"] * 20
    for index, observation_date in enumerate(dates):
        row: dict[str, object] = {
            "observation_id": f"OBS-{index:03d}",
            "observation_date": observation_date,
            "expected_supply_end_date": observation_date,
            "patient_id": f"PAT-{index % 20:03d}",
            "medication_id": "MED-001",
            "ndc": "90000000001",
            "prescription_id": f"RX-{index:03d}",
            "current_refills_remaining": 0,
        }
        for feature in FEATURE_COLUMNS:
            row[feature] = float(rng.normal())
        row["std_previous_refill_interval_days"] = (
            None if index % 4 == 0 else abs(float(rng.normal(3, 1)))
        )
        row[TARGET_COLUMN] = int(float(row["previous_on_time_fill_rate"]) > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_baseline_training_is_reproducible():
    partitions = prepare_baseline_partitions(_baseline_observations())
    first = compare_baselines(partitions)
    second = compare_baselines(partitions)
    assert first == second


def test_only_logistic_baseline_uses_standard_scaling():
    factories = baseline_model_factories()
    logistic = factories["Logistic Regression"][0]()
    assert isinstance(logistic, Pipeline)
    assert "scaler" in logistic.named_steps
    for name in ("Random Forest", "XGBoost", "LightGBM"):
        assert not isinstance(factories[name][0](), Pipeline)


def test_baseline_comparison_interface_cannot_access_test_partition():
    partitions = prepare_baseline_partitions(_baseline_observations())
    assert not hasattr(partitions, "test")
    assert len(partitions.train.features) == 80
    assert len(partitions.validation.features) == 30


def test_baseline_threshold_is_fixed_reference_value():
    assert BASELINE_THRESHOLD == 0.50
