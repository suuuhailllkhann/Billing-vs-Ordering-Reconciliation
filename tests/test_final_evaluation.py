import pandas as pd

from pharmacy_reconciliation.research.feature_analysis import tuned_logistic_pipeline
from pharmacy_reconciliation.research.final_evaluation import (
    FINAL_FITTING_POLICY,
    LOCKED_THRESHOLD,
    evaluate_locked_test,
)
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS


def test_locked_contract_is_logistic_full_feature_train_only(
    longitudinal_observations: pd.DataFrame,
) -> None:
    assert FINAL_FITTING_POLICY == "Train only"
    assert LOCKED_THRESHOLD == 0.50
    model = tuned_logistic_pipeline()
    assert tuple(model.named_steps) == ("preprocessor", "scaler", "classifier")
    assert model.named_steps["classifier"].C == 0.01
    frame = longitudinal_observations
    train = frame.loc[pd.to_datetime(frame["observation_date"]) < "2026-02-01"]
    transformed = model.named_steps["preprocessor"].fit_transform(train)
    assert tuple(transformed.columns) == MODEL_FEATURE_COLUMNS
    assert "patient_id" not in transformed.columns


def test_final_evaluation_is_deterministic_and_test_cannot_fit_preprocessing(
    longitudinal_observations: pd.DataFrame,
) -> None:
    frame = longitudinal_observations
    first = evaluate_locked_test(frame)
    changed = frame.copy()
    test_mask = pd.to_datetime(changed["observation_date"]) >= "2026-04-01"
    changed.loc[test_mask, "std_previous_refill_interval_days"] = 99999.0
    second = evaluate_locked_test(changed)
    assert first.validation == second.validation
    assert first.overall == evaluate_locked_test(frame).overall
    assert first.seen_patients["observations"] + first.unseen_patients["observations"] == first.overall["observations"]
