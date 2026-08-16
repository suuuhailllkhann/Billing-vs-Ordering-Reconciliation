import logging
import uuid
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pharmacy_reconciliation.api.app import NEW_YORK, create_app, current_new_york_date
from pharmacy_reconciliation.api.model_loader import load_locked_model
from pharmacy_reconciliation.api.schemas import PredictionRequest
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS
from pharmacy_reconciliation.research.final_evaluation import LOCKED_THRESHOLD
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS

TODAY = date(2026, 8, 15)


class SpyModel:
    def __init__(self, probability: float = 0.60) -> None:
        self.probability = probability
        self.received: list[pd.DataFrame] = []

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.received.append(features.copy())
        return np.array([[1 - self.probability, self.probability]])


def _record(days_remaining: int = 5, **updates: Any) -> dict[str, Any]:
    days_supply = 30
    fill_date = TODAY + timedelta(days=days_remaining - days_supply)
    record: dict[str, Any] = {
        "rx_number": "RX-AUDIT-ONLY",
        "fill_date": fill_date.isoformat(),
        "current_refills_remaining": 0,
        "current_quantity_billed": 30.0,
        "current_days_supply": days_supply,
        "current_refill_number": 2,
        "previous_fill_count": 4,
        "days_since_previous_fill": 30.0,
        "average_previous_refill_interval_days": 31.0,
        "std_previous_refill_interval_days": None,
        "latest_refill_timing_gap_days": 0.0,
        "average_previous_timing_gap_days": 0.5,
        "previous_early_fill_rate": 0.2,
        "previous_on_time_fill_rate": 0.7,
        "prescription_age_days": 120.0,
        "medication_prior_fill_count": 100,
        "medication_prior_average_days_supply": 30.0,
        "medication_prior_average_quantity": 30.0,
    }
    record.update(updates)
    return record


def _client(model: Any) -> TestClient:
    application = create_app(model=model, persistence_enabled=False)
    application.dependency_overrides[current_new_york_date] = lambda: TODAY
    return TestClient(application)


def test_health_is_minimal_and_locked_model_loads() -> None:
    model = load_locked_model()
    assert tuple(model.named_steps) == ("preprocessor", "scaler", "classifier")
    transformed = model.named_steps["preprocessor"].transform(
        pd.DataFrame([{column: 0.0 for column in FEATURE_COLUMNS}])
    )
    assert tuple(transformed.columns) == MODEL_FEATURE_COLUMNS
    assert transformed.loc[0, "refill_interval_std_available"] == 1
    with _client(model) as client:
        assert client.get("/health").json() == {
            "status": "healthy", "model_loaded": True, "model_status": "locked_final"
        }
        assert client.get("/health/live").json() == {"status": "alive"}
        assert client.get("/health/ready").json() == {
            "status": "ready",
            "model_loaded": True,
            "database_available": True,
            "model_status": "locked_final",
        }


def test_request_id_header_and_safe_structured_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="uvicorn.error.pharmacy_reconciliation.api",
    )
    with _client(SpyModel()) as client:
        response = client.post(
            "/predict",
            json=_record(rx_number="RX-PRIVATE-LOG-CHECK"),
        )
    request_id = response.headers["X-Request-ID"]
    assert str(uuid.UUID(request_id)) == request_id
    assert f"request_id={request_id}" in caplog.text
    assert "method=POST" in caplog.text
    assert "path=/predict" in caplog.text
    assert "status_code=200" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "RX-PRIVATE-LOG-CHECK" not in caplog.text
    assert "renewal_probability" not in caplog.text


def test_single_prediction_derives_dates_and_excludes_audit_business_fields() -> None:
    model = SpyModel(0.60)
    with _client(model) as client:
        response = client.post("/predict", json=_record(days_remaining=5))
    assert response.status_code == 200
    body = response.json()
    assert body["expected_supply_end_date"] == "2026-08-20"
    assert body["days_until_supply_end"] == 5
    assert body["priority"] == "medium"
    assert body["renewal_probability"] == 0.60
    assert body["threshold"] == LOCKED_THRESHOLD == 0.50
    assert body["prediction"] == 1
    assert body["review_status"] == "prescription_review"
    assert tuple(model.received[0].columns) == FEATURE_COLUMNS
    assert "rx_number" not in model.received[0]
    assert "current_refills_remaining" not in model.received[0]
    assert "refill_interval_std_available" not in model.received[0]
    assert pd.isna(model.received[0].iloc[0]["std_previous_refill_interval_days"])


@pytest.mark.parametrize(
    ("days_remaining", "priority", "eligible", "review_status"),
    [
        (10, "low", True, "prescription_review"),
        (8, "low", True, "prescription_review"),
        (7, "medium", True, "prescription_review"),
        (5, "medium", True, "prescription_review"),
        (4, "high", True, "prescription_review"),
        (1, "high", True, "prescription_review"),
        (0, "urgent", True, "prescription_review"),
        (-1, "urgent_overdue", True, "prescription_review"),
        (-7, "urgent_overdue", True, "prescription_review"),
        (11, None, False, "not_in_prediction_window"),
        (-8, None, False, "manual_review"),
    ],
)
def test_priority_and_eligibility_bands(
    days_remaining: int, priority: str | None, eligible: bool, review_status: str
) -> None:
    with _client(SpyModel()) as client:
        body = client.post("/predict", json=_record(days_remaining)).json()
    assert body["priority"] == priority
    assert body["eligible"] is eligible
    assert body["review_status"] == review_status
    assert (body["prediction"] is not None) is eligible


def test_refills_remaining_blocks_model() -> None:
    model = SpyModel()
    with _client(model) as client:
        body = client.post(
            "/predict", json=_record(current_refills_remaining=1)
        ).json()
    assert body["review_status"] == "refills_remaining"
    assert body["prediction"] is None
    assert model.received == []


def test_batch_continues_after_invalid_record_and_logs_no_identifier(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(
        logging.INFO,
        logger="uvicorn.error.pharmacy_reconciliation.api",
    )
    records = [_record(5), _record(5, rx_number="SECRET-RX", previous_on_time_fill_rate=2)]
    with _client(SpyModel()) as client:
        response = client.post("/predict/batch", json={"records": records})
    body = response.json()
    assert response.status_code == 200
    assert (body["total_records"], body["processed"], body["failed"]) == (2, 1, 1)
    assert body["eligible_records"] == 1
    assert body["flagged_for_review"] == 1
    assert len(body["results"]) == 2
    assert body["results"][1]["status"] == "error"
    assert body["results"][1]["rx_number"] == "SECRET-RX"
    assert "SECRET-RX" not in caplog.text
    assert "RX-AUDIT-ONLY" not in caplog.text


def test_batch_limit_and_invalid_source_values() -> None:
    with _client(SpyModel()) as client:
        too_large = client.post("/predict/batch", json={"records": [_record()] * 501})
        invalid = client.post("/predict", json=_record(current_quantity_billed=-1))
        invalid_type = client.post("/predict", json=_record(current_days_supply="thirty"))
        malformed = client.post("/predict/batch", json={"wrong": []})
    assert too_large.status_code == 422
    assert invalid.status_code == 422
    assert invalid_type.status_code == 422
    assert malformed.status_code == 422


def test_probability_threshold_consistency_and_clock_contract() -> None:
    assert NEW_YORK.key == "America/New_York"
    with _client(SpyModel(0.49)) as client:
        body = client.post("/predict", json=_record()).json()
    assert body["renewal_probability"] == 0.49
    assert body["threshold"] == 0.50
    assert body["prediction"] == 0
    assert body["review_status"] == "no_review"


def test_startup_fails_clearly_when_model_cannot_load() -> None:
    def broken_loader() -> Any:
        raise RuntimeError("missing")

    with pytest.raises(RuntimeError, match="Locked model failed to load"):
        with TestClient(create_app(model_loader=broken_loader)):
            pass


def test_openapi_contains_valid_synthetic_request_examples() -> None:
    schema = create_app(model=SpyModel(), persistence_enabled=False).openapi()["components"]["schemas"]
    single_example = schema["PredictionRequest"]["example"]
    batch_example = schema["BatchPredictionRequest"]["example"]

    assert PredictionRequest.model_validate(single_example).rx_number == "RX-DEMO-001"
    assert len(batch_example["records"]) == 2
    assert [
        PredictionRequest.model_validate(record).rx_number
        for record in batch_example["records"]
    ] == ["RX-DEMO-001", "RX-DEMO-002"]
    assert all("patient" not in key for key in single_example)
    assert "ndc" not in single_example
