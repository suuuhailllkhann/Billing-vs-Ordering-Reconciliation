from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from pharmacy_reconciliation.api.app import (
    create_app,
    current_new_york_date,
    current_utc_datetime,
)
from pharmacy_reconciliation.persistence.config import load_database_settings
from pharmacy_reconciliation.persistence.models import (
    Base,
    CaseResolution,
    FollowUpActivity,
    FollowUpCase,
    PredictionRecord,
)

TODAY = date(2026, 8, 15)
NOW = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)


class MutableModel:
    def __init__(self, probability: float = 0.75) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[1 - self.probability, self.probability]])


def _record(rx_number: str, days_remaining: int = 5, **updates: Any) -> dict[str, Any]:
    days_supply = 30
    record: dict[str, Any] = {
        "rx_number": rx_number,
        "fill_date": (TODAY + timedelta(days=days_remaining - days_supply)).isoformat(),
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


@pytest.fixture
def database() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _client(model: MutableModel, database: sessionmaker[Session]) -> TestClient:
    application = create_app(model=model, session_factory=database)
    application.dependency_overrides[current_new_york_date] = lambda: TODAY
    application.dependency_overrides[current_utc_datetime] = lambda: NOW
    return TestClient(application)


def _counts(database: sessionmaker[Session]) -> tuple[int, int]:
    with database() as session:
        return (
            session.scalar(select(func.count()).select_from(PredictionRecord)) or 0,
            session.scalar(select(func.count()).select_from(FollowUpCase)) or 0,
        )


def test_database_configuration_is_environment_only_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_env = Path("definitely-missing-phase3c.env")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not configured"):
        load_database_settings(missing_env)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///not-allowed.db")
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        load_database_settings(missing_env)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://wrong_user:synthetic@localhost/pharmacy_reconciliation",
    )
    with pytest.raises(RuntimeError, match="application database user"):
        load_database_settings(missing_env)


def test_positive_and_negative_prediction_persistence(database: sessionmaker[Session]) -> None:
    model = MutableModel(0.75)
    with _client(model, database) as client:
        assert client.post("/predict", json=_record("RX-SYNTH-001")).status_code == 200
        model.probability = 0.25
        assert client.post("/predict", json=_record("RX-SYNTH-002")).status_code == 200
    assert _counts(database) == (2, 1)
    with database() as session:
        records = list(session.scalars(select(PredictionRecord).order_by(PredictionRecord.rx_number)))
        assert [record.prediction for record in records] == [1, 0]
        assert all(record.threshold == 0.50 for record in records)


def test_ineligible_prediction_is_not_persisted(database: sessionmaker[Session]) -> None:
    with _client(MutableModel(), database) as client:
        response = client.post(
            "/predict", json=_record("RX-SYNTH-INELIGIBLE", current_refills_remaining=1)
        )
    assert response.json()["eligible"] is False
    assert _counts(database) == (0, 0)


def test_repeated_predictions_preserve_history_and_case_lifecycle(
    database: sessionmaker[Session],
) -> None:
    model = MutableModel(0.80)
    with _client(model, database) as client:
        client.post("/predict", json=_record("RX-SYNTH-REPEAT", 7))
        with database() as session:
            first_case = session.scalar(select(FollowUpCase))
            assert first_case is not None
            initial_id = first_case.initial_prediction_id
            case_id = first_case.case_id
        client.post("/predict", json=_record("RX-SYNTH-REPEAT", 1))
        model.probability = 0.20
        client.post("/predict", json=_record("RX-SYNTH-REPEAT", 0))
        case_body = client.get(f"/cases/{case_id}").json()
        assert case_body["status"] == "open"
        assert case_body["priority"] == "high"
        client.post(
            f"/cases/{case_id}/resolve",
            json={"resolution_type": "new_prescription_received"},
        )
        model.probability = 0.80
        client.post("/predict", json=_record("RX-SYNTH-REPEAT", -1))
    assert _counts(database) == (4, 2)
    with database() as session:
        cases = list(session.scalars(select(FollowUpCase).order_by(FollowUpCase.opened_at)))
        assert cases[0].initial_prediction_id == initial_id
        assert cases[0].status == "resolved"
        assert cases[1].status == "open"


def test_database_prevents_duplicate_open_cases(database: sessionmaker[Session]) -> None:
    with _client(MutableModel(), database) as client:
        client.post("/predict", json=_record("RX-SYNTH-UNIQUE"))
    with database.begin() as session:
        existing = session.scalar(select(FollowUpCase))
        assert existing is not None
        duplicate = FollowUpCase(
            rx_number=existing.rx_number,
            initial_prediction_id=existing.initial_prediction_id,
            latest_prediction_id=existing.latest_prediction_id,
            status="open",
            priority="medium",
            latest_renewal_probability=0.7,
            latest_days_until_supply_end=5,
            opened_at=NOW,
            last_evaluated_at=NOW,
        )
        session.add(duplicate)
        with pytest.raises(IntegrityError):
            session.flush()


def test_activity_and_atomic_resolution_rules(database: sessionmaker[Session]) -> None:
    with _client(MutableModel(), database) as client:
        client.post("/predict", json=_record("RX-SYNTH-WORKFLOW"))
        case_id = client.get("/cases").json()[0]["case_id"]
        activity = client.post(
            f"/cases/{case_id}/activities",
            json={"activity_type": "left_voicemail"},
        )
        assert activity.status_code == 201
        assert activity.json()["activity_note"] is None
        assert client.get(f"/cases/{case_id}").json()["status"] == "open"
        assert client.post(
            f"/cases/{case_id}/resolve", json={"resolution_type": "other"}
        ).status_code == 422
        resolved = client.post(
            f"/cases/{case_id}/resolve",
            json={"resolution_type": "other", "resolution_note": "Synthetic closure"},
        )
        assert resolved.status_code == 201
        assert client.post(
            f"/cases/{case_id}/resolve",
            json={"resolution_type": "medication_discontinued"},
        ).status_code == 409
        assert client.post(
            f"/cases/{case_id}/activities",
            json={"activity_type": "patient_contacted"},
        ).status_code == 409
    with database() as session:
        assert session.scalar(select(func.count()).select_from(FollowUpActivity)) == 1
        assert session.scalar(select(func.count()).select_from(CaseResolution)) == 1


def test_queue_order_and_filters(database: sessionmaker[Session]) -> None:
    with _client(MutableModel(), database) as client:
        for rx_number, days in (
            ("RX-SYNTH-LOW", 9),
            ("RX-SYNTH-HIGH", 2),
            ("RX-SYNTH-OVERDUE", -2),
            ("RX-SYNTH-URGENT", 0),
        ):
            client.post("/predict", json=_record(rx_number, days))
        queue = client.get("/cases").json()
        assert [item["priority"] for item in queue] == [
            "urgent_overdue", "urgent", "high", "low"
        ]
        assert len(client.get("/cases", params={"priority": "high"}).json()) == 1
        assert client.get("/cases", params={"status": "resolved"}).json() == []


def test_batch_records_commit_independently(database: sessionmaker[Session]) -> None:
    with _client(MutableModel(), database) as client:
        body = client.post("/predict/batch", json={"records": [
            _record("RX-SYNTH-BATCH-1"),
            _record("RX-SYNTH-BATCH-BAD", previous_on_time_fill_rate=2),
            _record("RX-SYNTH-BATCH-2"),
        ]}).json()
    assert (body["processed"], body["failed"]) == (2, 1)
    assert _counts(database) == (2, 2)


def test_batch_database_error_is_safe_and_does_not_stop_other_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR", logger="pharmacy_reconciliation.api")
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    missing_schema = sessionmaker(bind=engine, expire_on_commit=False)
    with _client(MutableModel(), missing_schema) as client:
        body = client.post("/predict/batch", json={"records": [
            _record("RX-SYNTH-DB-ERROR-1"),
            _record("RX-SYNTH-DB-ERROR-2"),
        ]}).json()
    assert (body["processed"], body["failed"]) == (0, 2)
    assert [item["error_code"] for item in body["results"]] == [
        "persistence_error", "persistence_error"
    ]
    assert "RX-SYNTH" not in caplog.text
    assert "sqlite" not in caplog.text.lower()


def test_model_schema_stores_no_features_or_patient_identity() -> None:
    columns = set(PredictionRecord.__table__.columns.keys())
    assert not columns.intersection({
        "patient_id", "patient_name", "date_of_birth", "ndc", "raw_payload",
        "previous_on_time_fill_rate", "current_days_supply",
    })
    assert {"model_version", "model_run_id"}.issubset(columns)
