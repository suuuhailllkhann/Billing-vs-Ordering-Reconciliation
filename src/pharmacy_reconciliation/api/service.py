"""Eligibility and inference orchestration without web-framework dependencies."""

from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd

from pharmacy_reconciliation.api.schemas import PredictionRequest, PredictionResponse
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS
from pharmacy_reconciliation.research.final_evaluation import LOCKED_THRESHOLD

MANUAL_RESOLUTION_REASONS = (
    "new_prescription_received", "medication_discontinued", "dose_changed",
    "medication_changed", "patient_changed_pharmacy", "patient_unreachable",
    "prescriber_no_response", "other",
)


class PredictionService:
    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, record: PredictionRequest, today: date) -> PredictionResponse:
        supply_end = record.fill_date + timedelta(days=record.current_days_supply)
        days_remaining = (supply_end - today).days
        common = {
            "rx_number": record.rx_number,
            "expected_supply_end_date": supply_end,
            "days_until_supply_end": days_remaining,
            "threshold": LOCKED_THRESHOLD,
        }
        if record.current_refills_remaining != 0:
            return PredictionResponse(
                **common, status="ineligible", eligible=False,
                renewal_probability=None, prediction=None, priority=None,
                review_status="refills_remaining",
            )
        priority = self._priority(days_remaining)
        if priority is None:
            review = "manual_review" if days_remaining < -7 else "not_in_prediction_window"
            return PredictionResponse(
                **common, status="ineligible", eligible=False,
                renewal_probability=None, prediction=None, priority=None,
                review_status=review,
            )
        raw_features = record.model_dump(include=set(FEATURE_COLUMNS))
        features = pd.DataFrame([raw_features], columns=FEATURE_COLUMNS)
        probability = float(self._model.predict_proba(features)[0, 1])
        prediction: Literal[0, 1] = 1 if probability >= LOCKED_THRESHOLD else 0
        return PredictionResponse(
            **common, status="success", eligible=True,
            renewal_probability=probability, prediction=prediction, priority=priority,
            review_status="prescription_review" if prediction else "no_review",
        )

    @staticmethod
    def _priority(
        days_remaining: int,
    ) -> Literal["low", "medium", "high", "urgent", "urgent_overdue"] | None:
        if 8 <= days_remaining <= 10:
            return "low"
        if 5 <= days_remaining <= 7:
            return "medium"
        if 1 <= days_remaining <= 4:
            return "high"
        if days_remaining == 0:
            return "urgent"
        if -7 <= days_remaining <= -1:
            return "urgent_overdue"
        return None
