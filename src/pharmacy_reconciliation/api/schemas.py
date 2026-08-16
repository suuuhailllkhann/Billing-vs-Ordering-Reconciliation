"""Public request and response schemas for batch renewal inference."""

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

PREDICTION_REQUEST_EXAMPLE: dict[str, JsonValue] = {
    "rx_number": "RX-DEMO-001",
    "fill_date": "2026-08-01",
    "current_refills_remaining": 0,
    "current_quantity_billed": 30.0,
    "current_days_supply": 30,
    "current_refill_number": 2,
    "previous_fill_count": 4,
    "days_since_previous_fill": 31.0,
    "average_previous_refill_interval_days": 30.5,
    "std_previous_refill_interval_days": 2.1,
    "latest_refill_timing_gap_days": 1.0,
    "average_previous_timing_gap_days": 0.5,
    "previous_early_fill_rate": 0.25,
    "previous_on_time_fill_rate": 0.75,
    "prescription_age_days": 150.0,
    "medication_prior_fill_count": 120,
    "medication_prior_average_days_supply": 30.0,
    "medication_prior_average_quantity": 30.0,
}

BATCH_PREDICTION_REQUEST_EXAMPLE: dict[str, JsonValue] = {
    "records": [
        PREDICTION_REQUEST_EXAMPLE,
        {
            **PREDICTION_REQUEST_EXAMPLE,
            "rx_number": "RX-DEMO-002",
            "fill_date": "2026-07-20",
            "current_quantity_billed": 90.0,
            "current_days_supply": 30,
            "current_refill_number": 3,
            "previous_fill_count": 6,
            "days_since_previous_fill": 29.0,
            "average_previous_refill_interval_days": 29.8,
            "std_previous_refill_interval_days": None,
            "previous_early_fill_rate": 0.33,
            "previous_on_time_fill_rate": 0.67,
            "prescription_age_days": 210.0,
            "medication_prior_average_quantity": 75.0,
        },
    ]
}


class PredictionRequest(BaseModel):
    """Raw source facts required by the locked model and business rules."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": PREDICTION_REQUEST_EXAMPLE},
    )

    rx_number: str = Field(min_length=1, description="Audit lookup only; never a model feature.")
    fill_date: date
    current_refills_remaining: Annotated[int, Field(ge=0)]
    current_quantity_billed: Annotated[float, Field(ge=0)]
    current_days_supply: Annotated[int, Field(gt=0)]
    current_refill_number: Annotated[int, Field(ge=0)]
    previous_fill_count: Annotated[int, Field(ge=0)]
    days_since_previous_fill: Annotated[float, Field(ge=0)]
    average_previous_refill_interval_days: Annotated[float, Field(ge=0)]
    std_previous_refill_interval_days: Annotated[float | None, Field(ge=0)] = None
    latest_refill_timing_gap_days: float
    average_previous_timing_gap_days: float
    previous_early_fill_rate: Annotated[float, Field(ge=0, le=1)]
    previous_on_time_fill_rate: Annotated[float, Field(ge=0, le=1)]
    prescription_age_days: Annotated[float, Field(ge=0)]
    medication_prior_fill_count: Annotated[int, Field(ge=0)]
    medication_prior_average_days_supply: Annotated[float, Field(gt=0)]
    medication_prior_average_quantity: Annotated[float, Field(ge=0)]


class PredictionResponse(BaseModel):
    rx_number: str
    status: Literal["success", "ineligible"]
    eligible: bool
    expected_supply_end_date: date
    days_until_supply_end: int
    renewal_probability: float | None
    threshold: float
    prediction: Literal[0, 1] | None
    priority: Literal["low", "medium", "high", "urgent", "urgent_overdue"] | None
    review_status: Literal[
        "prescription_review", "no_review", "refills_remaining",
        "not_in_prediction_window", "manual_review",
    ]


class PredictionError(BaseModel):
    rx_number: str | None = None
    status: Literal["error"] = "error"
    error_code: str
    message: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": BATCH_PREDICTION_REQUEST_EXAMPLE},
    )
    records: Annotated[list[Any], Field(min_length=1, max_length=500)]


class BatchPredictionResponse(BaseModel):
    total_records: int
    processed: int
    failed: int
    eligible_records: int
    flagged_for_review: int
    results: list[PredictionResponse | PredictionError]


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
    model_loaded: Literal[True] = True
    model_status: Literal["locked_final"] = "locked_final"
