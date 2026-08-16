"""FastAPI application for locked renewal inference."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI
from pydantic import ValidationError

from pharmacy_reconciliation.api.model_loader import load_locked_model
from pharmacy_reconciliation.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionError,
    PredictionRequest,
    PredictionResponse,
)
from pharmacy_reconciliation.api.service import PredictionService

LOGGER = logging.getLogger("pharmacy_reconciliation.api")
NEW_YORK = ZoneInfo("America/New_York")


def current_new_york_date() -> date:
    return datetime.now(NEW_YORK).date()


def create_app(
    model: Any | None = None,
    model_loader: Callable[[], Any] = load_locked_model,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            loaded = model if model is not None else model_loader()
            application.state.prediction_service = PredictionService(loaded)
        except Exception as error:
            raise RuntimeError("Locked model failed to load during application startup.") from error
        yield

    application = FastAPI(
        title="Pharmacy Renewal Review API",
        description="Local batch inference using the locked synthetic-data research model.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        record: PredictionRequest,
        today: Annotated[date, Depends(current_new_york_date)],
    ) -> PredictionResponse:
        return application.state.prediction_service.predict(record, today)

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        request: BatchPredictionRequest,
        today: Annotated[date, Depends(current_new_york_date)],
    ) -> BatchPredictionResponse:
        results: list[PredictionResponse | PredictionError] = []
        processed = eligible = flagged = 0
        for raw in request.records:
            try:
                record = PredictionRequest.model_validate(raw)
                result = application.state.prediction_service.predict(record, today)
                results.append(result)
                processed += 1
                eligible += int(result.eligible)
                flagged += int(result.prediction == 1)
            except ValidationError as error:
                first = error.errors(include_url=False)[0]
                field = ".".join(str(part) for part in first["loc"])
                rx_number = (
                    raw.get("rx_number")
                    if isinstance(raw, dict) and isinstance(raw.get("rx_number"), str)
                    else None
                )
                results.append(PredictionError(
                    rx_number=rx_number,
                    error_code="validation_error",
                    message=f"Invalid {field}: {first['msg']}",
                ))
        failed = len(request.records) - processed
        LOGGER.info(
            "Batch prediction completed records=%d processed=%d failed=%d",
            len(request.records), processed, failed,
        )
        return BatchPredictionResponse(
            total_records=len(request.records), processed=processed, failed=failed,
            eligible_records=eligible, flagged_for_review=flagged, results=results,
        )

    return application


app = create_app()
