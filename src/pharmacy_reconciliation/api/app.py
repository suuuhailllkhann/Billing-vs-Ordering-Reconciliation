"""FastAPI application for locked renewal inference."""

import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from time import perf_counter
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from pharmacy_reconciliation.api.model_loader import load_locked_model
from pharmacy_reconciliation.api.schemas import (
    ActivityCreate,
    ActivityResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CaseResponse,
    HealthResponse,
    LivenessResponse,
    PredictionError,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
    ResolutionCreate,
    ResolutionResponse,
)
from pharmacy_reconciliation.api.service import PredictionService
from pharmacy_reconciliation.persistence.config import (
    create_database_engine,
    create_session_factory,
    load_database_settings,
    verify_database,
)
from pharmacy_reconciliation.persistence.repository import (
    ConflictError,
    FollowUpRepository,
    NotFoundError,
)

LOGGER = logging.getLogger("uvicorn.error.pharmacy_reconciliation.api")
NEW_YORK = ZoneInfo("America/New_York")


def current_new_york_date() -> date:
    return datetime.now(NEW_YORK).date()


def current_utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def create_app(
    model: Any | None = None,
    model_loader: Callable[[], Any] = load_locked_model,
    session_factory: sessionmaker[Session] | None = None,
    persistence_enabled: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            loaded = model if model is not None else model_loader()
            application.state.prediction_service = PredictionService(loaded)
            application.state.model_run_id = getattr(
                loaded, "_pharmacy_model_run_id", "locked-final-test"
            )
            application.state.model_version = getattr(
                loaded, "_pharmacy_model_version", "locked-final"
            )
            application.state.persistence_enabled = persistence_enabled
            application.state.database_engine = None
            if persistence_enabled and session_factory is None:
                engine = create_database_engine(load_database_settings())
                verify_database(engine)
                application.state.database_engine = engine
                application.state.session_factory = create_session_factory(engine)
            else:
                application.state.session_factory = session_factory
        except Exception as error:
            raise RuntimeError(
                "Locked model failed to load or persistence failed during startup."
            ) from error
        yield
        if application.state.database_engine is not None:
            application.state.database_engine.dispose()

    application = FastAPI(
        title="Pharmacy Renewal Review API",
        description="Local batch inference using the locked synthetic-data research model.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_observability(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = str(uuid.uuid4())
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error."},
            )
        duration_ms = (perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "request_completed request_id=%s method=%s path=%s status_code=%d duration_ms=%.3f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @application.get("/health/live", response_model=LivenessResponse)
    def health_live() -> LivenessResponse:
        return LivenessResponse()

    @application.get("/health/ready", response_model=ReadinessResponse)
    def health_ready() -> ReadinessResponse | Response:
        readiness = _readiness(application)
        if readiness.status == "not_ready":
            return JSONResponse(status_code=503, content=readiness.model_dump())
        return readiness

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse | Response:
        readiness = _readiness(application)
        if readiness.status == "not_ready":
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "model_loaded": readiness.model_loaded,
                    "model_status": "locked_final",
                },
            )
        return HealthResponse()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        record: PredictionRequest,
        today: Annotated[date, Depends(current_new_york_date)],
        now: Annotated[datetime, Depends(current_utc_datetime)],
    ) -> PredictionResponse:
        result = application.state.prediction_service.predict(record, today)
        try:
            _persist_result(application, result, now)
        except Exception as error:
            LOGGER.error("Eligible prediction persistence failed error_code=database_error")
            raise HTTPException(status_code=503, detail="Prediction could not be persisted.") from error
        return result

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        request: BatchPredictionRequest,
        today: Annotated[date, Depends(current_new_york_date)],
        now: Annotated[datetime, Depends(current_utc_datetime)],
    ) -> BatchPredictionResponse:
        results: list[PredictionResponse | PredictionError] = []
        processed = eligible = flagged = 0
        for raw in request.records:
            try:
                record = PredictionRequest.model_validate(raw)
                result = application.state.prediction_service.predict(record, today)
                try:
                    _persist_result(application, result, now)
                except Exception:
                    LOGGER.error("Batch record persistence failed error_code=database_error")
                    results.append(PredictionError(
                        rx_number=record.rx_number,
                        error_code="persistence_error",
                        message="Prediction could not be persisted.",
                    ))
                    continue
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

    @application.get("/cases", response_model=list[CaseResponse])
    def list_cases(
        status: Annotated[str, Query(pattern="^(open|resolved)$")] = "open",
        priority: Annotated[
            str | None,
            Query(pattern="^(low|medium|high|urgent|urgent_overdue)$"),
        ] = None,
    ) -> list[CaseResponse]:
        with _required_session(application)() as session:
            cases = FollowUpRepository(session).list_cases(status, priority)
            return [CaseResponse.model_validate(case) for case in cases]

    @application.get("/cases/{case_id}", response_model=CaseResponse)
    def get_case(case_id: uuid.UUID) -> CaseResponse:
        with _required_session(application)() as session:
            try:
                return CaseResponse.model_validate(FollowUpRepository(session).get_case(case_id))
            except NotFoundError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get("/cases/{case_id}/activities", response_model=list[ActivityResponse])
    def list_activities(case_id: uuid.UUID) -> list[ActivityResponse]:
        with _required_session(application)() as session:
            try:
                activities = FollowUpRepository(session).list_activities(case_id)
                return [ActivityResponse.model_validate(item) for item in activities]
            except NotFoundError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error

    @application.post(
        "/cases/{case_id}/activities", response_model=ActivityResponse, status_code=201
    )
    def add_activity(
        case_id: uuid.UUID,
        request: ActivityCreate,
        now: Annotated[datetime, Depends(current_utc_datetime)],
    ) -> ActivityResponse:
        try:
            with _required_session(application).begin() as session:
                activity = FollowUpRepository(session).add_activity(
                    case_id,
                    activity_type=request.activity_type,
                    activity_note=request.activity_note,
                    performed_at=request.performed_at or now,
                    performed_by=request.performed_by,
                )
                response = ActivityResponse.model_validate(activity)
            return response
        except NotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post(
        "/cases/{case_id}/resolve", response_model=ResolutionResponse, status_code=201
    )
    def resolve_case(
        case_id: uuid.UUID,
        request: ResolutionCreate,
        now: Annotated[datetime, Depends(current_utc_datetime)],
    ) -> ResolutionResponse:
        if request.resolution_type == "other" and not (request.resolution_note or "").strip():
            raise HTTPException(
                status_code=422,
                detail="A resolution note is required when resolution_type is other.",
            )
        try:
            with _required_session(application).begin() as session:
                resolution = FollowUpRepository(session).resolve_case(
                    case_id,
                    resolution_type=request.resolution_type,
                    resolution_note=request.resolution_note,
                    resolved_at=now,
                )
                response = ResolutionResponse.model_validate(resolution)
            return response
        except NotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return application


def _readiness(application: FastAPI) -> ReadinessResponse:
    model_loaded = hasattr(application.state, "prediction_service")
    database_available = not application.state.persistence_enabled
    if application.state.persistence_enabled:
        factory = application.state.session_factory
        if factory is not None:
            try:
                with factory() as session:
                    session.execute(text("SELECT 1"))
                database_available = True
            except Exception:
                database_available = False
    return ReadinessResponse(
        status="ready" if model_loaded and database_available else "not_ready",
        model_loaded=model_loaded,
        database_available=database_available,
    )


def _required_session(application: FastAPI) -> sessionmaker[Session]:
    factory = application.state.session_factory
    if factory is None:
        raise HTTPException(status_code=503, detail="Persistence is unavailable.")
    return factory


def _persist_result(application: FastAPI, result: PredictionResponse, now: datetime) -> None:
    if not application.state.persistence_enabled or not result.eligible:
        return
    factory = _required_session(application)
    with factory.begin() as session:
        FollowUpRepository(session).persist_eligible_prediction(
            result,
            predicted_at=now,
            model_version=application.state.model_version,
            model_run_id=application.state.model_run_id,
        )


app = create_app()
