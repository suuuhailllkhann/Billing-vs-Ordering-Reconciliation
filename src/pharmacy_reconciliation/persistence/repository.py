"""Transactional persistence and follow-up workflow operations."""

import uuid
from datetime import datetime

from sqlalchemy import case as sql_case
from sqlalchemy import select
from sqlalchemy.orm import Session

from pharmacy_reconciliation.api.schemas import PredictionResponse
from pharmacy_reconciliation.persistence.models import (
    CaseResolution,
    FollowUpActivity,
    FollowUpCase,
    PredictionRecord,
)


class PersistenceError(RuntimeError):
    pass


class NotFoundError(PersistenceError):
    pass


class ConflictError(PersistenceError):
    pass


class FollowUpRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def persist_eligible_prediction(
        self,
        result: PredictionResponse,
        *,
        predicted_at: datetime,
        model_version: str,
        model_run_id: str,
    ) -> PredictionRecord:
        if not result.eligible or result.prediction is None:
            raise ValueError("Only eligible predictions may be persisted.")
        if result.renewal_probability is None or result.priority is None:
            raise ValueError("Eligible prediction is missing required inference outputs.")
        probability = result.renewal_probability
        priority = result.priority
        record = PredictionRecord(
            rx_number=result.rx_number,
            predicted_at=predicted_at,
            expected_supply_end_date=result.expected_supply_end_date,
            days_until_supply_end=result.days_until_supply_end,
            renewal_probability=probability,
            threshold=result.threshold,
            prediction=result.prediction,
            priority=priority,
            model_version=model_version,
            model_run_id=model_run_id,
        )
        self.session.add(record)
        self.session.flush()
        if result.prediction == 1:
            open_case = self.session.scalar(
                select(FollowUpCase).where(
                    FollowUpCase.rx_number == result.rx_number,
                    FollowUpCase.status == "open",
                )
            )
            if open_case is None:
                self.session.add(FollowUpCase(
                    rx_number=result.rx_number,
                    initial_prediction_id=record.prediction_id,
                    latest_prediction_id=record.prediction_id,
                    status="open",
                    priority=priority,
                    latest_renewal_probability=probability,
                    latest_days_until_supply_end=result.days_until_supply_end,
                    opened_at=predicted_at,
                    last_evaluated_at=predicted_at,
                ))
            else:
                open_case.latest_prediction_id = record.prediction_id
                open_case.latest_renewal_probability = probability
                open_case.latest_days_until_supply_end = result.days_until_supply_end
                open_case.priority = priority
                open_case.last_evaluated_at = predicted_at
        return record

    def list_cases(self, status: str | None, priority: str | None) -> list[FollowUpCase]:
        priority_order = sql_case(
            (FollowUpCase.priority == "urgent_overdue", 1),
            (FollowUpCase.priority == "urgent", 2),
            (FollowUpCase.priority == "high", 3),
            (FollowUpCase.priority == "medium", 4),
            (FollowUpCase.priority == "low", 5),
            else_=6,
        )
        query = select(FollowUpCase)
        if status is not None:
            query = query.where(FollowUpCase.status == status)
        if priority is not None:
            query = query.where(FollowUpCase.priority == priority)
        query = query.order_by(
            priority_order,
            FollowUpCase.latest_days_until_supply_end.asc(),
            FollowUpCase.latest_renewal_probability.desc(),
            FollowUpCase.opened_at.asc(),
        )
        return list(self.session.scalars(query))

    def get_case(self, case_id: uuid.UUID) -> FollowUpCase:
        case = self.session.get(FollowUpCase, case_id)
        if case is None:
            raise NotFoundError("Follow-up case was not found.")
        return case

    def list_activities(self, case_id: uuid.UUID) -> list[FollowUpActivity]:
        self.get_case(case_id)
        return list(self.session.scalars(
            select(FollowUpActivity)
            .where(FollowUpActivity.case_id == case_id)
            .order_by(FollowUpActivity.performed_at, FollowUpActivity.created_at)
        ))

    def add_activity(
        self,
        case_id: uuid.UUID,
        *,
        activity_type: str,
        activity_note: str | None,
        performed_at: datetime,
        performed_by: str | None,
    ) -> FollowUpActivity:
        case = self.get_case(case_id)
        if case.status != "open":
            raise ConflictError("Activities cannot be added to a resolved case.")
        activity = FollowUpActivity(
            case_id=case_id,
            activity_type=activity_type,
            activity_note=activity_note,
            performed_at=performed_at,
            performed_by=performed_by,
        )
        self.session.add(activity)
        self.session.flush()
        return activity

    def resolve_case(
        self,
        case_id: uuid.UUID,
        *,
        resolution_type: str,
        resolution_note: str | None,
        resolved_at: datetime,
    ) -> CaseResolution:
        case = self.get_case(case_id)
        if case.status != "open" or case.resolution is not None:
            raise ConflictError("Follow-up case is already resolved.")
        if resolution_type == "other" and not (resolution_note or "").strip():
            raise ValueError("A resolution note is required when resolution_type is other.")
        resolution = CaseResolution(
            case_id=case_id,
            resolution_type=resolution_type,
            resolution_note=resolution_note,
            resolved_at=resolved_at,
        )
        case.status = "resolved"
        case.resolved_at = resolved_at
        self.session.add(resolution)
        self.session.flush()
        return resolution
