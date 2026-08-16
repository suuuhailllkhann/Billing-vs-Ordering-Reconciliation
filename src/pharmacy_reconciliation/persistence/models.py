"""SQLAlchemy models for append-only prediction and follow-up history."""

import uuid
from datetime import date, datetime, timezone
from typing import ClassVar

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


PRIORITIES = ("low", "medium", "high", "urgent", "urgent_overdue")
ACTIVITY_TYPES = (
    "called_prescriber", "left_voicemail", "fax_sent", "message_sent",
    "spoke_with_prescriber", "patient_contacted", "other",
)
RESOLUTION_TYPES = (
    "new_prescription_received", "medication_discontinued", "dose_changed",
    "medication_changed", "patient_changed_pharmacy", "patient_unreachable",
    "prescriber_no_response", "other",
)


class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    __table_args__: ClassVar = (
        CheckConstraint("prediction IN (0, 1)", name="ck_prediction_records_prediction"),
        CheckConstraint("threshold >= 0 AND threshold <= 1", name="ck_prediction_records_threshold"),
        CheckConstraint(
            "priority IN ('low','medium','high','urgent','urgent_overdue')",
            name="ck_prediction_records_priority",
        ),
    )

    prediction_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    rx_number: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expected_supply_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    days_until_supply_end: Mapped[int] = mapped_column(Integer, nullable=False)
    renewal_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    priority: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class FollowUpCase(Base):
    __tablename__ = "follow_up_cases"
    __table_args__: ClassVar = (
        CheckConstraint("status IN ('open', 'resolved')", name="ck_follow_up_cases_status"),
        CheckConstraint(
            "priority IN ('low','medium','high','urgent','urgent_overdue')",
            name="ck_follow_up_cases_priority",
        ),
        Index(
            "uq_follow_up_cases_open_rx",
            "rx_number",
            unique=True,
            postgresql_where=text("status = 'open'"),
            sqlite_where=text("status = 'open'"),
        ),
    )

    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    rx_number: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    initial_prediction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("prediction_records.prediction_id"), nullable=False
    )
    latest_prediction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("prediction_records.prediction_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    priority: Mapped[str] = mapped_column(String(32), nullable=False)
    latest_renewal_probability: Mapped[float] = mapped_column(Float, nullable=False)
    latest_days_until_supply_end: Mapped[int] = mapped_column(Integer, nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    activities: Mapped[list["FollowUpActivity"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    resolution: Mapped["CaseResolution | None"] = relationship(
        back_populates="case", uselist=False, cascade="all, delete-orphan"
    )


class FollowUpActivity(Base):
    __tablename__ = "follow_up_activities"
    __table_args__: ClassVar = (
        CheckConstraint(
            "activity_type IN ('called_prescriber','left_voicemail','fax_sent',"
            "'message_sent','spoke_with_prescriber','patient_contacted','other')",
            name="ck_follow_up_activities_type",
        ),
    )

    activity_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("follow_up_cases.case_id", ondelete="CASCADE"), nullable=False
    )
    activity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    activity_note: Mapped[str | None] = mapped_column(Text)
    performed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    performed_by: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    case: Mapped[FollowUpCase] = relationship(back_populates="activities")


class CaseResolution(Base):
    __tablename__ = "case_resolutions"
    __table_args__: ClassVar = (
        CheckConstraint(
            "resolution_type IN ('new_prescription_received','medication_discontinued',"
            "'dose_changed','medication_changed','patient_changed_pharmacy',"
            "'patient_unreachable','prescriber_no_response','other')",
            name="ck_case_resolutions_type",
        ),
        CheckConstraint(
            "resolution_type <> 'other' OR length(trim(resolution_note)) > 0",
            name="ck_case_resolutions_other_note",
        ),
    )

    resolution_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("follow_up_cases.case_id", ondelete="CASCADE"), unique=True
    )
    resolution_type: Mapped[str] = mapped_column(String(48), nullable=False)
    resolution_note: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    case: Mapped[FollowUpCase] = relationship(back_populates="resolution")
