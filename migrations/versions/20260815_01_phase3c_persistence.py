"""Create Phase 3C prediction and follow-up tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260815_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_records",
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column("rx_number", sa.String(128), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expected_supply_end_date", sa.Date(), nullable=False),
        sa.Column("days_until_supply_end", sa.Integer(), nullable=False),
        sa.Column("renewal_probability", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("priority", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("model_run_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("prediction IN (0, 1)", name="ck_prediction_records_prediction"),
        sa.CheckConstraint(
            "threshold >= 0 AND threshold <= 1", name="ck_prediction_records_threshold"
        ),
        sa.CheckConstraint(
            "priority IN ('low','medium','high','urgent','urgent_overdue')",
            name="ck_prediction_records_priority",
        ),
        sa.PrimaryKeyConstraint("prediction_id"),
    )
    op.create_index("ix_prediction_records_rx_number", "prediction_records", ["rx_number"])
    op.create_table(
        "follow_up_cases",
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("rx_number", sa.String(128), nullable=False),
        sa.Column("initial_prediction_id", sa.Uuid(), nullable=False),
        sa.Column("latest_prediction_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("priority", sa.String(32), nullable=False),
        sa.Column("latest_renewal_probability", sa.Float(), nullable=False),
        sa.Column("latest_days_until_supply_end", sa.Integer(), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('open', 'resolved')", name="ck_follow_up_cases_status"),
        sa.CheckConstraint(
            "priority IN ('low','medium','high','urgent','urgent_overdue')",
            name="ck_follow_up_cases_priority",
        ),
        sa.ForeignKeyConstraint(["initial_prediction_id"], ["prediction_records.prediction_id"]),
        sa.ForeignKeyConstraint(["latest_prediction_id"], ["prediction_records.prediction_id"]),
        sa.PrimaryKeyConstraint("case_id"),
    )
    op.create_index("ix_follow_up_cases_rx_number", "follow_up_cases", ["rx_number"])
    op.create_index(
        "uq_follow_up_cases_open_rx",
        "follow_up_cases",
        ["rx_number"],
        unique=True,
        postgresql_where=sa.text("status = 'open'"),
    )
    op.create_table(
        "follow_up_activities",
        sa.Column("activity_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("activity_type", sa.String(32), nullable=False),
        sa.Column("activity_note", sa.Text()),
        sa.Column("performed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("performed_by", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "activity_type IN ('called_prescriber','left_voicemail','fax_sent',"
            "'message_sent','spoke_with_prescriber','patient_contacted','other')",
            name="ck_follow_up_activities_type",
        ),
        sa.ForeignKeyConstraint(["case_id"], ["follow_up_cases.case_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("activity_id"),
    )
    op.create_table(
        "case_resolutions",
        sa.Column("resolution_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("resolution_type", sa.String(48), nullable=False),
        sa.Column("resolution_note", sa.Text()),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "resolution_type IN ('new_prescription_received','medication_discontinued',"
            "'dose_changed','medication_changed','patient_changed_pharmacy',"
            "'patient_unreachable','prescriber_no_response','other')",
            name="ck_case_resolutions_type",
        ),
        sa.CheckConstraint(
            "resolution_type <> 'other' OR length(trim(resolution_note)) > 0",
            name="ck_case_resolutions_other_note",
        ),
        sa.ForeignKeyConstraint(["case_id"], ["follow_up_cases.case_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("resolution_id"),
        sa.UniqueConstraint("case_id"),
    )


def downgrade() -> None:
    op.drop_table("case_resolutions")
    op.drop_table("follow_up_activities")
    op.drop_index("uq_follow_up_cases_open_rx", table_name="follow_up_cases")
    op.drop_index("ix_follow_up_cases_rx_number", table_name="follow_up_cases")
    op.drop_table("follow_up_cases")
    op.drop_index("ix_prediction_records_rx_number", table_name="prediction_records")
    op.drop_table("prediction_records")
