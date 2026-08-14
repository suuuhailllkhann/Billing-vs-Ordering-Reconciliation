"""Testable controller connecting file ingestion, reconciliation, and analytics."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

import pandas as pd

from pharmacy_reconciliation.analytics.insurance import insurance_billing_summary
from pharmacy_reconciliation.analytics.patients import (
    patient_billing_details,
    patient_medication_summary,
)
from pharmacy_reconciliation.ingestion.loaders import (
    ingest_billing_file,
    ingest_ordering_file,
)
from pharmacy_reconciliation.ingestion.quality import IngestionResult
from pharmacy_reconciliation.reconciliation.reconcile import (
    ReconciliationResult,
    reconcile_inventory,
)


class WorkflowError(ValueError):
    """An understandable workflow problem suitable for presentation by a UI."""


@dataclass(frozen=True)
class DashboardResult:
    reconciliation: ReconciliationResult
    inventory: pd.DataFrame
    insurance: pd.DataFrame
    patient_summary: pd.DataFrame
    patient_details: pd.DataFrame


class ReconciliationController:
    def __init__(self) -> None:
        self.billing_ingestion: IngestionResult | None = None
        self.ordering_ingestion: IngestionResult | None = None
        self.dashboard_result: DashboardResult | None = None

    def load_billing(
        self,
        path: str,
        manual_mapping: Mapping[str, str] | None = None,
    ) -> IngestionResult:
        self.billing_ingestion = ingest_billing_file(path, manual_mapping)
        self.dashboard_result = None
        return self.billing_ingestion

    def load_ordering(
        self,
        path: str,
        manual_mapping: Mapping[str, str] | None = None,
    ) -> IngestionResult:
        self.ordering_ingestion = ingest_ordering_file(path, manual_mapping)
        self.dashboard_result = None
        return self.ordering_ingestion

    @property
    def inputs_ready(self) -> bool:
        return bool(
            self.billing_ingestion
            and self.billing_ingestion.ready_for_reconciliation
            and self.ordering_ingestion
            and self.ordering_ingestion.ready_for_reconciliation
        )

    def available_date_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if not self.inputs_ready:
            return None
        billing_ingestion = self.billing_ingestion
        ordering_ingestion = self.ordering_ingestion
        if (
            billing_ingestion is None
            or ordering_ingestion is None
            or billing_ingestion.canonical_data is None
            or ordering_ingestion.canonical_data is None
        ):
            return None
        billing = billing_ingestion.canonical_data["billing_date"].dropna()
        ordering = ordering_ingestion.canonical_data["ordered_date"].dropna()
        all_dates = pd.concat([billing, ordering], ignore_index=True)
        if all_dates.empty:
            return None
        return pd.Timestamp(all_dates.min()), pd.Timestamp(all_dates.max())

    def reconcile(
        self,
        start_date: str | date | datetime | pd.Timestamp,
        end_date: str | date | datetime | pd.Timestamp,
    ) -> DashboardResult:
        if not self.billing_ingestion:
            raise WorkflowError("Load a billing file before running reconciliation.")
        if not self.ordering_ingestion:
            raise WorkflowError("Load an ordering file before running reconciliation.")
        if not self.billing_ingestion.ready_for_reconciliation:
            raise WorkflowError("The billing file is not ready. Review its mapping and data-quality issues.")
        if not self.ordering_ingestion.ready_for_reconciliation:
            raise WorkflowError("The ordering file is not ready. Review its mapping and data-quality issues.")
        if self.billing_ingestion.canonical_data is None or self.ordering_ingestion.canonical_data is None:
            raise WorkflowError("Validated canonical data is unavailable. Reload both source files.")

        reconciliation = reconcile_inventory(
            self.billing_ingestion.canonical_data,
            self.ordering_ingestion.canonical_data,
            start_date,
            end_date,
        )
        result = DashboardResult(
            reconciliation=reconciliation,
            inventory=reconciliation.inventory,
            insurance=insurance_billing_summary(reconciliation.billing_in_period),
            patient_summary=patient_medication_summary(reconciliation.billing_in_period),
            patient_details=patient_billing_details(reconciliation.billing_in_period),
        )
        self.dashboard_result = result
        return result
