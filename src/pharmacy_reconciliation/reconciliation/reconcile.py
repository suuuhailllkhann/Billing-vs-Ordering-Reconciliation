"""Inventory reconciliation at medication level for one inclusive date window."""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from pharmacy_reconciliation.ingestion.validation import (
    DatasetValidationError,
    ValidationResult,
    validate_billing,
    validate_ordering,
)
from pharmacy_reconciliation.reconciliation.billing import aggregate_billing
from pharmacy_reconciliation.reconciliation.ordering import aggregate_ordering


@dataclass(frozen=True)
class PeriodCoverage:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    billing_rows_total: int
    billing_rows_in_period: int
    ordering_rows_total: int
    ordering_rows_in_period: int
    billing_date_min: pd.Timestamp | None
    billing_date_max: pd.Timestamp | None
    ordered_date_min: pd.Timestamp | None
    ordered_date_max: pd.Timestamp | None

    @property
    def coverage_differs_materially(self) -> bool:
        """Flag obvious non-overlap; a business-specific tolerance is not assumed."""
        billing_min = self.billing_date_min
        billing_max = self.billing_date_max
        ordering_min = self.ordered_date_min
        ordering_max = self.ordered_date_max
        billing_missing = billing_min is None or billing_max is None
        ordering_missing = ordering_min is None or ordering_max is None
        if billing_missing or ordering_missing:
            return billing_missing != ordering_missing
        assert billing_min is not None and billing_max is not None
        assert ordering_min is not None and ordering_max is not None
        return billing_min > ordering_max or ordering_min > billing_max


@dataclass(frozen=True)
class ReconciliationResult:
    inventory: pd.DataFrame
    billing_in_period: pd.DataFrame
    ordering_in_period: pd.DataFrame
    billing_validation: ValidationResult
    ordering_validation: ValidationResult
    coverage: PeriodCoverage


DateBoundary = str | date | datetime | pd.Timestamp


def _normalize_boundary(value: DateBoundary, label: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid date.") from exc


def _date_or_none(series: pd.Series, operation: str) -> pd.Timestamp | None:
    if series.empty:
        return None
    value = getattr(series, operation)()
    return None if pd.isna(value) else pd.Timestamp(value)


def filter_inclusive_period(
    frame: pd.DataFrame,
    date_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    dates = frame[date_column].dt.normalize()
    return frame.loc[dates.between(start_date, end_date, inclusive="both")].copy()


def reconcile_inventory(
    billing: pd.DataFrame,
    ordering: pd.DataFrame,
    start_date: DateBoundary,
    end_date: DateBoundary,
) -> ReconciliationResult:
    """Reconcile all medications using an inclusive common date range.

    Validation errors stop reconciliation so bad records are neither discarded nor
    silently included. Exact duplicate candidates are warnings and remain present
    until a business owner decides whether they are legitimate transactions.
    """
    start = _normalize_boundary(start_date, "start_date")
    end = _normalize_boundary(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date.")

    billing_validation = validate_billing(billing)
    ordering_validation = validate_ordering(ordering)
    if billing_validation.has_errors:
        raise DatasetValidationError("billing", billing_validation)
    if ordering_validation.has_errors:
        raise DatasetValidationError("ordering", ordering_validation)

    billing_clean = billing_validation.data
    ordering_clean = ordering_validation.data
    billing_period = filter_inclusive_period(billing_clean, "billing_date", start, end)
    ordering_period = filter_inclusive_period(ordering_clean, "ordered_date", start, end)

    billed = aggregate_billing(billing_period)
    ordered = aggregate_ordering(ordering_period)
    inventory = billed.merge(ordered, on=["ndc", "drug_name"], how="outer")
    inventory[["total_billed", "total_ordered"]] = inventory[
        ["total_billed", "total_ordered"]
    ].fillna(0)
    inventory["net_difference"] = inventory["total_ordered"] - inventory["total_billed"]
    inventory["short_quantity"] = (-inventory["net_difference"]).clip(lower=0)
    inventory["extra_quantity"] = inventory["net_difference"].clip(lower=0)
    inventory["status"] = "MATCHED"
    inventory.loc[inventory["net_difference"] < 0, "status"] = "SHORT"
    inventory.loc[inventory["net_difference"] > 0, "status"] = "EXTRA"
    inventory = inventory.sort_values(["ndc", "drug_name"], kind="stable").reset_index(drop=True)

    coverage = PeriodCoverage(
        start_date=start,
        end_date=end,
        billing_rows_total=len(billing_clean),
        billing_rows_in_period=len(billing_period),
        ordering_rows_total=len(ordering_clean),
        ordering_rows_in_period=len(ordering_period),
        billing_date_min=_date_or_none(billing_period["billing_date"], "min"),
        billing_date_max=_date_or_none(billing_period["billing_date"], "max"),
        ordered_date_min=_date_or_none(ordering_period["ordered_date"], "min"),
        ordered_date_max=_date_or_none(ordering_period["ordered_date"], "max"),
    )
    return ReconciliationResult(
        inventory, billing_period, ordering_period,
        billing_validation, ordering_validation, coverage,
    )
