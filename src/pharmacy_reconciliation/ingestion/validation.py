"""Deterministic normalization and validation for canonical DataFrames."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pharmacy_reconciliation.ingestion.schemas import (
    BILLING_COLUMNS,
    BILLING_REQUIRED_COLUMNS,
    ORDERING_COLUMNS,
    ORDERING_REQUIRED_COLUMNS,
)

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    field: str | None
    row_indices: tuple[object, ...]
    message: str

    @property
    def count(self) -> int:
        return len(self.row_indices)


@dataclass(frozen=True)
class ValidationResult:
    data: pd.DataFrame
    issues: tuple[ValidationIssue, ...]
    valid_row_mask: pd.Series

    @property
    def valid_row_count(self) -> int:
        return int(self.valid_row_mask.sum())

    @property
    def invalid_row_count(self) -> int:
        return int((~self.valid_row_mask).sum())

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def summary(self) -> dict[str, int]:
        counts = {"total_rows": len(self.data), "valid_rows": self.valid_row_count}
        for issue in self.issues:
            counts[issue.code] = counts.get(issue.code, 0) + issue.count
        return counts


class DatasetValidationError(ValueError):
    """Raised when reconciliation receives a dataset with validation errors."""

    def __init__(self, dataset: str, result: ValidationResult):
        self.dataset = dataset
        self.result = result
        details = "; ".join(issue.message for issue in result.issues if issue.severity == "error")
        super().__init__(f"{dataset} validation failed: {details}")


_BILLING_STRINGS = (
    "billing_id", "patient_id", "patient_name", "prescription_id", "ndc",
    "drug_name", "insurance_name", "bin_number",
)
_ORDERING_STRINGS = ("order_id", "ndc", "drug_name")


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _issue(code: str, severity: Severity, field: str | None, mask: pd.Series, message: str) -> ValidationIssue:
    return ValidationIssue(code, severity, field, tuple(mask.index[mask]), message)


def _validate(
    frame: pd.DataFrame,
    *,
    canonical_columns: tuple[str, ...],
    required_columns: tuple[str, ...],
    string_columns: tuple[str, ...],
    date_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    integer_columns: tuple[str, ...],
) -> ValidationResult:
    data = frame.copy()
    data.columns = data.columns.astype(str).str.strip().str.lower()
    issues: list[ValidationIssue] = []
    error_rows = pd.Series(False, index=data.index, dtype=bool)

    missing_columns = [column for column in required_columns if column not in data.columns]
    for column in missing_columns:
        issues.append(ValidationIssue(
            "missing_required_column", "error", column, tuple(data.index),
            f"Required column `{column}` is missing.",
        ))

    for column in canonical_columns:
        if column not in data.columns:
            data[column] = pd.NA
    data = data.loc[:, list(canonical_columns)]

    for column in string_columns:
        data[column] = data[column].astype("string").str.strip()

    for column in required_columns:
        missing = _missing_mask(data[column])
        if missing.any():
            issues.append(_issue(
                "missing_required_value", "error", column, missing,
                f"Required field `{column}` is missing in {int(missing.sum())} row(s).",
            ))
            error_rows |= missing

    for column in date_columns:
        source = data[column]
        present = ~_missing_mask(source)
        date_text = source.astype("string").str.strip()
        numeric_parts = date_text.str.extract(
            r"^(?P<first>\d{1,2})[/-](?P<second>\d{1,2})[/-](?P<year>\d{2,4})$"
        )
        first = pd.to_numeric(numeric_parts["first"], errors="coerce")
        second = pd.to_numeric(numeric_parts["second"], errors="coerce")
        ambiguous = (
            present & first.between(1, 12) & second.between(1, 12) & first.ne(second)
        ).fillna(False)
        if ambiguous.any():
            issues.append(_issue(
                "ambiguous_date", "error", column, ambiguous,
                f"Field `{column}` contains {int(ambiguous.sum())} ambiguous numeric date value(s); use an unambiguous format such as YYYY-MM-DD.",
            ))
            error_rows |= ambiguous
        parsed = pd.to_datetime(source, errors="coerce", format="mixed")
        invalid = present & parsed.isna()
        if invalid.any():
            issues.append(_issue(
                "invalid_date", "error", column, invalid,
                f"Field `{column}` contains {int(invalid.sum())} invalid date value(s).",
            ))
            error_rows |= invalid
        data[column] = parsed

    for column in numeric_columns:
        source = data[column]
        present = ~_missing_mask(source)
        parsed = pd.to_numeric(source, errors="coerce")
        invalid = present & parsed.isna()
        if invalid.any():
            issues.append(_issue(
                "invalid_numeric", "error", column, invalid,
                f"Field `{column}` contains {int(invalid.sum())} non-numeric value(s).",
            ))
            error_rows |= invalid
        negative = parsed.lt(0).fillna(False)
        if negative.any():
            issues.append(_issue(
                "negative_numeric", "error", column, negative,
                f"Field `{column}` contains {int(negative.sum())} negative value(s).",
            ))
            error_rows |= negative
        if column in integer_columns:
            non_integer = (parsed.notna() & parsed.mod(1).ne(0)).fillna(False)
            if non_integer.any():
                issues.append(_issue(
                    "invalid_integer", "error", column, non_integer,
                    f"Field `{column}` contains {int(non_integer.sum())} non-integer value(s).",
                ))
                error_rows |= non_integer
        data[column] = parsed

    duplicates = data.duplicated(keep=False)
    if duplicates.any():
        issues.append(_issue(
            "possible_duplicate", "warning", None, duplicates,
            f"Detected {int(duplicates.sum())} row(s) participating in exact duplicates; records were not removed.",
        ))

    if missing_columns and len(data):
        error_rows[:] = True
    return ValidationResult(data, tuple(issues), ~error_rows)


def validate_billing(frame: pd.DataFrame) -> ValidationResult:
    return _validate(
        frame,
        canonical_columns=BILLING_COLUMNS,
        required_columns=BILLING_REQUIRED_COLUMNS,
        string_columns=_BILLING_STRINGS,
        date_columns=("date_of_birth", "billing_date"),
        numeric_columns=("quantity_billed", "days_supply", "refills_remaining"),
        integer_columns=("days_supply", "refills_remaining"),
    )


def validate_ordering(frame: pd.DataFrame) -> ValidationResult:
    return _validate(
        frame,
        canonical_columns=ORDERING_COLUMNS,
        required_columns=ORDERING_REQUIRED_COLUMNS,
        string_columns=_ORDERING_STRINGS,
        date_columns=("ordered_date",),
        numeric_columns=("quantity_ordered",),
        integer_columns=(),
    )
