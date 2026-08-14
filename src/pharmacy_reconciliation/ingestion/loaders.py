"""Canonical and raw-export loaders with string-sensitive file reading."""

from pathlib import Path
from typing import Mapping

import pandas as pd

from pharmacy_reconciliation.ingestion.mapping import map_columns
from pharmacy_reconciliation.ingestion.quality import IngestionResult
from pharmacy_reconciliation.ingestion.validation import (
    ValidationResult,
    validate_billing,
    validate_ordering,
)


class UnsupportedFileTypeError(ValueError):
    """Raised when a requested source file is not CSV or XLSX."""


def _read_csv_as_strings(path: str | Path) -> pd.DataFrame:
    # Validation performs explicit date and numeric conversion after identifiers are safe.
    return pd.read_csv(path, dtype="string", keep_default_na=True)


def _read_raw_file(path: str | Path, sheet_name: str | int = 0) -> tuple[pd.DataFrame, str]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return _read_csv_as_strings(source), "csv"
    if suffix == ".xlsx":
        return (
            pd.read_excel(
                source,
                sheet_name=sheet_name,
                dtype="string",
                keep_default_na=True,
                engine="openpyxl",
            ),
            "xlsx",
        )
    raise UnsupportedFileTypeError(
        f"Unsupported file type `{suffix or '(none)'}`. Supported types are .csv and .xlsx."
    )


def _ingest_file(
    path: str | Path,
    dataset_type: str,
    manual_mapping: Mapping[str, str] | None = None,
    sheet_name: str | int = 0,
) -> IngestionResult:
    raw, file_type = _read_raw_file(path, sheet_name)
    mapping = map_columns(raw.columns.tolist(), dataset_type, manual_mapping)
    canonical_data = None
    validation = None
    if mapping.is_unambiguous:
        canonical_data = raw.loc[:, list(mapping.resolved_mapping)].rename(
            columns=mapping.resolved_mapping
        ).copy()
        validation = (
            validate_billing(canonical_data)
            if dataset_type == "billing"
            else validate_ordering(canonical_data)
        )
        canonical_data = validation.data
    return IngestionResult(
        input_file=str(Path(path)),
        file_type=file_type,
        dataset_type=dataset_type,
        row_count=len(raw),
        original_columns=tuple(str(column) for column in raw.columns),
        mapping=mapping,
        canonical_data=canonical_data,
        validation=validation,
        normalization_steps=(
            "Headers normalized for comparison only; original header text preserved.",
            "Mapped string values trimmed and empty strings treated as missing.",
            "NDC and BIN loaded as strings; no padding or format conversion applied.",
            "Dates and numeric values parsed by Phase 1A validation.",
        ),
    )


def load_billing_csv(path: str | Path) -> ValidationResult:
    return validate_billing(_read_csv_as_strings(path))


def load_ordering_csv(path: str | Path) -> ValidationResult:
    return validate_ordering(_read_csv_as_strings(path))


def ingest_billing_file(
    path: str | Path,
    manual_mapping: Mapping[str, str] | None = None,
    sheet_name: str | int = 0,
) -> IngestionResult:
    return _ingest_file(path, "billing", manual_mapping, sheet_name)


def ingest_ordering_file(
    path: str | Path,
    manual_mapping: Mapping[str, str] | None = None,
    sheet_name: str | int = 0,
) -> IngestionResult:
    return _ingest_file(path, "ordering", manual_mapping, sheet_name)
