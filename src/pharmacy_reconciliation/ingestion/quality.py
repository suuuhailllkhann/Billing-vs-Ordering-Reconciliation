"""Structured file-ingestion and data-quality reporting for future UI clients."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.ingestion.mapping import MappingResult
from pharmacy_reconciliation.ingestion.validation import ValidationResult


@dataclass(frozen=True)
class IngestionResult:
    input_file: str
    file_type: str
    dataset_type: str
    row_count: int
    original_columns: tuple[str, ...]
    mapping: MappingResult
    canonical_data: pd.DataFrame | None
    validation: ValidationResult | None
    normalization_steps: tuple[str, ...]

    @property
    def ready_for_reconciliation(self) -> bool:
        return (
            self.row_count > 0
            and self.mapping.is_unambiguous
            and self.validation is not None
            and not self.validation.has_errors
        )

    @property
    def valid_row_count(self) -> int:
        return self.validation.valid_row_count if self.validation else 0

    @property
    def invalid_row_count(self) -> int:
        return self.validation.invalid_row_count if self.validation else self.row_count

    @property
    def warning_count(self) -> int:
        if not self.validation:
            return 0
        return sum(issue.count for issue in self.validation.issues if issue.severity == "warning")

    @property
    def validation_summary(self) -> dict[str, int]:
        return self.validation.summary if self.validation else {}

    @property
    def report(self) -> dict[str, object]:
        """Serializable summary; detailed mappings/issues remain on this object."""
        return {
            "input_file": Path(self.input_file).name,
            "file_type": self.file_type,
            "dataset_type": self.dataset_type,
            "rows_read": self.row_count,
            "columns_detected": len(self.original_columns),
            "columns_mapped": self.mapping.mapped_count,
            "columns_ambiguous": self.mapping.ambiguous_count,
            "columns_unmapped": self.mapping.unmapped_count,
            "required_fields_missing": list(self.mapping.required_fields_missing),
            "mapping_conflict_count": len(self.mapping.conflicts),
            "valid_rows": self.valid_row_count,
            "invalid_rows": self.invalid_row_count,
            "warning_count": self.warning_count,
            "validation": self.validation_summary,
            "ready_for_reconciliation": self.ready_for_reconciliation,
        }
