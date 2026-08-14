from pathlib import Path

import pandas as pd
import pytest

from pharmacy_reconciliation.ingestion.loaders import (
    UnsupportedFileTypeError,
    ingest_billing_file,
    ingest_ordering_file,
)
from pharmacy_reconciliation.ingestion.mapping import MappingStatus

SYNTHETIC_DIR = Path(__file__).parents[1] / "data" / "synthetic"


def test_clean_alias_csv_is_ready_and_preserves_identifiers():
    result = ingest_billing_file(SYNTHETIC_DIR / "billing_export_clean_aliases.csv")
    assert result.ready_for_reconciliation
    assert result.file_type == "csv"
    assert result.row_count == 2
    assert result.canonical_data.loc[0, "ndc"] == "00000001001"
    assert result.canonical_data.loc[0, "bin_number"] == "001234"
    assert "Store Number" in result.original_columns
    store = next(item for item in result.mapping.columns if item.source_column == "Store Number")
    assert store.status == MappingStatus.UNMAPPED


def test_messy_headers_transform_to_canonical_schema():
    result = ingest_billing_file(SYNTHETIC_DIR / "billing_export_messy_headers.csv")
    assert result.ready_for_reconciliation
    assert result.canonical_data.loc[0, "ndc"] == "00000002001"
    assert result.canonical_data.loc[0, "bin_number"] == "000321"
    assert result.canonical_data.loc[0, "quantity_billed"] == 15
    source_mapping = next(
        item for item in result.mapping.columns if item.source_column == "DISPENSED-QUANTITY"
    )
    assert source_mapping.canonical_field == "quantity_billed"
    assert source_mapping.status == MappingStatus.ALIAS


def test_ambiguous_export_is_not_ready_until_manually_resolved():
    path = SYNTHETIC_DIR / "billing_export_ambiguous.csv"
    ambiguous = ingest_billing_file(path)
    assert not ambiguous.ready_for_reconciliation
    assert ambiguous.canonical_data is None
    assert ambiguous.mapping.conflicts

    resolved = ingest_billing_file(path, {"Qty Dispensed": "quantity_billed"})
    assert resolved.ready_for_reconciliation
    assert resolved.canonical_data.loc[0, "quantity_billed"] == 12


def test_missing_required_mapping_blocks_readiness():
    result = ingest_billing_file(SYNTHETIC_DIR / "billing_export_missing_required.csv")
    assert not result.ready_for_reconciliation
    assert "quantity_billed" in result.mapping.required_fields_missing
    assert result.validation is None


def test_missing_optional_mappings_do_not_block_readiness():
    result = ingest_billing_file(SYNTHETIC_DIR / "billing_export_optional_missing.csv")
    assert result.ready_for_reconciliation
    assert pd.isna(result.canonical_data.loc[0, "patient_name"])
    assert pd.isna(result.canonical_data.loc[0, "insurance_name"])


def test_validation_errors_block_ready_state_and_populate_report():
    result = ingest_billing_file(SYNTHETIC_DIR / "billing_export_invalid_values.csv")
    assert not result.ready_for_reconciliation
    assert result.validation is not None
    assert result.invalid_row_count == 1
    assert result.report["ready_for_reconciliation"] is False
    assert result.report["validation"]["invalid_numeric"] == 1
    assert result.report["validation"]["invalid_date"] == 1


def test_order_csv_with_unknown_column_is_ready():
    result = ingest_ordering_file(SYNTHETIC_DIR / "orders_export_messy_headers.csv")
    assert result.ready_for_reconciliation
    assert result.canonical_data.loc[0, "ndc"] == "00000001001"
    assert result.canonical_data.loc[0, "quantity_ordered"] == 30
    assert result.mapping.unmapped_count == 1


def test_ambiguous_order_export_is_not_ready():
    result = ingest_ordering_file(SYNTHETIC_DIR / "orders_export_ambiguous.csv")
    assert not result.ready_for_reconciliation
    assert result.mapping.ambiguous_count == 2


def test_xlsx_input_matches_csv_behavior():
    csv_result = ingest_billing_file(SYNTHETIC_DIR / "billing_export_clean_aliases.csv")
    xlsx_result = ingest_billing_file(SYNTHETIC_DIR / "billing_export_clean_aliases.xlsx")
    assert xlsx_result.file_type == "xlsx"
    assert xlsx_result.ready_for_reconciliation
    pd.testing.assert_frame_equal(xlsx_result.canonical_data, csv_result.canonical_data)


def test_unsupported_file_type_has_clear_error():
    source = Path(__file__).parents[1] / "README.md"
    with pytest.raises(UnsupportedFileTypeError, match="Supported types are .csv and .xlsx"):
        ingest_billing_file(source)


def test_empty_export_is_not_ready():
    result = ingest_ordering_file(SYNTHETIC_DIR / "orders_export_empty.csv")
    assert result.row_count == 0
    assert not result.ready_for_reconciliation
