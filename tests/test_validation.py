from pathlib import Path

import pandas as pd
import pytest

from pharmacy_reconciliation.ingestion.loaders import load_billing_csv, load_ordering_csv
from pharmacy_reconciliation.ingestion.validation import (
    DatasetValidationError,
    validate_ordering,
)

SYNTHETIC_DIR = Path(__file__).parents[1] / "data" / "synthetic"


def _codes(result):
    return {issue.code for issue in result.issues}


def test_missing_required_column_has_understandable_error(valid_ordering_result):
    frame = valid_ordering_result.data.drop(columns="quantity_ordered")
    result = validate_ordering(frame)
    assert result.has_errors
    assert "missing_required_column" in _codes(result)
    assert any(issue.field == "quantity_ordered" for issue in result.issues)


def test_invalid_numeric_negative_date_and_missing_identifier_are_reported():
    billing = load_billing_csv(SYNTHETIC_DIR / "billing_validation_cases.csv")
    ordering = load_ordering_csv(SYNTHETIC_DIR / "orders_validation_cases.csv")
    assert {"invalid_numeric", "negative_numeric", "invalid_date", "missing_required_value"} <= _codes(billing)
    assert {"invalid_numeric", "negative_numeric", "invalid_date", "missing_required_value"} <= _codes(ordering)


def test_exact_duplicate_candidates_are_flagged_not_removed():
    result = load_billing_csv(SYNTHETIC_DIR / "billing_validation_cases.csv")
    duplicate_issue = next(issue for issue in result.issues if issue.code == "possible_duplicate")
    assert duplicate_issue.severity == "warning"
    assert duplicate_issue.count == 2
    assert len(result.data) == 4


def test_ndc_is_string_leading_zero_preserved_and_whitespace_trimmed(valid_billing_result, valid_ordering_result):
    assert str(valid_billing_result.data["ndc"].dtype) == "string"
    assert str(valid_ordering_result.data["ndc"].dtype) == "string"
    assert valid_billing_result.data.loc[0, "ndc"] == "00000000001"
    assert valid_billing_result.data.loc[3, "ndc"] == "00000000003"
    assert valid_billing_result.data.loc[3, "drug_name"] == "Citrine Solution"
    assert valid_ordering_result.data.loc[2, "ndc"] == "00000000003"


def test_reconciliation_refuses_invalid_rows(valid_ordering_result):
    from pharmacy_reconciliation.reconciliation.reconcile import reconcile_inventory

    invalid_billing = load_billing_csv(SYNTHETIC_DIR / "billing_validation_cases.csv")
    with pytest.raises(DatasetValidationError, match="billing validation failed"):
        reconcile_inventory(invalid_billing.data, valid_ordering_result.data, "2026-07-01", "2026-07-31")


def test_surrounding_whitespace_is_normalized_for_strings():
    frame = pd.DataFrame({
        "order_id": [" O-1 "],
        "ndc": [" 000123 "],
        "drug_name": [" Example Drug "],
        "ordered_date": ["2026-07-01"],
        "quantity_ordered": ["1"],
    })
    result = validate_ordering(frame)
    assert not result.has_errors
    assert result.data.loc[0, ["order_id", "ndc", "drug_name"]].tolist() == [
        "O-1", "000123", "Example Drug",
    ]


def test_ambiguous_numeric_date_is_reported_not_silently_accepted():
    frame = pd.DataFrame({
        "order_id": ["O-2"],
        "ndc": ["000124"],
        "drug_name": ["Another Example Drug"],
        "ordered_date": ["07/08/2026"],
        "quantity_ordered": ["1"],
    })
    result = validate_ordering(frame)
    assert result.has_errors
    assert "ambiguous_date" in _codes(result)
