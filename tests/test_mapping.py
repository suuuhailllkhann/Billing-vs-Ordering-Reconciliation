import pytest

from pharmacy_reconciliation.ingestion.mapping import (
    ManualMappingError,
    MappingStatus,
    map_columns,
)
from pharmacy_reconciliation.ingestion.normalization import normalize_header


@pytest.mark.parametrize(
    "header",
    [" NDC Number ", "NDC_NUMBER", "ndc-number", "Ndc Number", "...NDC   Number!!!"],
)
def test_header_normalization_variants(header):
    assert normalize_header(header) == "ndc_number"


def test_exact_and_alias_mapping_are_distinguished():
    result = map_columns(["quantity_billed", "NDC Number"], "billing")
    statuses = {item.source_column: item.status for item in result.columns}
    assert statuses["quantity_billed"] == MappingStatus.EXACT
    assert statuses["NDC Number"] == MappingStatus.ALIAS


def test_unknown_columns_are_unmapped_without_conflict():
    result = map_columns(["Store Number", "Internal Comment"], "billing")
    assert all(item.status == MappingStatus.UNMAPPED for item in result.columns)
    assert not result.conflicts


def test_two_aliases_for_one_field_are_ambiguous():
    result = map_columns(["Qty Dispensed", "Billed Qty"], "billing")
    assert result.ambiguous_count == 2
    assert any(conflict.code == "multiple_sources_for_field" for conflict in result.conflicts)
    assert "quantity_billed" in result.required_fields_missing


def test_duplicate_normalized_headers_are_conflicts():
    result = map_columns(["NDC Number", "ndc-number"], "billing")
    assert result.ambiguous_count == 2
    assert any(conflict.code == "duplicate_normalized_header" for conflict in result.conflicts)


def test_manual_mapping_takes_precedence_over_alias():
    result = map_columns(
        ["Disp Qty", "Qty Dispensed"],
        "billing",
        manual_mapping={"Disp Qty": "quantity_billed"},
    )
    by_source = {item.source_column: item for item in result.columns}
    assert by_source["Disp Qty"].status == MappingStatus.MANUAL
    assert by_source["Disp Qty"].canonical_field == "quantity_billed"
    assert by_source["Qty Dispensed"].status == MappingStatus.UNMAPPED
    assert not result.conflicts


def test_invalid_manual_destination_is_rejected():
    with pytest.raises(ManualMappingError, match="not a canonical billing field"):
        map_columns(["Disp Qty"], "billing", {"Disp Qty": "made_up_field"})


def test_missing_manual_source_is_rejected():
    with pytest.raises(ManualMappingError, match="does not exist"):
        map_columns(["Disp Qty"], "billing", {"Absent": "quantity_billed"})


def test_two_manual_sources_for_one_destination_conflict():
    result = map_columns(
        ["First Qty", "Second Qty"],
        "billing",
        {"First Qty": "quantity_billed", "Second Qty": "quantity_billed"},
    )
    assert result.ambiguous_count == 2
    assert not result.is_unambiguous


def test_source_to_multiple_fields_is_detected_with_explicit_registry(monkeypatch):
    # Exercise the generic conflict path without changing the production profiles.
    from pharmacy_reconciliation.ingestion import mapping

    profile = mapping.MappingProfile(
        "billing", ("field_a", "field_b"), (),
        {"field_a": ("shared",), "field_b": ("shared",)},
    )
    monkeypatch.setattr(mapping, "PROFILES", {**mapping.PROFILES, "test": profile})
    result = mapping.map_columns(["shared"], "test")
    assert result.columns[0].status == MappingStatus.AMBIGUOUS
    assert result.conflicts[0].code == "source_maps_to_multiple_fields"
