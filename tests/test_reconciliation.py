import pytest
from conftest import inventory_row


@pytest.mark.parametrize(
    ("ndc", "billed", "ordered", "difference", "short", "extra", "status"),
    [
        ("00000000001", 100, 100, 0, 0, 0, "MATCHED"),
        ("00000000002", 80, 50, -30, 30, 0, "SHORT"),
        ("00000000003", 20, 35, 15, 0, 15, "EXTRA"),
        ("00000000004", 25, 0, -25, 25, 0, "SHORT"),
        ("00000000005", 0, 40, 40, 0, 40, "EXTRA"),
    ],
)
def test_inventory_outcomes(july_result, ndc, billed, ordered, difference, short, extra, status):
    row = inventory_row(july_result, ndc)
    assert row["total_billed"] == billed
    assert row["total_ordered"] == ordered
    assert row["net_difference"] == difference
    assert row["short_quantity"] == short
    assert row["extra_quantity"] == extra
    assert row["status"] == status
    assert not (row["short_quantity"] > 0 and row["extra_quantity"] > 0)


def test_multiple_insurers_do_not_duplicate_ordered_quantity(july_result):
    row = inventory_row(july_result, "00000000001")
    assert row["total_billed"] == 100
    assert row["total_ordered"] == 100
    assert len(july_result.inventory.loc[july_result.inventory["ndc"] == "00000000001"]) == 1


def test_date_window_is_inclusive_and_excludes_outside_rows(july_result):
    assert july_result.coverage.start_date.strftime("%Y-%m-%d") == "2026-07-01"
    assert july_result.coverage.end_date.strftime("%Y-%m-%d") == "2026-07-31"
    assert july_result.coverage.billing_rows_total == 6
    assert july_result.coverage.billing_rows_in_period == 5
    assert july_result.coverage.ordering_rows_total == 5
    assert july_result.coverage.ordering_rows_in_period == 4
    assert inventory_row(july_result, "00000000001")["total_billed"] == 100
    assert inventory_row(july_result, "00000000001")["total_ordered"] == 100


def test_invalid_date_window_is_rejected(valid_billing_result, valid_ordering_result):
    from pharmacy_reconciliation.reconciliation.reconcile import reconcile_inventory

    with pytest.raises(ValueError, match="start_date must be on or before end_date"):
        reconcile_inventory(valid_billing_result.data, valid_ordering_result.data, "2026-08-01", "2026-07-01")

