from pathlib import Path

import pytest

from pharmacy_reconciliation.application.controller import (
    ReconciliationController,
    WorkflowError,
)

SYNTHETIC_DIR = Path(__file__).parents[1] / "data" / "synthetic"


def _ready_controller():
    controller = ReconciliationController()
    controller.load_billing(str(SYNTHETIC_DIR / "billing_export_clean_aliases.csv"))
    controller.load_ordering(str(SYNTHETIC_DIR / "orders_export_messy_headers.csv"))
    assert controller.inputs_ready
    return controller


def test_successful_ingestion_to_reconciliation_flow():
    controller = _ready_controller()
    result = controller.reconcile("2026-07-01", "2026-07-31")
    row = result.inventory.loc[result.inventory["ndc"] == "00000001001"].iloc[0]
    assert row["total_billed"] == 30
    assert row["total_ordered"] == 30
    assert row["status"] == "MATCHED"
    assert not result.insurance.empty
    assert not result.patient_details.empty


def test_invalid_file_is_blocked_before_reconciliation():
    controller = ReconciliationController()
    ingestion = controller.load_billing(
        str(SYNTHETIC_DIR / "billing_export_invalid_values.csv")
    )
    assert not ingestion.ready_for_reconciliation
    controller.load_ordering(str(SYNTHETIC_DIR / "orders_export_messy_headers.csv"))
    with pytest.raises(WorkflowError, match="billing file is not ready"):
        controller.reconcile("2026-07-01", "2026-07-31")


def test_ambiguous_mapping_is_blocked_until_manual_resolution():
    controller = ReconciliationController()
    path = str(SYNTHETIC_DIR / "billing_export_ambiguous.csv")
    ambiguous = controller.load_billing(path)
    assert not ambiguous.ready_for_reconciliation

    resolved = controller.load_billing(path, {"Qty Dispensed": "quantity_billed"})
    assert resolved.ready_for_reconciliation


def test_selected_date_range_reaches_reconciliation_result():
    controller = _ready_controller()
    result = controller.reconcile("2026-07-03", "2026-07-08")
    assert result.reconciliation.coverage.start_date.strftime("%Y-%m-%d") == "2026-07-03"
    assert result.reconciliation.coverage.end_date.strftime("%Y-%m-%d") == "2026-07-08"
    billed_only = result.inventory.loc[result.inventory["ndc"] == "00000001002"].iloc[0]
    assert billed_only["total_billed"] == 20
    assert billed_only["total_ordered"] == 0
    assert billed_only["status"] == "SHORT"


def test_empty_selected_period_returns_empty_views():
    controller = _ready_controller()
    result = controller.reconcile("2025-01-01", "2025-01-31")
    assert result.inventory.empty
    assert result.insurance.empty
    assert result.patient_details.empty
