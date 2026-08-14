from pharmacy_reconciliation.analytics.insurance import insurance_billing_summary
from pharmacy_reconciliation.analytics.patients import (
    patient_billing_details,
    patient_medication_summary,
)


def test_insurance_totals_and_unique_patients(july_result):
    summary = insurance_billing_summary(july_result.billing_in_period)
    medication = summary.loc[summary["ndc"] == "00000000001"]
    assert medication["total_billed"].tolist() == [40, 60]
    assert medication["unique_patient_count"].tolist() == [1, 1]
    assert medication["total_billed"].sum() == 100


def test_insurance_totals_equal_inventory_billing(july_result):
    insurance = insurance_billing_summary(july_result.billing_in_period)
    insurance_totals = insurance.groupby(["ndc", "drug_name"])["total_billed"].sum().sort_index()
    inventory_totals = july_result.inventory.set_index(["ndc", "drug_name"])["total_billed"]
    inventory_totals = inventory_totals.loc[inventory_totals > 0].sort_index()
    assert insurance_totals.to_dict() == inventory_totals.to_dict()
    assert "total_ordered" not in insurance.columns


def test_patient_summary_counts_and_quantities(july_result):
    summary = patient_medication_summary(july_result.billing_in_period)
    row = summary.loc[summary["ndc"] == "00000000001"].iloc[0]
    assert row["unique_patient_count"] == 2
    assert row["total_billed"] == 100


def test_patient_details_preserve_fill_rows(july_result):
    details = patient_billing_details(july_result.billing_in_period)
    medication = details.loc[details["ndc"] == "00000000001"]
    assert len(details) == len(july_result.billing_in_period)
    assert set(medication["patient_id"]) == {"PT-001", "PT-002"}
    assert set(medication["prescription_id"]) == {"RX-001", "RX-002"}
    assert medication.groupby("patient_id")["quantity_billed"].sum().to_dict() == {
        "PT-001": 60,
        "PT-002": 40,
    }

