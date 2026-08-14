"""Explicit, auditable aliases for generic billing and ordering exports."""

from types import MappingProxyType

BILLING_ALIASES = MappingProxyType({
    "billing_id": ("billing_record_id", "claim_id", "transaction_id", "billing_number"),
    "patient_id": ("patient_identifier", "patient_number", "patient_no", "patient_id_number"),
    "patient_name": ("patient", "patient_full_name", "member_name", "customer_name"),
    "date_of_birth": ("dob", "birth_date", "patient_dob", "patient_birth_date"),
    "prescription_id": ("rx_number", "rx_no", "prescription_number", "script_number"),
    "ndc": ("ndc_number", "product_ndc", "drug_ndc", "ndc_code"),
    "drug_name": ("medication_name", "drug_description", "medication_description", "product_name"),
    "billing_date": ("date_filled", "fill_date", "dispensed_date", "service_date", "claim_date"),
    "insurance_name": ("insurance", "plan", "plan_name", "payer_name", "insurance_plan"),
    "bin_number": ("bin", "bin_no", "processor_bin", "rx_bin"),
    "quantity_billed": ("qty_billed", "billed_qty", "qty_dispensed", "dispensed_quantity", "fill_quantity"),
    "days_supply": ("day_supply", "days_supplied", "supply_days"),
    "refills_remaining": ("refills_left", "remaining_refills", "refill_remaining"),
})

ORDERING_ALIASES = MappingProxyType({
    "order_id": ("order_number", "order_no", "purchase_order_id", "po_number", "transaction_id"),
    "ndc": ("ndc_number", "product_ndc", "drug_ndc", "ndc_code"),
    "drug_name": ("medication_name", "drug_description", "medication_description", "product_name"),
    "ordered_date": ("order_date", "date_ordered", "purchase_date", "po_date"),
    "quantity_ordered": ("qty_ordered", "ordered_qty", "order_quantity", "purchase_quantity"),
})

