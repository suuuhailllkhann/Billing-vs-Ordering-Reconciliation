"""Patient-level detail and medication patient summaries."""

import pandas as pd

PATIENT_DETAIL_COLUMNS = [
    "patient_id", "patient_name", "date_of_birth", "prescription_id", "ndc",
    "drug_name", "insurance_name", "bin_number", "billing_date",
    "quantity_billed", "days_supply", "refills_remaining",
]


def patient_billing_details(billing_in_period: pd.DataFrame) -> pd.DataFrame:
    """Preserve one output row per input billing record and its fill detail."""
    return billing_in_period.loc[:, PATIENT_DETAIL_COLUMNS].copy().reset_index(drop=True)


def patient_medication_summary(billing_in_period: pd.DataFrame) -> pd.DataFrame:
    return (
        billing_in_period.groupby(["ndc", "drug_name"], as_index=False, dropna=False)
        .agg(
            unique_patient_count=("patient_id", "nunique"),
            total_billed=("quantity_billed", "sum"),
        )
        .sort_values(["ndc", "drug_name"], kind="mergesort")
        .reset_index(drop=True)
    )
