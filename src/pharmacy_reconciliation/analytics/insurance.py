"""Insurance-level billed-quantity analytics."""

import pandas as pd


def insurance_billing_summary(billing_in_period: pd.DataFrame) -> pd.DataFrame:
    """Summarize billing without attaching or duplicating order quantities.

    Missing insurance or BIN values are assigned the explicit ``UNKNOWN`` category
    so the insurance totals still equal medication-level billed totals.
    """
    data = billing_in_period.copy()
    data["insurance_name"] = data["insurance_name"].fillna("UNKNOWN")
    data["bin_number"] = data["bin_number"].fillna("UNKNOWN")
    return (
        data.groupby(
            ["ndc", "drug_name", "insurance_name", "bin_number"],
            as_index=False,
            dropna=False,
        )
        .agg(
            total_billed=("quantity_billed", "sum"),
            unique_patient_count=("patient_id", "nunique"),
        )
        .sort_values(["ndc", "drug_name", "insurance_name", "bin_number"], kind="mergesort")
        .reset_index(drop=True)
    )
