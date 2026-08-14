"""Billing-period filtering and medication-level aggregation."""

import pandas as pd

MEDICATION_KEY = ["ndc", "drug_name"]


def aggregate_billing(billing: pd.DataFrame) -> pd.DataFrame:
    return (
        billing.groupby(MEDICATION_KEY, as_index=False, dropna=False)
        .agg(total_billed=("quantity_billed", "sum"))
    )

