"""Ordering-period filtering and medication-level aggregation."""

import pandas as pd

from pharmacy_reconciliation.reconciliation.billing import MEDICATION_KEY


def aggregate_ordering(ordering: pd.DataFrame) -> pd.DataFrame:
    return (
        ordering.groupby(MEDICATION_KEY, as_index=False, dropna=False)
        .agg(total_ordered=("quantity_ordered", "sum"))
    )

