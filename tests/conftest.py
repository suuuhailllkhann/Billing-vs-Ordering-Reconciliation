from pathlib import Path

import pandas as pd
import pytest

from pharmacy_reconciliation.ingestion.loaders import load_billing_csv, load_ordering_csv
from pharmacy_reconciliation.research.features import build_supply_end_observation_result
from pharmacy_reconciliation.synthetic.longitudinal import (
    LongitudinalConfig,
    generate_longitudinal_dataset,
)

SYNTHETIC_DIR = Path(__file__).parents[1] / "data" / "synthetic"


@pytest.fixture(scope="session")
def longitudinal_observations() -> pd.DataFrame:
    """Build the authoritative fixed-seed research observations without local files."""
    config = LongitudinalConfig()
    dataset = generate_longitudinal_dataset(config)
    return build_supply_end_observation_result(
        dataset.fills,
        dataset.prescriptions,
        config.end_date,
    ).observations


@pytest.fixture
def valid_billing_result():
    return load_billing_csv(SYNTHETIC_DIR / "billing_valid.csv")


@pytest.fixture
def valid_ordering_result():
    return load_ordering_csv(SYNTHETIC_DIR / "orders_valid.csv")


@pytest.fixture
def july_result(valid_billing_result, valid_ordering_result):
    from pharmacy_reconciliation.reconciliation.reconcile import reconcile_inventory

    return reconcile_inventory(
        valid_billing_result.data,
        valid_ordering_result.data,
        "2026-07-01",
        "2026-07-31",
    )


def inventory_row(result, ndc: str) -> pd.Series:
    rows = result.inventory.loc[result.inventory["ndc"] == ndc]
    assert len(rows) == 1
    return rows.iloc[0]
