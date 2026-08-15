"""Record established Phase 2 history in the ignored local MLflow store."""

from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.mlflow_tracking import (
    DEFAULT_TRACKING_DIRECTORY,
    log_modeling_history,
)

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    runs = log_modeling_history(pd.read_csv(OBSERVATIONS))
    print(f"Logged {sum(map(len, runs.values()))} runs to {DEFAULT_TRACKING_DIRECTORY}")


if __name__ == "__main__":
    main()
