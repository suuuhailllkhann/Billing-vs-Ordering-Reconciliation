"""Run the Phase 2F reduced-feature searches without touching Test."""

import json
from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.reduced_features import tune_reduced_models
from pharmacy_reconciliation.research.tuning import prepare_tuning_partitions

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    observations = pd.read_csv(OBSERVATIONS)
    partitions = prepare_tuning_partitions(observations)
    results = tune_reduced_models(partitions)
    print(json.dumps([result.to_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
