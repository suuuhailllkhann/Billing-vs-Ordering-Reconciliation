"""Run the fixed Phase 2C Train/Validation baseline comparison."""

import json
from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.baselines import (
    compare_baselines,
    prepare_baseline_partitions,
)

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    observations = pd.read_csv(OBSERVATIONS)
    partitions = prepare_baseline_partitions(observations)
    results = compare_baselines(partitions)
    print(json.dumps([result.to_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
