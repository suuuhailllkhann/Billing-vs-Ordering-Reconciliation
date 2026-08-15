"""Print Phase 2L locked-model Validation permutation importance."""

from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.permutation_analysis import (
    locked_logistic_permutation_importance,
)

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    result = locked_logistic_permutation_importance(pd.read_csv(OBSERVATIONS))
    print(result.importances.to_csv(index=False, float_format="%.8f"))


if __name__ == "__main__":
    main()
