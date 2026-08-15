"""Print the locked Logistic Regression standardized odds-ratio table."""

from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.odds_ratio_analysis import locked_logistic_odds_ratios

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    table = locked_logistic_odds_ratios(pd.read_csv(OBSERVATIONS))
    print(table.to_csv(index=False, float_format="%.8f"))


if __name__ == "__main__":
    main()
