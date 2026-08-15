"""Run the locked one-time Phase 2I Test evaluation."""

import json
from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.final_evaluation import evaluate_locked_test

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    result = evaluate_locked_test(pd.read_csv(OBSERVATIONS))
    print(json.dumps({
        "overall": result.overall,
        "seen_patients": result.seen_patients,
        "unseen_patients": result.unseen_patients,
        "validation": result.validation,
    }, indent=2))


if __name__ == "__main__":
    main()
