"""Print aggregate Phase 2J locked-model diagnostics without identifiers."""

import json
from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.error_analysis import analyze_locked_errors

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    result = analyze_locked_errors(pd.read_csv(OBSERVATIONS))
    print(json.dumps({
        "confusion": result.confusion,
        "feature_statistics": result.feature_statistics.to_dict(orient="records"),
        "patient_history": result.patient_history.to_dict(orient="records"),
        "outcome_probabilities": result.outcome_probabilities.to_dict(orient="records"),
        "error_confidence": result.error_confidence.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
