"""Run Phase 2G validation threshold analysis without touching Test."""

import json
from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.threshold_analysis import analyze_validation_thresholds

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    results = analyze_validation_thresholds(pd.read_csv(OBSERVATIONS))
    output = [{
        "model_name": result.model_name,
        "pr_auc": result.pr_auc,
        "roc_auc": result.roc_auc,
        "thresholds": result.thresholds.to_dict(orient="records"),
    } for result in results]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
