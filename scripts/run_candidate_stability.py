"""Run Phase 2H candidate stability without touching Test."""

import json
from pathlib import Path

import pandas as pd

from pharmacy_reconciliation.research.candidate_stability import (
    analyze_candidate_validation,
    analyze_train_cv_stability,
    matched_recall_comparison,
)

ROOT = Path(__file__).parents[1]
OBSERVATIONS = ROOT / "data" / "synthetic" / "longitudinal" / "refill_observations.csv"


def main() -> None:
    observations = pd.read_csv(OBSERVATIONS)
    stability = analyze_train_cv_stability(observations)
    validation = analyze_candidate_validation(observations)
    output = {
        "stability": [{
            "model_name": result.model_name,
            "folds": result.folds.to_dict(orient="records"),
            "summary": result.summary.to_dict(orient="records"),
        } for result in stability],
        "validation": [{
            "model_name": result.model_name,
            "pr_auc": result.pr_auc,
            "roc_auc": result.roc_auc,
            "thresholds": result.thresholds.loc[
                result.thresholds["threshold"].isin((0.35, 0.40, 0.45, 0.50, 0.55))
            ].to_dict(orient="records"),
        } for result in validation],
        "matched_recall": matched_recall_comparison(validation).to_dict(orient="records"),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
